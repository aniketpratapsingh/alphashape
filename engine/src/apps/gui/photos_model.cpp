#include "apps/gui/photos_model.hpp"

#include <algorithm>
#include <exception>
#include <utility>

#include <QBrush>
#include <QDir>
#include <QFileInfo>
#include <QGuiApplication>
#include <QImage>
#include <QMetaObject>
#include <QPointer>
#include <QRunnable>
#include <QThreadPool>

#include "apps/gui/app.hpp"
#include "base/utils/log.hpp"
#include "common/core/chunk.hpp"
#include "common/core/document.hpp"
#include "common/core/frame.hpp"
#include "image/image.hpp"
#include "image/io/image_io.hpp"

namespace alphashape::gui {

namespace {

bool is_aligned(const Camera& camera) { return camera.pose != Transform::Identity(); }

/// Engine Image -> QImage at thumbnail size, aspect preserved, never upscaled.
/// Runs on a pool thread: it touches no Qt widget state.
QImage make_thumbnail(const std::string& path) {
    const Image decoded = read_image(path);
    if (decoded.empty()) {
        return {};
    }
    const int width = decoded.width();
    const int height = decoded.height();
    const int channels = decoded.channels();
    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
        return {};
    }
    const QImage::Format format =
        channels == 1 ? QImage::Format_Grayscale8 : QImage::Format_RGB888;
    // QImage does not own the bytes; copy() detaches them before the engine Image dies, and
    // scaled() before copy() keeps the intermediate small.
    const QImage view(decoded.data(), width, height, width * channels, format);
    return view.scaled(ThumbnailCache::kMaxWidth, ThumbnailCache::kMaxHeight, Qt::KeepAspectRatio,
                       Qt::SmoothTransformation)
        .copy();
}

}  // namespace

// --- ThumbnailPool -------------------------------------------------------------------------

/// The bounded pool. Kept in a struct of its own so the header needs no QThreadPool include.
struct ThumbnailPool {
    QThreadPool pool;

    ThumbnailPool() {
        pool.setMaxThreadCount(ThumbnailCache::kMaxConcurrent);
        // Decodes are short and independent; keeping the threads around for a while avoids
        // paying the start-up cost again when a folder of photos is scrolled.
        pool.setExpiryTimeout(30000);
    }
};

/// One decode. Runs on the pool thread; posts the result back to the GUI thread.
struct ThumbnailJob : QRunnable {
    ThumbnailJob(QPointer<ThumbnailCache> owner, QString path, qint64 mtime, quint64 generation)
        : owner_(std::move(owner)),
          path_(std::move(path)),
          mtime_(mtime),
          generation_(generation) {}

    void run() override {
        ThumbnailCache* owner = owner_.data();
        if (owner != nullptr) {
            owner->off_thread_.store(true);
        }
        QImage thumbnail;
        try {
            thumbnail = make_thumbnail(path_.toStdString());
        } catch (const std::exception& error) {
            alphashape::log(LogLevel::Debug)
                << "thumbnail for '" << path_.toStdString() << "' failed: " << error.what();
        } catch (...) {
            alphashape::log(LogLevel::Debug) << "thumbnail for '" << path_.toStdString()
                                             << "' failed with an unknown error";
        }
        if (owner == nullptr) {
            return;  // the cache went away while the decode ran
        }
        // Local copies: the lambda must not depend on this job outliving the queued call. The
        // guarded pointer makes a cache destroyed before the call is delivered a no-op, and the
        // invoke targets it as the context object as well.
        const QString path = path_;
        const qint64 mtime = mtime_;
        const quint64 generation = generation_;
        QMetaObject::invokeMethod(
            owner,
            [guard = QPointer<ThumbnailCache>(owner), path, mtime, generation,
             thumbnail = std::move(thumbnail)]() {
                if (ThumbnailCache* cache = guard.data()) {
                    cache->deliver(path, mtime, generation, thumbnail);
                }
            },
            Qt::QueuedConnection);
    }

    QPointer<ThumbnailCache> owner_;
    QString path_;
    qint64 mtime_ = 0;
    quint64 generation_ = 0;
};

// --- ThumbnailCache ------------------------------------------------------------------------

ThumbnailCache::ThumbnailCache(QObject* parent)
    : QObject(parent), pool_(std::make_unique<ThumbnailPool>()) {}

ThumbnailCache::~ThumbnailCache() {
    // Jobs post back through a guarded pointer, so the cache could die first; waiting is still
    // the clean shutdown (and keeps the off-thread flag meaningful for a torn-down cache).
    pool_->pool.clear();
    pool_->pool.waitForDone();
}

void ThumbnailCache::touch(const QString& path) {
    order_.removeAll(path);
    order_.append(path);
    evict();
}

void ThumbnailCache::evict() {
    while (order_.size() > kMaxEntries) {
        entries_.remove(order_.takeFirst());
    }
}

QPixmap ThumbnailCache::lookup(const QString& path) const {
    const auto it = entries_.constFind(path);
    return it != entries_.constEnd() ? it->pixmap : QPixmap();
}

bool ThumbnailCache::unavailable(const QString& path) const {
    const auto it = entries_.constFind(path);
    return it != entries_.constEnd() && it->attempted && it->pixmap.isNull();
}

QPixmap ThumbnailCache::pixmap(const QString& path) {
    if (path.isEmpty()) {
        return {};
    }
    const QFileInfo info(path);
    const qint64 mtime = info.exists() ? info.lastModified().toMSecsSinceEpoch() : -1;
    auto it = entries_.find(path);
    if (it != entries_.end()) {
        if (it->mtime == mtime) {
            touch(path);
            return it->pixmap;
        }
        entries_.erase(it);  // the file changed (or appeared): decode it again
    }
    if (mtime < 0) {
        Entry missing;
        missing.mtime = -1;
        missing.attempted = true;
        entries_.insert(path, missing);
        touch(path);
        return {};
    }
    request(path, mtime);
    return {};
}

void ThumbnailCache::request(const QString& path, qint64 mtime) {
    Entry& entry = entries_[path];
    if (entry.pending) {
        return;
    }
    entry.mtime = mtime;
    entry.pending = true;
    touch(path);
    // The job holds a guarded pointer back to this object, so a destroyed cache is safe.
    pool_->pool.start(new ThumbnailJob(QPointer<ThumbnailCache>(this), path, mtime, generation_));
}

void ThumbnailCache::deliver(const QString& path, qint64 mtime, quint64 generation,
                             const QImage& image) {
    if (generation != generation_) {
        return;  // the cache was cleared while this decode ran
    }
    auto it = entries_.find(path);
    if (it == entries_.end() || it->mtime != mtime) {
        return;  // the file changed under us; a newer decode is on its way
    }
    it->pending = false;
    it->attempted = true;
    it->pixmap = image.isNull() ? QPixmap() : QPixmap::fromImage(image);
    Q_EMIT ready(path);
}

void ThumbnailCache::clear() {
    ++generation_;
    entries_.clear();
    order_.clear();
}

// --- PhotosModel ---------------------------------------------------------------------------

PhotosModel::PhotosModel(GuiApp& app, QObject* parent) : QAbstractListModel(parent), app_(app) {
    thumbnails_ = new ThumbnailCache(this);
    connect(thumbnails_, &ThumbnailCache::ready, this, &PhotosModel::thumbnails_changed);
    connect(&app_, &GuiApp::document_replaced, this, [this]() {
        thumbnails_->clear();  // results for the old document are dropped
        rebuild();
    });
    connect(&app_, &GuiApp::undo_changed, this, &PhotosModel::refresh);
    // A task may add, remove or relabel cameras without going through a command; every finished
    // job rebuilds, whatever its outcome.
    connect(&app_.jobs(), &JobManager::job_finished, this,
            [this](quint64, JobManager::State, const QString&) { refresh_after_job(); });
    rebuild();
}

void PhotosModel::refresh_after_job() {
    if (app_.jobs().busy()) {
        stale_ = true;
        return;
    }
    stale_ = false;
    rebuild();
    Q_EMIT reset_after_job();
}

void PhotosModel::refresh() {
    if (app_.jobs().busy()) {
        stale_ = true;
        return;
    }
    stale_ = false;
    rebuild();
}

void PhotosModel::rebuild() {
    if (app_.jobs().busy()) {
        stale_ = true;
        return;
    }
    stale_ = false;
    std::vector<Row> rows;
    std::int32_t chunk_id = -1;
    QString document_path;
    app_.read_document([&](const Document& document) {
        document_path = QString::fromStdString(document.path());
        // Document has no const active_chunk(); resolve it the way the Properties pane does.
        const Chunk* chunk = nullptr;
        const Chunk* first = nullptr;
        for (const auto& workspace : document.workspaces()) {
            for (const auto& candidate : workspace->chunks()) {
                if (first == nullptr) {
                    first = candidate.get();
                }
                if (candidate->id() == document.active_chunk_id()) {
                    chunk = candidate.get();
                }
            }
        }
        if (chunk == nullptr) {
            chunk = first;
        }
        if (chunk == nullptr) {
            return;
        }
        chunk_id = chunk->id();
        rows.reserve(chunk->cameras().size());
        for (const Camera& camera : chunk->cameras()) {
            Row row;
            row.camera_id = camera.id;
            row.label = QString::fromStdString(camera.label);
            row.aligned = is_aligned(camera);
            row.enabled = camera.enabled;
            for (const auto& frame : chunk->frames()) {
                if (frame && frame->camera_id() == camera.id) {
                    row.path =
                        absolute_path(document_path, QString::fromStdString(frame->path()));
                    break;
                }
            }
            rows.push_back(std::move(row));
        }
    });

    const bool same_document = chunk_id == chunk_id_ && document_path == document_path_;
    if (same_document && rows == rows_) {
        return;  // nothing visible changed; keep the view's scroll position and selection
    }

    // Same cameras, same labels, same files: only the badges moved (an enable/disable round
    // trip). Update in place so the list keeps its selection and scroll position.
    bool same_identity = same_document && rows.size() == rows_.size();
    for (std::size_t i = 0; same_identity && i < rows.size(); ++i) {
        same_identity = rows[i].camera_id == rows_[i].camera_id && rows[i].label == rows_[i].label &&
                        rows[i].path == rows_[i].path;
    }
    if (same_identity) {
        for (std::size_t i = 0; i < rows.size(); ++i) {
            if (rows[i].aligned == rows_[i].aligned && rows[i].enabled == rows_[i].enabled) {
                continue;
            }
            rows_[i] = rows[i];
            for (std::size_t v = 0; v < visible_.size(); ++v) {
                if (visible_[v] != static_cast<int>(i)) {
                    continue;
                }
                const QModelIndex changed = index(static_cast<int>(v), 0);
                Q_EMIT dataChanged(changed, changed,
                                   {Qt::ToolTipRole, Qt::ForegroundRole, AlignedRole,
                                    EnabledRole, BadgesRole});
            }
        }
        return;
    }

    if (!same_document) {
        thumbnails_->clear();  // thumbnails are keyed by path but belong to this document
    }
    rows_ = std::move(rows);
    chunk_id_ = chunk_id;
    document_path_ = document_path;
    beginResetModel();
    rebuild_visible();
    endResetModel();
    request_thumbnails();
}

void PhotosModel::rebuild_visible() {
    visible_.clear();
    const QString needle = filter_.trimmed();
    for (std::size_t i = 0; i < rows_.size(); ++i) {
        if (needle.isEmpty() || rows_[i].label.contains(needle, Qt::CaseInsensitive)) {
            visible_.push_back(static_cast<int>(i));
        }
    }
}

void PhotosModel::request_thumbnails() {
    for (const Row& row : rows_) {
        if (!row.path.isEmpty()) {
            (void)thumbnails_->pixmap(row.path);  // queues a decode on first use
        }
    }
}

void PhotosModel::set_filter(const QString& text) {
    if (text == filter_) {
        return;
    }
    filter_ = text;
    beginResetModel();
    rebuild_visible();
    endResetModel();
}

void PhotosModel::thumbnails_changed(const QString& path) {
    for (std::size_t i = 0; i < visible_.size(); ++i) {
        const Row& row = rows_[static_cast<std::size_t>(visible_[i])];
        if (row.path != path) {
            continue;
        }
        const QModelIndex changed = index(static_cast<int>(i), 0);
        Q_EMIT dataChanged(changed, changed, {Qt::DecorationRole, ThumbnailStateRole});
    }
}

QString PhotosModel::absolute_path(const QString& document_path, const QString& frame_path) const {
    if (frame_path.isEmpty()) {
        return {};
    }
    if (QFileInfo(frame_path).isAbsolute() || document_path.isEmpty()) {
        return frame_path;
    }
    // A relative frame path is relative to the project file, as the project format stores it.
    return QDir(QFileInfo(document_path).absolutePath()).filePath(frame_path);
}

int PhotosModel::rowCount(const QModelIndex& parent) const {
    return parent.isValid() ? 0 : static_cast<int>(visible_.size());
}

const PhotosModel::Row* PhotosModel::row_at(int visible_row) const {
    if (visible_row < 0 || static_cast<std::size_t>(visible_row) >= visible_.size()) {
        return nullptr;
    }
    return &rows_[static_cast<std::size_t>(visible_[static_cast<std::size_t>(visible_row)])];
}

int PhotosModel::camera_id_at(int row) const {
    const Row* found = row_at(row);
    return found != nullptr ? found->camera_id : -1;
}

int PhotosModel::row_for_camera(std::int32_t camera_id) const {
    for (std::size_t i = 0; i < visible_.size(); ++i) {
        if (rows_[static_cast<std::size_t>(visible_[i])].camera_id == camera_id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

std::vector<std::int32_t> PhotosModel::camera_ids() const {
    std::vector<std::int32_t> ids;
    ids.reserve(visible_.size());
    for (const int row : visible_) {
        ids.push_back(rows_[static_cast<std::size_t>(row)].camera_id);
    }
    return ids;
}

QString PhotosModel::label_at(int row) const {
    const Row* found = row_at(row);
    return found != nullptr ? found->label : QString();
}

QVariant PhotosModel::data(const QModelIndex& index, int role) const {
    const Row* row = row_at(index.row());
    if (row == nullptr || index.column() != 0) {
        return {};
    }
    switch (role) {
        case Qt::DisplayRole:
            return row->label;
        case Qt::DecorationRole: {
            if (row->path.isEmpty()) {
                return QVariant();
            }
            const QPixmap thumbnail = thumbnails_->lookup(row->path);
            // A null icon is what the delegate draws as "no preview yet"; the text stays.
            return thumbnail.isNull() ? QVariant() : QVariant(thumbnail);
        }
        case Qt::ToolTipRole: {
            QStringList parts{row->label};
            if (!row->path.isEmpty()) {
                parts << row->path;
            }
            if (!row->enabled) {
                parts << tr("disabled");
            }
            if (!row->aligned) {
                parts << tr("not aligned");
            }
            return parts.join(QLatin1Char('\n'));
        }
        case Qt::ForegroundRole:
            // A disabled photo is greyed, the same cue the Workspace tree uses.
            if (!row->enabled) {
                return QBrush(QGuiApplication::palette().color(QPalette::Disabled,
                                                               QPalette::Text));
            }
            return {};
        case CameraIdRole:
            return row->camera_id;
        case ChunkIdRole:
            return chunk_id_;
        case AlignedRole:
            return row->aligned;
        case EnabledRole:
            return row->enabled;
        case BadgesRole: {
            QStringList badges;
            if (!row->aligned) {
                badges << tr("not aligned");
            }
            if (!row->enabled) {
                badges << tr("disabled");
            }
            return badges;
        }
        case ThumbnailStateRole:
            if (row->path.isEmpty() || thumbnails_->unavailable(row->path)) {
                return static_cast<int>(ThumbnailState::Missing);
            }
            if (!thumbnails_->lookup(row->path).isNull()) {
                return static_cast<int>(ThumbnailState::Ready);
            }
            return static_cast<int>(ThumbnailState::Pending);
        default:
            return {};
    }
}

Qt::ItemFlags PhotosModel::flags(const QModelIndex& index) const {
    if (!index.isValid()) {
        return Qt::NoItemFlags;
    }
    return Qt::ItemIsEnabled | Qt::ItemIsSelectable;
}

}  // namespace alphashape::gui
