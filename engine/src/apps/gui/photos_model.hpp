#pragma once

// PhotosModel: the cameras of the active chunk as a flat icon list for the Photos dock.
//
// One row per camera: the label, the thumbnail, and the badges the user needs to judge a
// photo's state at a glance ("not aligned", "disabled"). The model never holds pointers into
// the document: a row is a small value copy, so the document may be edited or replaced
// underneath it and the model is simply rebuilt.
//
// Thumbnails are decoded off the GUI thread by ThumbnailCache (at most 8 concurrent decodes).
// A row shows a placeholder until its thumbnail is ready; results for a document that has been
// replaced are ignored.
//
// Concurrency rule (see GuiApp): the document is never read while a job is active. refresh()
// then only marks the model stale; the rebuild happens when the job finishes.

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include <QAbstractListModel>
#include <QHash>
#include <QImage>
#include <QObject>
#include <QPixmap>
#include <QString>
#include <QStringList>

namespace alphashape::gui {

class GuiApp;
struct ThumbnailPool;
struct ThumbnailJob;

/// Decodes and scales photo thumbnails on a bounded thread pool, and keeps them in memory by
/// path, revalidating the file's modification time on every lookup.
///
/// Decoding uses the engine's image IO, so every format the project can reference (JPEG, PNG,
/// TIFF) thumbnails the same way. Only QImage crosses the thread boundary; the QPixmap is made
/// on the GUI thread.
class ThumbnailCache : public QObject {
    Q_OBJECT

public:
    /// Longest thumbnail edge; a photo is scaled to fit inside kMaxWidth x kMaxHeight.
    static constexpr int kMaxWidth = 160;
    static constexpr int kMaxHeight = 120;
    /// Decodes allowed to run at once.
    static constexpr int kMaxConcurrent = 8;
    /// Entries kept; the least recently requested beyond this are dropped.
    static constexpr int kMaxEntries = 512;

    explicit ThumbnailCache(QObject* parent = nullptr);
    ~ThumbnailCache() override;

    /// The thumbnail for `path` if it is in memory, else a null pixmap. Queues a decode the
    /// first time a path is asked for, and again after the file changed. Never blocks.
    QPixmap pixmap(const QString& path);

    /// The thumbnail in memory only: no lookup on disk, no decode queued. What a model's
    /// data() returns.
    QPixmap lookup(const QString& path) const;

    /// True when `path` has been decoded and has no thumbnail and never will (missing file,
    /// undecodable). Lets the view say "no preview" instead of "loading".
    bool unavailable(const QString& path) const;

    /// Forget every entry and ignore results still in flight (the document was replaced).
    void clear();

    /// True when a decode has run on a thread other than the GUI thread. Diagnostic, used by
    /// the tests to prove the work really happens off the GUI thread.
    bool decoded_off_gui_thread() const { return off_thread_.load(); }

Q_SIGNALS:
    /// The thumbnail for `path` reached a final state (emitted on the GUI thread).
    void ready(const QString& path);

private:
    friend struct ThumbnailJob;

    struct Entry {
        qint64 mtime = 0;
        QPixmap pixmap;
        bool attempted = false;  ///< a decode finished (successfully or not)
        bool pending = false;    ///< a decode is queued or running
    };

    /// Ask for a decode of `path` as of `mtime` unless one is already on the way.
    void request(const QString& path, qint64 mtime);
    /// A decode finished. `generation` is the cache generation it was started in.
    void deliver(const QString& path, qint64 mtime, quint64 generation, const QImage& image);
    void touch(const QString& path);
    void evict();

    std::unique_ptr<ThumbnailPool> pool_;
    QHash<QString, Entry> entries_;
    QStringList order_;  ///< least recently requested first
    quint64 generation_ = 0;
    std::atomic<bool> off_thread_{false};
};

class PhotosModel : public QAbstractListModel {
    Q_OBJECT

public:
    /// The state badges a row can carry. The model publishes them as roles so a view can pick a
    /// colour from the kind instead of comparing translated text.
    enum Badge {
        NotAlignedBadge = 0,  ///< the camera has no pose yet (amber)
        DisabledBadge,        ///< the user disabled the photo (red)
    };
    Q_ENUM(Badge)

    enum Role {
        CameraIdRole = Qt::UserRole + 1,  ///< int: the camera's id inside the chunk
        ChunkIdRole,                      ///< int: the camera's chunk id
        AlignedRole,                      ///< bool
        EnabledRole,                      ///< bool
        BadgesRole,                       ///< QStringList: the badge texts, in BadgeKindsRole order
        BadgeKindsRole,                   ///< QVariantList of int(Badge)
        ThumbnailStateRole,               ///< int(ThumbnailState)
    };

    /// What the thumbnail for a row is doing.
    enum class ThumbnailState { Pending, Ready, Missing };
    Q_ENUM(ThumbnailState)

    explicit PhotosModel(GuiApp& app, QObject* parent = nullptr);

    int rowCount(const QModelIndex& parent = {}) const override;
    QVariant data(const QModelIndex& index, int role = Qt::DisplayRole) const override;
    Qt::ItemFlags flags(const QModelIndex& index) const override;

    /// Rows before the filter: every camera of the chunk.
    int total_count() const { return static_cast<int>(rows_.size()); }

    /// The active chunk the rows describe, or -1 when the document has no chunks.
    std::int32_t chunk_id() const { return chunk_id_; }
    /// Camera id of `row`, or -1.
    std::int32_t camera_id_at(int row) const;
    /// Row showing `camera_id` under the current filter, or -1.
    int row_for_camera(std::int32_t camera_id) const;
    /// Camera ids in row order (filtered).
    std::vector<std::int32_t> camera_ids() const;
    /// Display label of `row`, or an empty string.
    QString label_at(int row) const;

    /// Case-insensitive substring match on the label. An empty string shows every camera.
    void set_filter(const QString& text);
    QString filter() const { return filter_; }

    /// Re-read the document. Only marks the model stale while a job is active.
    void refresh();
    bool stale() const { return stale_; }

    /// True from beginResetModel() to endResetModel(). A reset clears the view's selection
    /// (QAbstractItemView::reset -> QItemSelectionModel::reset), and a selection model that does
    /// report it -- a proxy, or a Qt that stops blocking signals there -- would emit
    /// selectionChanged mid-rebuild. A pane that mirrors the selection into shared state must
    /// ignore those clears: the model is being rebuilt, the user did not deselect.
    bool resetting() const { return resetting_; }

    ThumbnailCache& thumbnails() { return *thumbnails_; }

Q_SIGNALS:
    /// A structural rebuild finished and the model is consistent again. Every reset (a finished
    /// job, an undo that changed the camera set, a filter change, a replaced document) ends with
    /// this. The reset dropped the view's row selection while the shared selection still names
    /// the cameras, so a pane re-applies the shared selection here.
    void reset_completed();

private:
    struct Row {
        std::int32_t camera_id = -1;
        QString label;
        QString path;
        bool aligned = false;
        bool enabled = false;

        bool operator==(const Row&) const = default;
    };

    void refresh_after_job();
    void rebuild();
    void rebuild_visible();
    void request_thumbnails();
    /// Emit dataChanged for every visible row that shows `path`.
    void thumbnails_changed(const QString& path);
    const Row* row_at(int visible_row) const;
    QString absolute_path(const QString& document_path, const QString& frame_path) const;
    /// beginResetModel()/endResetModel() with resetting_ held across both ends.
    void begin_reset();
    void end_reset();

    GuiApp& app_;
    ThumbnailCache* thumbnails_ = nullptr;
    std::vector<Row> rows_;
    std::vector<int> visible_;  ///< indices into rows_ after the filter
    QString filter_;
    std::int32_t chunk_id_ = -1;
    QString document_path_;
    bool stale_ = false;
    bool resetting_ = false;
};

}  // namespace alphashape::gui
