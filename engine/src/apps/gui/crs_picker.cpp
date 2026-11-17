#include "apps/gui/crs_picker.hpp"

#include <exception>
#include <string>
#include <vector>

#include <QCheckBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QListWidget>
#include <QListWidgetItem>
#include <QPlainTextEdit>
#include <QSignalBlocker>
#include <QStringList>
#include <QTabWidget>
#include <QVBoxLayout>

namespace alphashape::gui {
namespace {

QString entry_text(const crs::BuiltinCrs& entry) {
    return QStringLiteral("EPSG:%1  %2")
        .arg(entry.epsg)
        .arg(QString::fromStdString(entry.name));
}

QString entry_text(const crs::CrsCatalogEntry& entry) {
    QString text = QStringLiteral("EPSG:%1  %2")
                       .arg(entry.epsg)
                       .arg(QString::fromStdString(entry.name));
    if (!entry.area.empty()) {
        text += QStringLiteral("  (%1)").arg(QString::fromStdString(entry.area));
    }
    return text;
}

/// The whole PROJ catalogue with each entry's searchable text, lower-cased
/// once: the filter runs on every keystroke over several thousand entries.
/// When PROJ cannot list its database, the curated list is searched instead.
struct SearchableCatalog {
    std::vector<crs::CrsCatalogEntry> entries;
    std::vector<QString> haystacks;
};

const SearchableCatalog& searchable_catalog() {
    static const SearchableCatalog catalog = [] {
        SearchableCatalog built;
        built.entries = crs::crs_catalog();
        if (built.entries.empty()) {
            for (const crs::BuiltinCrs& entry : crs::builtin_crs_list()) {
                built.entries.push_back(crs::CrsCatalogEntry{entry.epsg, entry.name, entry.kind, {}});
            }
        }
        built.haystacks.reserve(built.entries.size());
        for (const crs::CrsCatalogEntry& entry : built.entries) {
            built.haystacks.push_back(entry_text(entry).toLower());
        }
        return built;
    }();
    return catalog;
}

bool is_curated(int epsg) {
    for (const crs::BuiltinCrs& entry : crs::builtin_crs_list()) {
        if (entry.epsg == epsg) {
            return true;
        }
    }
    return false;
}

}  // namespace

CrsPicker::CrsPicker(QWidget* parent) : QWidget(parent) {
    setObjectName(QStringLiteral("crs.picker"));
    auto* layout = new QVBoxLayout(this);
    layout->setContentsMargins(0, 0, 0, 0);
    tabs_ = new QTabWidget(this);
    tabs_->setObjectName(QStringLiteral("crs.tabs"));
    layout->addWidget(tabs_);
    build_search_tab();
    build_wkt_tab();
    error_ = new QLabel(this);
    error_->setObjectName(QStringLiteral("crs.error"));
    error_->setWordWrap(true);
    layout->addWidget(error_);
    connect(tabs_, &QTabWidget::currentChanged, this, [this](int) { refresh_validity(); });
    refresh_validity();
}

void CrsPicker::build_search_tab() {
    auto* page = new QWidget(this);
    page->setObjectName(QStringLiteral("crs.search_page"));
    auto* layout = new QVBoxLayout(page);
    filter_ = new QLineEdit(page);
    filter_->setObjectName(QStringLiteral("crs.filter"));
    filter_->setPlaceholderText(tr("Search every EPSG system by code, name or area"));
    filter_->setClearButtonEnabled(true);
    layout->addWidget(filter_);
    auto* epsg_row = new QHBoxLayout();
    epsg_row->addWidget(new QLabel(tr("EPSG code:"), page));
    epsg_ = new QLineEdit(page);
    epsg_->setObjectName(QStringLiteral("crs.epsg"));
    epsg_->setPlaceholderText(tr("Any EPSG code the engine knows"));
    epsg_row->addWidget(epsg_, 1);
    layout->addLayout(epsg_row);
    list_ = new QListWidget(page);
    list_->setObjectName(QStringLiteral("crs.list"));
    show_curated();
    layout->addWidget(list_, 1);
    auto* vertical_row = new QHBoxLayout();
    vertical_ = new QCheckBox(tr("Vertical CRS"), page);
    vertical_->setObjectName(QStringLiteral("crs.vertical_enabled"));
    vertical_name_ = new QLineEdit(page);
    vertical_name_->setObjectName(QStringLiteral("crs.vertical_name"));
    vertical_name_->setPlaceholderText(tr("Vertical datum name"));
    vertical_name_->setEnabled(false);
    vertical_row->addWidget(vertical_);
    vertical_row->addWidget(vertical_name_, 1);
    layout->addLayout(vertical_row);
    tabs_->addTab(page, tr("Search"));

    connect(filter_, &QLineEdit::textChanged, this, &CrsPicker::apply_filter);
    connect(epsg_, &QLineEdit::textChanged, this, [this](const QString&) {
        refresh_validity();
        Q_EMIT selection_changed();
    });
    connect(list_, &QListWidget::currentRowChanged, this, [this](int) {
        refresh_validity();
        Q_EMIT selection_changed();
    });
    connect(vertical_, &QCheckBox::toggled, this, [this](bool on) {
        vertical_name_->setEnabled(on);
        refresh_validity();
        Q_EMIT selection_changed();
    });
    connect(vertical_name_, &QLineEdit::textChanged, this, [this](const QString&) {
        refresh_validity();
        Q_EMIT selection_changed();
    });
}

void CrsPicker::build_wkt_tab() {
    auto* page = new QWidget(this);
    page->setObjectName(QStringLiteral("crs.wkt_page"));
    auto* layout = new QVBoxLayout(page);
    auto* hint = new QLabel(tr("Paste a WKT definition of the coordinate system."), page);
    hint->setWordWrap(true);
    layout->addWidget(hint);
    wkt_ = new QPlainTextEdit(page);
    wkt_->setObjectName(QStringLiteral("crs.wkt"));
    wkt_->setPlaceholderText(QStringLiteral("PROJCS[...]"));
    layout->addWidget(wkt_, 1);
    tabs_->addTab(page, tr("WKT"));
    connect(wkt_, &QPlainTextEdit::textChanged, this, [this]() {
        refresh_validity();
        Q_EMIT selection_changed();
    });
}

void CrsPicker::show_curated() {
    const QSignalBlocker blocker(list_);
    list_->clear();
    for (const crs::BuiltinCrs& entry : crs::builtin_crs_list()) {
        auto* item = new QListWidgetItem(entry_text(entry), list_);
        item->setData(Qt::UserRole, entry.epsg);
    }
    list_->setCurrentRow(-1);
}

void CrsPicker::apply_filter(const QString& text) {
    const QString needle = text.trimmed().toLower();
    if (needle.isEmpty()) {
        show_curated();
        refresh_validity();
        Q_EMIT selection_changed();
        return;
    }
    // Every word must appear somewhere in "EPSG:code  name  (area)". A typed
    // code that names a system exactly comes first, then the curated systems,
    // then the rest of the catalogue in code order.
    const QStringList words = needle.split(QLatin1Char(' '), Qt::SkipEmptyParts);
    bool numeric = false;
    const int exact_code = needle.toInt(&numeric);
    const SearchableCatalog& catalog = searchable_catalog();
    std::vector<std::size_t> exact;
    std::vector<std::size_t> curated;
    std::vector<std::size_t> others;
    for (std::size_t i = 0; i < catalog.haystacks.size(); ++i) {
        const QString& haystack = catalog.haystacks[i];
        bool all = true;
        for (const QString& word : words) {
            if (!haystack.contains(word)) {
                all = false;
                break;
            }
        }
        if (!all) {
            continue;
        }
        const int epsg = catalog.entries[i].epsg;
        if (numeric && epsg == exact_code) {
            exact.push_back(i);
        } else if (is_curated(epsg)) {
            curated.push_back(i);
        } else {
            others.push_back(i);
        }
    }
    const std::size_t total = exact.size() + curated.size() + others.size();
    {
        const QSignalBlocker blocker(list_);
        list_->clear();
        std::size_t shown = 0;
        for (const std::vector<std::size_t>* group : {&exact, &curated, &others}) {
            for (const std::size_t i : *group) {
                if (shown == kMaxSearchResults) {
                    break;
                }
                const crs::CrsCatalogEntry& entry = catalog.entries[i];
                auto* item = new QListWidgetItem(entry_text(entry), list_);
                item->setData(Qt::UserRole, entry.epsg);
                ++shown;
            }
        }
        if (total > shown) {
            // Informational only: no code, not selectable.
            auto* more = new QListWidgetItem(
                tr("%n more, refine the search", nullptr, static_cast<int>(total - shown)), list_);
            more->setFlags(Qt::NoItemFlags);
        }
        list_->setCurrentRow(!exact.empty() || total == 1 ? 0 : -1);
    }
    refresh_validity();
    Q_EMIT selection_changed();
}

int CrsPicker::selected_epsg() const {
    const QListWidgetItem* item = list_->currentItem();
    if (item == nullptr || item->isHidden()) {
        return -1;
    }
    const int code = item->data(Qt::UserRole).toInt();
    return code > 0 ? code : -1;
}

int CrsPicker::typed_epsg() const {
    const QString text = epsg_->text().trimmed();
    if (text.isEmpty()) {
        return -1;
    }
    bool ok = false;
    const int code = text.toInt(&ok);
    return ok && code > 0 ? code : -1;
}

void CrsPicker::refresh_validity() {
    if (error_ == nullptr) {
        return;  // construction: the tabs emit before the error label exists
    }
    error_text_.clear();
    valid_ = false;
    if (tabs_->currentIndex() == 0) {
        const int typed = typed_epsg();
        const int code = typed >= 0 ? typed : selected_epsg();
        if (code < 0) {
            error_text_ = tr("Select a coordinate system or type an EPSG code.");
        } else {
            // Typed or picked from the PROJ catalogue: either way the engine
            // has to be able to build it.
            try {
                (void)crs::CoordinateSystem::from_epsg(code);
            } catch (const std::exception& failure) {
                error_text_ = QString::fromUtf8(failure.what());
            }
        }
        if (error_text_.isEmpty() && vertical_->isChecked() &&
            vertical_name_->text().trimmed().isEmpty()) {
            error_text_ = tr("Name the vertical datum or clear the vertical option.");
        }
        valid_ = error_text_.isEmpty();
    } else {
        const QString text = wkt_->toPlainText().trimmed();
        if (text.isEmpty()) {
            error_text_ = tr("Paste a WKT definition.");
        } else {
            try {
                (void)crs::CoordinateSystem::from_wkt(text.toStdString());
                valid_ = true;
            } catch (const std::exception& failure) {
                error_text_ = QString::fromUtf8(failure.what());
            }
        }
    }
    error_->setText(error_text_);
    error_->setVisible(!error_text_.isEmpty());
}

void CrsPicker::set_coordinate_system(const crs::CoordinateSystem& cs) {
    vertical_state_ = cs.vertical();
    {
        const QSignalBlocker blocker(vertical_);
        vertical_->setChecked(vertical_state_.has_value());
    }
    {
        const QSignalBlocker blocker(vertical_name_);
        vertical_name_->setText(vertical_state_.has_value()
                                    ? QString::fromStdString(vertical_state_->name)
                                    : QString());
    }
    vertical_name_->setEnabled(vertical_state_.has_value());
    epsg_->clear();
    {
        const QSignalBlocker blocker(filter_);
        filter_->clear();
    }
    show_curated();
    if (cs.is_set() && !cs.is_local() && cs.epsg() > 0) {
        tabs_->setCurrentIndex(0);
        bool listed = false;
        for (int row = 0; row < list_->count(); ++row) {
            if (list_->item(row)->data(Qt::UserRole).toInt() == cs.epsg()) {
                list_->setCurrentRow(row);
                listed = true;
                break;
            }
        }
        if (!listed) {
            // A code outside the curated list: keep it in the free-form field.
            list_->setCurrentRow(-1);
            epsg_->setText(QString::number(cs.epsg()));
        }
    } else if (cs.is_set() && !cs.is_local()) {
        tabs_->setCurrentIndex(1);
        wkt_->setPlainText(QString::fromStdString(cs.to_wkt()));
    } else {
        tabs_->setCurrentIndex(0);
        list_->setCurrentRow(-1);
    }
    refresh_validity();
}

bool CrsPicker::is_valid() const { return valid_; }

QString CrsPicker::error() const { return error_text_; }

crs::CoordinateSystem CrsPicker::coordinate_system() const {
    if (tabs_->currentIndex() == 0) {
        const int typed = typed_epsg();
        const int epsg = typed >= 0 ? typed : selected_epsg();
        if (epsg < 0) {
            return crs::CoordinateSystem::local();
        }
        crs::CoordinateSystem chosen = crs::CoordinateSystem::from_epsg(epsg);
        if (vertical_->isChecked()) {
            crs::VerticalCrs vertical = vertical_state_.value_or(crs::VerticalCrs{});
            if (!vertical.is_set()) {
                vertical.kind = crs::VerticalCrs::Kind::GravityRelated;
            }
            vertical.name = vertical_name_->text().trimmed().toStdString();
            chosen = chosen.with_vertical(vertical);
        }
        return chosen;
    }
    try {
        return crs::CoordinateSystem::from_wkt(
            wkt_->toPlainText().trimmed().toStdString());
    } catch (const std::exception&) {
        return crs::CoordinateSystem::local();
    }
}

}  // namespace alphashape::gui
