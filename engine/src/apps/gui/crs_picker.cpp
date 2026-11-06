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
#include <QTabWidget>
#include <QVBoxLayout>

namespace alphashape::gui {
namespace {

QString entry_text(const crs::BuiltinCrs& entry) {
    return QStringLiteral("EPSG:%1  %2")
        .arg(entry.epsg)
        .arg(QString::fromStdString(entry.name));
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
    filter_->setPlaceholderText(tr("Search by EPSG code or name"));
    filter_->setClearButtonEnabled(true);
    layout->addWidget(filter_);
    list_ = new QListWidget(page);
    list_->setObjectName(QStringLiteral("crs.list"));
    for (const crs::BuiltinCrs& entry : crs::builtin_crs_list()) {
        auto* item = new QListWidgetItem(entry_text(entry), list_);
        item->setData(Qt::UserRole, entry.epsg);
    }
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

void CrsPicker::apply_filter(const QString& text) {
    const QString needle = text.trimmed();
    int visible = 0;
    int only_row = -1;
    for (int row = 0; row < list_->count(); ++row) {
        QListWidgetItem* item = list_->item(row);
        const bool matches =
            needle.isEmpty() || item->text().contains(needle, Qt::CaseInsensitive);
        item->setHidden(!matches);
        if (matches) {
            ++visible;
            only_row = row;
        }
    }
    if (visible == 1 && only_row >= 0) {
        list_->setCurrentRow(only_row);
    }
}

int CrsPicker::selected_epsg() const {
    const QListWidgetItem* item = list_->currentItem();
    if (item == nullptr || item->isHidden()) {
        return -1;
    }
    return item->data(Qt::UserRole).toInt();
}

void CrsPicker::refresh_validity() {
    if (error_ == nullptr) {
        return;  // construction: the tabs emit before the error label exists
    }
    error_text_.clear();
    valid_ = false;
    if (tabs_->currentIndex() == 0) {
        if (selected_epsg() < 0) {
            error_text_ = tr("Select a coordinate system.");
        } else if (vertical_->isChecked() &&
                   vertical_name_->text().trimmed().isEmpty()) {
            error_text_ = tr("Name the vertical datum or clear the vertical option.");
        } else {
            valid_ = true;
        }
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
    vertical_->setChecked(false);
    vertical_name_->clear();
    if (cs.is_set() && !cs.is_local() && cs.epsg() > 0) {
        tabs_->setCurrentIndex(0);
        for (int row = 0; row < list_->count(); ++row) {
            if (list_->item(row)->data(Qt::UserRole).toInt() == cs.epsg()) {
                list_->setCurrentRow(row);
                break;
            }
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
        const int epsg = selected_epsg();
        if (epsg < 0) {
            return crs::CoordinateSystem::local();
        }
        crs::CoordinateSystem chosen = crs::CoordinateSystem::from_epsg(epsg);
        if (vertical_->isChecked()) {
            crs::VerticalCrs vertical;
            vertical.kind = crs::VerticalCrs::Kind::GravityRelated;
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
