#pragma once

// PropertiesPane: a read-only two-column property sheet for whatever is selected in the
// Workspace tree. describe() builds the rows from the engine Document (it is a free function so
// tests can call it without a widget); the pane re-describes the selection whenever the
// selection changes or a command ran, so the sheet is never stale.

#include <QList>
#include <QPair>
#include <QString>
#include <QWidget>

#include "apps/gui/workspace_model.hpp"

class QLabel;
class QTableWidget;

namespace alphashape {
class Document;
}

namespace alphashape::gui {

class GuiApp;

struct PropertySheet {
    QString title;                              ///< "" when nothing (valid) is selected
    QList<QPair<QString, QString>> rows;        ///< (property, value)
};

/// The sheet for `items` (the selection, first = focus item) against `document`. An empty list,
/// or a first item that no longer exists in the document, yields an empty sheet.
PropertySheet describe(const Document& document, const QList<NodeRef>& items);

class PropertiesPane : public QWidget {
    Q_OBJECT

public:
    PropertiesPane(GuiApp& app, WorkspaceSelection& selection, QWidget* parent = nullptr);

    const PropertySheet& sheet() const { return sheet_; }
    QString title() const { return sheet_.title; }
    QList<QPair<QString, QString>> rows() const { return sheet_.rows; }
    /// Value of the row named `name`, or a null string.
    QString value(const QString& name) const;
    /// True when a refresh was requested during a job and is waiting for job_finished.
    bool stale() const { return stale_; }

private:
    void rebuild();

    GuiApp& app_;
    WorkspaceSelection& selection_;
    QLabel* title_ = nullptr;
    QLabel* hint_ = nullptr;
    QTableWidget* table_ = nullptr;
    PropertySheet sheet_;
    bool stale_ = false;
};

}  // namespace alphashape::gui
