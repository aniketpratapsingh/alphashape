#pragma once

// Test seam for the dialogs' modal paths.
//
// The dialogs report a rejected edit with QMessageBox and, when a CRS slot
// changes under existing references, ask the user what to do. Both go through
// these hooks so a GUI test never opens a modal dialog: the test installs a
// hook that records the call and answers instead. The defaults are the real
// Qt dialogs, so the application behaviour is unchanged.

#include <functional>

#include <QFileDialog>
#include <QMessageBox>
#include <QPushButton>
#include <QString>
#include <QWidget>

namespace alphashape::gui {

/// What to do when a CRS slot change would make existing reference values
/// mean something new.
enum class ReferenceChangeChoice {
    Convert,  ///< Convert the stored values into the new slot's CRS.
    Relabel,  ///< Keep the numbers and let them denote the new CRS.
    Cancel,   ///< Leave the settings unchanged.
};

struct ModalHooks {
    /// Report a rejected edit. Default: QMessageBox::critical.
    std::function<void(QWidget*, const QString&, const QString&)> error;
    /// Ask how to resolve a CRS change over existing references. Default: a
    /// three-button message box. The first string is the slot name.
    std::function<ReferenceChangeChoice(QWidget*, const QString&)>
        choose_reference_change;
    /// Ask where to save a generated file (title, suggested name, filter).
    /// Default: QFileDialog::getSaveFileName. Empty = cancelled.
    std::function<QString(QWidget*, const QString&, const QString&, const QString&)>
        save_file;
};

/// The real Qt dialogs.
inline ModalHooks default_modal_hooks() {
    return ModalHooks{
        [](QWidget* parent, const QString& title, const QString& text) {
            QMessageBox::critical(parent, title, text);
        },
        [](QWidget* parent, const QString& slot) {
            QMessageBox box(QMessageBox::Question, QObject::tr("Coordinate Systems"),
                            QObject::tr("The %1 coordinate system changes, and existing "
                                        "reference values were recorded in the previous "
                                        "one.\n\nConvert them into the new system, or keep "
                                        "the numbers and relabel them.\n\n"
                                        "Tools -> Convert Reference performs the same "
                                        "conversion later.")
                                .arg(slot),
                            QMessageBox::NoButton, parent);
            QPushButton* convert = box.addButton(
                QObject::tr("Convert values"), QMessageBox::AcceptRole);
            QPushButton* relabel = box.addButton(
                QObject::tr("Keep values (relabel)"), QMessageBox::DestructiveRole);
            box.addButton(QObject::tr("Cancel"), QMessageBox::RejectRole);
            box.setDefaultButton(convert);
            box.exec();
            if (box.clickedButton() == convert) {
                return ReferenceChangeChoice::Convert;
            }
            if (box.clickedButton() == relabel) {
                return ReferenceChangeChoice::Relabel;
            }
            return ReferenceChangeChoice::Cancel;
        },
        [](QWidget* parent, const QString& title, const QString& suggested,
           const QString& filter) {
            return QFileDialog::getSaveFileName(parent, title, suggested, filter);
        },
    };
}

/// The process-wide hooks. A GUI test installs recording hooks in SetUp and
/// puts the defaults back in TearDown.
inline ModalHooks& modal_hooks() {
    static ModalHooks hooks = default_modal_hooks();
    return hooks;
}

inline void reset_modal_hooks() { modal_hooks() = default_modal_hooks(); }

/// Report a rejected edit through the hook.
inline void report_error(QWidget* parent, const QString& title,
                         const QString& text) {
    modal_hooks().error(parent, title, text);
}

}  // namespace alphashape::gui
