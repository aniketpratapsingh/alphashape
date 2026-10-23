#include "apps/gui/task_dialog.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <exception>
#include <utility>

#include <QCheckBox>
#include <QComboBox>
#include <QDialogButtonBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QRegularExpression>
#include <QSettings>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStyle>
#include <QVBoxLayout>

namespace alphashape::gui {

namespace {

constexpr double kFloatLimit = 1.0e9;
constexpr int kFloatDecimals = 6;

QString q(const std::string& text) { return QString::fromStdString(text); }

bool is_unset(const ParamValue& value) { return std::holds_alternative<std::monostate>(value); }

/// The accepted values of a String parameter: the schema's own list when it declares one,
/// otherwise the words the doc quotes (the older, weaker convention). Empty = free text.
QStringList accepted_values(const ParamSpec& spec) {
    if (spec.type != ParamType::String) {
        return {};
    }
    if (!spec.choices.empty()) {
        QStringList declared;
        declared.reserve(static_cast<int>(spec.choices.size()));
        for (const std::string& choice : spec.choices) {
            declared << q(choice);
        }
        return declared;
    }
    if (!std::holds_alternative<std::string>(spec.default_value)) {
        return {};
    }
    const QStringList scraped = TaskDialog::choices_from_doc(spec.doc);
    return scraped.contains(q(std::get<std::string>(spec.default_value))) ? scraped
                                                                         : QStringList{};
}

bool has_choices(const ParamSpec& spec) { return !accepted_values(spec).isEmpty(); }

/// The inclusive integer bound the schema declares, or `fallback` when it declares none.
int declared_bound(const ParamValue& bound, int fallback) {
    if (const auto* value = std::get_if<std::int64_t>(&bound)) {
        return static_cast<int>(std::clamp<std::int64_t>(*value, INT_MIN, INT_MAX));
    }
    return fallback;
}

/// The power of two closest to `value` (ties round up); at least 1.
int nearest_power_of_two(int value) {
    if (value <= 1) {
        return 1;
    }
    int lower = 1;
    while (lower <= value / 2) {
        lower *= 2;
    }
    const int upper = lower <= INT_MAX / 2 ? lower * 2 : lower;
    return (value - lower) <= (upper - value) ? lower : upper;
}

/// The legal range for a power-of-two parameter: the smallest power of two at or above
/// `lower`, and the largest at or below `upper`.
std::pair<int, int> power_of_two_range(int lower, int upper) {
    // The first legal value at or above `lower`, by doubling from 1 (never by incrementing
    // from `lower`, which would overshoot the bound when no power of two fits between them).
    int low = 1;
    while (low < lower && low <= INT_MAX / 2) {
        low *= 2;
    }
    // The last legal value at or below `upper`. Clear the bits under the highest set one:
    // halving an odd `upper` would not converge on a power of two.
    int high = std::max(1, upper);
    while ((high & (high - 1)) != 0) {
        high &= high - 1;
    }
    return {low, std::max(high, low)};
}

/// A spin box for a parameter whose legal values are the powers of two. stepBy is overridden
/// rather than the value signal watched, because snapping on every change would undo the very
/// step that asked for the next legal value (stepUp from 4 moves to 5, which snaps back to 4).
/// Here the arrows double and halve within the range, and a typed value is snapped when it is
/// committed, so the box only ever holds a power of two.
class PowerOfTwoSpinBox : public QSpinBox {
public:
    explicit PowerOfTwoSpinBox(QWidget* parent) : QSpinBox(parent) {}

    void stepBy(int steps) override {
        int value = this->value();
        for (int i = 0; i < std::abs(steps); ++i) {
            if (steps > 0) {
                if (value > maximum() / 2) {
                    break;
                }
                value *= 2;
            } else {
                if (value <= minimum() || value / 2 < minimum()) {
                    break;
                }
                value /= 2;
            }
        }
        setValue(value);
    }

protected:
    int valueFromText(const QString& text) const override {
        bool ok = false;
        const int typed = text.trimmed().toInt(&ok);
        return ok ? nearest_power_of_two(typed) : value();
    }
};

TaskDialog::EditorKind kind_for(const ParamSpec& spec) {
    switch (spec.type) {
        case ParamType::Bool:
            return TaskDialog::EditorKind::CheckBox;
        case ParamType::Int:
            return TaskDialog::EditorKind::SpinBox;
        case ParamType::Float:
            return TaskDialog::EditorKind::DoubleSpinBox;
        case ParamType::String:
            if (spec.name == "path") {
                return TaskDialog::EditorKind::PathEdit;
            }
            return has_choices(spec) ? TaskDialog::EditorKind::ComboBox
                                     : TaskDialog::EditorKind::LineEdit;
        case ParamType::StringList:
            return TaskDialog::EditorKind::ListEdit;
    }
    return TaskDialog::EditorKind::LineEdit;
}

std::string list_text(const std::vector<std::string>& items) {
    return format_param_value(ParamValue(items));
}

/// True when the parameter has to be filled in before the task can run.
///
/// The schema mostly says so itself (no default at all, or a doc that starts a sentence with
/// "Required"). The exception is the file-path parameter, which has an empty-string default so
/// the schema still type-checks but which every task that declares one rejects when empty: an
/// empty path would be a guaranteed failure, so it is asked for up front.
bool is_required(const ParamSpec& spec) {
    if (is_unset(spec.default_value)) {
        return true;
    }
    if (spec.type != ParamType::String) {
        return false;
    }
    if (spec.doc.find("Required") != std::string::npos) {
        return true;
    }
    const auto* text = std::get_if<std::string>(&spec.default_value);
    return spec.name == "path" && text != nullptr && text->empty();
}

}  // namespace

QString TaskDialog::display_name(const std::string& param) {
    QString text = q(param).replace(QLatin1Char('_'), QLatin1Char(' ')).trimmed();
    if (!text.isEmpty()) {
        text[0] = text[0].toUpper();
    }
    return text;
}

QString TaskDialog::task_title(const std::string& task_name) {
    const QString name = q(task_name);
    QString out;
    for (int i = 0; i < name.size(); ++i) {
        const QChar c = name[i];
        if (i > 0 && c.isUpper()) {
            const bool prev_lower = name[i - 1].isLower() || name[i - 1].isDigit();
            const bool acronym_end =
                name[i - 1].isUpper() && i + 1 < name.size() && name[i + 1].isLower();
            if (prev_lower || acronym_end) {
                out += QLatin1Char(' ');
            }
        }
        out += c;
    }
    return out;
}

QStringList TaskDialog::choices_from_doc(const std::string& doc) {
    static const QRegularExpression quoted(QStringLiteral("([\"'])([A-Za-z][A-Za-z0-9_]*)\\1"));
    QStringList found;
    auto it = quoted.globalMatch(q(doc));
    while (it.hasNext()) {
        const QString word = it.next().captured(2);
        if (!found.contains(word)) {
            found << word;
        }
    }
    return found.size() >= 2 ? found : QStringList{};
}

QString TaskDialog::settings_key(const std::string& task_name, const std::string& param) {
    return QStringLiteral("taskdialog/%1/%2").arg(q(task_name), q(param));
}

TaskDialog::TaskDialog(const std::string& task_name, const TaskRegistry& registry,
                       QSettings& settings, QWidget* parent)
    : QDialog(parent), task_name_(task_name), settings_(settings) {
    const TaskEntry& entry = registry.require(task_name);
    setObjectName(QStringLiteral("dialog.task"));
    setWindowTitle(task_title(task_name));
    for (const ParamSpec& spec : entry.schema.specs()) {
        Field field;
        field.spec = spec;
        field.kind = kind_for(spec);
        field.required = is_required(spec);
        fields_.push_back(std::move(field));
    }
    build();
    load_remembered();
}

TaskDialog::Field* TaskDialog::find(const std::string& param) {
    for (Field& field : fields_) {
        if (field.spec.name == param) {
            return &field;
        }
    }
    return nullptr;
}

const TaskDialog::Field* TaskDialog::find(const std::string& param) const {
    return const_cast<TaskDialog*>(this)->find(param);
}

void TaskDialog::build() {
    auto* root = new QVBoxLayout(this);

    auto* general_form = new QFormLayout();
    general_form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    auto* advanced_form = new QFormLayout();
    advanced_form->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);

    bool any_advanced = false;
    for (std::size_t i = 0; i < fields_.size(); ++i) {
        const bool is_advanced = i >= kGeneralCount;
        any_advanced = any_advanced || is_advanced;
        add_field(fields_[i].spec, is_advanced, is_advanced ? advanced_form : general_form);
    }

    if (any_advanced) {
        auto* general_box = new QGroupBox(tr("General"), this);
        general_box->setObjectName(QStringLiteral("group.general"));
        general_box->setLayout(general_form);
        root->addWidget(general_box);

        advanced_group_ = new QGroupBox(tr("Advanced"), this);
        advanced_group_->setObjectName(QStringLiteral("group.advanced"));
        advanced_group_->setCheckable(true);
        advanced_group_->setChecked(false);
        auto* content = new QWidget(advanced_group_);
        content->setObjectName(QStringLiteral("group.advanced.content"));
        content->setLayout(advanced_form);
        content->setVisible(false);
        auto* group_layout = new QVBoxLayout(advanced_group_);
        group_layout->addWidget(content);
        connect(advanced_group_, &QGroupBox::toggled, content, &QWidget::setVisible);
        root->addWidget(advanced_group_);
    } else {
        root->addLayout(general_form);
    }

    buttons_ = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel | QDialogButtonBox::RestoreDefaults, this);
    buttons_->setObjectName(QStringLiteral("task.buttons"));
    buttons_->button(QDialogButtonBox::RestoreDefaults)->setText(tr("Reset to defaults"));
    buttons_->button(QDialogButtonBox::RestoreDefaults)->setObjectName(QStringLiteral("task.reset"));
    buttons_->button(QDialogButtonBox::Ok)->setObjectName(QStringLiteral("task.ok"));
    buttons_->button(QDialogButtonBox::Cancel)->setObjectName(QStringLiteral("task.cancel"));
    connect(buttons_, &QDialogButtonBox::accepted, this, &TaskDialog::accept);
    connect(buttons_, &QDialogButtonBox::rejected, this, &QDialog::reject);
    connect(buttons_->button(QDialogButtonBox::RestoreDefaults), &QPushButton::clicked, this,
            &TaskDialog::reset_to_defaults);
    root->addWidget(buttons_);
}

void TaskDialog::add_field(const ParamSpec& spec, bool is_advanced, QFormLayout* form) {
    Field* field = find(spec.name);
    const std::size_t index = static_cast<std::size_t>(field - fields_.data());
    field->advanced = is_advanced;
    const QString object = q(spec.name);
    const QString tip = q(spec.doc);

    auto* host = new QWidget(this);
    auto* column = new QVBoxLayout(host);
    column->setContentsMargins(0, 0, 0, 0);
    column->setSpacing(2);

    auto mark_dirty = [this, index]() { fields_[index].dirty = true; };

    switch (field->kind) {
        case EditorKind::CheckBox: {
            auto* box = new QCheckBox(host);
            field->editor = box;
            connect(box, &QCheckBox::toggled, this, mark_dirty);
            column->addWidget(box);
            break;
        }
        case EditorKind::SpinBox: {
            auto* spin = spec.power_of_two ? static_cast<QSpinBox*>(new PowerOfTwoSpinBox(host))
                                           : new QSpinBox(host);
            // The declared range wins. Without one, the old heuristic: a negative default
            // suggests the parameter takes negative values, so the floor is the default and
            // anything below 0 stays out of reach; a non-negative default means 0 is a
            // sensible floor.
            const std::int64_t default_value =
                std::holds_alternative<std::int64_t>(spec.default_value)
                    ? std::get<std::int64_t>(spec.default_value)
                    : 0;
            const int minimum =
                declared_bound(spec.minimum, static_cast<int>(std::min<std::int64_t>(
                                                 default_value, 0)));
            const int maximum = declared_bound(spec.maximum, INT_MAX);
            if (spec.power_of_two) {
                const std::pair<int, int> legal = power_of_two_range(minimum, maximum);
                spin->setRange(legal.first, legal.second);
            } else {
                spin->setRange(minimum, maximum);
            }
            field->editor = spin;
            connect(spin, qOverload<int>(&QSpinBox::valueChanged), this, mark_dirty);
            column->addWidget(spin);
            break;
        }
        case EditorKind::DoubleSpinBox: {
            auto* spin = new QDoubleSpinBox(host);
            spin->setDecimals(kFloatDecimals);
            const double declared_min =
                std::holds_alternative<double>(spec.minimum) ? std::get<double>(spec.minimum)
                                                             : -kFloatLimit;
            const double declared_max =
                std::holds_alternative<double>(spec.maximum) ? std::get<double>(spec.maximum)
                                                             : kFloatLimit;
            spin->setRange(declared_min, declared_max);
            const double d = std::holds_alternative<double>(spec.default_value)
                                 ? std::get<double>(spec.default_value)
                                 : 0.0;
            spin->setSingleStep(std::abs(d) >= 10.0 ? 1.0 : 0.1);
            field->editor = spin;
            connect(spin, qOverload<double>(&QDoubleSpinBox::valueChanged), this, mark_dirty);
            column->addWidget(spin);
            break;
        }
        case EditorKind::ComboBox: {
            auto* combo = new QComboBox(host);
            combo->addItems(accepted_values(spec));
            field->editor = combo;
            connect(combo, qOverload<int>(&QComboBox::currentIndexChanged), this, mark_dirty);
            column->addWidget(combo);
            break;
        }
        case EditorKind::LineEdit:
        case EditorKind::PathEdit:
        case EditorKind::ListEdit: {
            auto* edit = new QLineEdit(host);
            field->editor = edit;
            connect(edit, &QLineEdit::textChanged, this, mark_dirty);
            if (field->kind == EditorKind::ListEdit) {
                edit->setPlaceholderText(QStringLiteral("[\"item 1\", \"item 2\"]"));
            }
            const bool browse = field->kind == EditorKind::PathEdit || spec.name == "filenames";
            if (browse) {
                auto* row = new QHBoxLayout();
                row->setContentsMargins(0, 0, 0, 0);
                row->addWidget(edit, 1);
                auto* button = new QPushButton(tr("Browse..."), host);
                button->setObjectName(QStringLiteral("browse.") + object);
                row->addWidget(button);
                column->addLayout(row);
                const bool saving = task_name_.rfind("Export", 0) == 0;
                const bool many = field->kind == EditorKind::ListEdit;
                connect(button, &QPushButton::clicked, this, [this, index, edit, saving, many]() {
                    const QString title = display_name(fields_[index].spec.name);
                    if (many) {
                        const QStringList files = QFileDialog::getOpenFileNames(this, title);
                        if (!files.isEmpty()) {
                            std::vector<std::string> items;
                            for (const QString& file : files) {
                                items.push_back(file.toStdString());
                            }
                            edit->setText(q(list_text(items)));
                            fields_[index].dirty = true;
                        }
                        return;
                    }
                    const QString file = saving ? QFileDialog::getSaveFileName(this, title)
                                                : QFileDialog::getOpenFileName(this, title);
                    if (!file.isEmpty()) {
                        edit->setText(file);
                        fields_[index].dirty = true;
                    }
                });
            } else {
                column->addWidget(edit);
            }
            break;
        }
    }
    field->editor->setObjectName(QStringLiteral("param.") + object);
    field->editor->setToolTip(tip);

    field->error_label = new QLabel(host);
    field->error_label->setObjectName(QStringLiteral("error.") + object);
    field->error_label->setStyleSheet(QStringLiteral("color: #d64545;"));
    field->error_label->setWordWrap(true);
    field->error_label->setVisible(false);
    column->addWidget(field->error_label);

    QString label_text = display_name(spec.name);
    if (field->required) {
        label_text += QStringLiteral(" *");
    }
    auto* label = new QLabel(label_text, this);
    label->setObjectName(QStringLiteral("label.") + object);
    label->setToolTip(tip);
    label->setBuddy(field->editor);
    form->addRow(label, host);

    load(*field, schema_default(*field));
}

ParamValue TaskDialog::schema_default(const Field& field) const {
    if (!is_unset(field.spec.default_value)) {
        return field.spec.default_value;
    }
    switch (field.spec.type) {
        case ParamType::Bool:
            return false;
        case ParamType::Int:
            return std::int64_t{0};
        case ParamType::Float:
            return 0.0;
        case ParamType::String:
            return std::string{};
        case ParamType::StringList:
            return std::vector<std::string>{};
    }
    return {};
}

void TaskDialog::load(Field& field, const ParamValue& value) {
    const QSignalBlocker blocker(field.editor);
    switch (field.kind) {
        case EditorKind::CheckBox:
            static_cast<QCheckBox*>(field.editor)->setChecked(std::get<bool>(value));
            break;
        case EditorKind::SpinBox: {
            auto* spin = static_cast<QSpinBox*>(field.editor);
            const std::int64_t v = std::get<std::int64_t>(value);
            spin->setValue(static_cast<int>(std::clamp<std::int64_t>(v, spin->minimum(), spin->maximum())));
            break;
        }
        case EditorKind::DoubleSpinBox:
            static_cast<QDoubleSpinBox*>(field.editor)->setValue(std::get<double>(value));
            break;
        case EditorKind::ComboBox: {
            auto* combo = static_cast<QComboBox*>(field.editor);
            const int at = combo->findText(q(std::get<std::string>(value)));
            combo->setCurrentIndex(std::max(at, 0));
            break;
        }
        case EditorKind::LineEdit:
        case EditorKind::PathEdit:
            static_cast<QLineEdit*>(field.editor)->setText(q(std::get<std::string>(value)));
            break;
        case EditorKind::ListEdit:
            static_cast<QLineEdit*>(field.editor)
                ->setText(q(list_text(std::get<std::vector<std::string>>(value))));
            break;
    }
    field.loaded = value;
    field.dirty = false;
}

ParamValue TaskDialog::read(const Field& field) const {
    if (!field.dirty) {
        return field.loaded;  // untouched: the exact schema (or remembered) value
    }
    switch (field.kind) {
        case EditorKind::CheckBox:
            return static_cast<QCheckBox*>(field.editor)->isChecked();
        case EditorKind::SpinBox:
            return static_cast<std::int64_t>(static_cast<QSpinBox*>(field.editor)->value());
        case EditorKind::DoubleSpinBox:
            return static_cast<QDoubleSpinBox*>(field.editor)->value();
        case EditorKind::ComboBox:
            return static_cast<QComboBox*>(field.editor)->currentText().toStdString();
        case EditorKind::LineEdit:
        case EditorKind::PathEdit:
            return static_cast<QLineEdit*>(field.editor)->text().toStdString();
        case EditorKind::ListEdit: {
            const std::string text = static_cast<QLineEdit*>(field.editor)->text().trimmed().toStdString();
            if (text.empty()) {
                return std::vector<std::string>{};
            }
            return parse_param_value(ParamType::StringList, text);
        }
    }
    return field.loaded;
}

void TaskDialog::set_error(Field& field, const QString& text) {
    const bool changed = field.error != text;
    field.error = text;
    field.error_label->setText(text);
    field.error_label->setVisible(!text.isEmpty());
    field.editor->setProperty("invalid", !text.isEmpty());
    field.editor->setToolTip(text.isEmpty() ? q(field.spec.doc) : text);
    if (changed) {
        // The "invalid" flag is a dynamic property the style sheet matches on; a property
        // change alone does not make Qt repaint the widget, so the style has to be
        // re-resolved. Only on a real transition: validate() runs on every OK.
        field.editor->style()->unpolish(field.editor);
        field.editor->style()->polish(field.editor);
        field.editor->update();
    }
}

/// The schema's own constraints, the ones an editor can also express. A spin box already
/// clamps to the declared range and a combo box can only hold a declared choice, so what is
/// left to catch here is a value that got in another way: a remembered setting, a typed
/// number, or a hand-set value in a test or a script.
QString TaskDialog::constraint_problem(const Field& field) const {
    const ParamSpec& spec = field.spec;
    const ParamValue value = read(field);
    if (const auto* integer = std::get_if<std::int64_t>(&value)) {
        if (const auto* low = std::get_if<std::int64_t>(&spec.minimum)) {
            if (*integer < *low) {
                return tr("Must be at least %1.").arg(*low);
            }
        }
        if (const auto* high = std::get_if<std::int64_t>(&spec.maximum)) {
            if (*integer > *high) {
                return tr("Must be at most %1.").arg(*high);
            }
        }
        if (spec.power_of_two && (*integer <= 0 || (*integer & (*integer - 1)) != 0)) {
            return tr("Must be a power of two.");
        }
    } else if (const auto* real = std::get_if<double>(&value)) {
        if (const auto* low = std::get_if<double>(&spec.minimum)) {
            if (*real < *low) {
                return tr("Must be at least %1.").arg(*low);
            }
        }
        if (const auto* high = std::get_if<double>(&spec.maximum)) {
            if (*real > *high) {
                return tr("Must be at most %1.").arg(*high);
            }
        }
    } else if (const auto* text = std::get_if<std::string>(&value)) {
        // Only a declared closed set is checked: the doc-scraped list is a display heuristic
        // and must not turn a free-text parameter into a rejected one.
        if (!spec.choices.empty() &&
            std::find(spec.choices.begin(), spec.choices.end(), *text) == spec.choices.end()) {
            return tr("Must be one of: %1.")
                .arg(accepted_values(spec).join(QStringLiteral(", ")));
        }
    }
    return {};
}

bool TaskDialog::validate() {
    bool ok = true;
    for (Field& field : fields_) {
        QString problem;
        try {
            const ParamValue value = read(field);
            problem = constraint_problem(field);
            if (problem.isEmpty() && field.required) {
                if (const auto* s = std::get_if<std::string>(&value)) {
                    if (QString::fromStdString(*s).trimmed().isEmpty()) {
                        problem = tr("This value is required.");
                    }
                } else if (const auto* list = std::get_if<std::vector<std::string>>(&value)) {
                    if (list->empty()) {
                        problem = tr("This value is required.");
                    }
                }
            }
        } catch (const std::exception& failure) {
            problem = tr("Not a valid list (%1). Use [\"a\", \"b\"].")
                          .arg(QString::fromUtf8(failure.what()));
        }
        set_error(field, problem);
        if (!problem.isEmpty()) {
            ok = false;
            if (field.advanced && advanced_group_ != nullptr) {
                advanced_group_->setChecked(true);
            }
        }
    }
    return ok;
}

QString TaskDialog::error(const std::string& param) const {
    const Field* field = find(param);
    return field != nullptr ? field->error : QString();
}

std::vector<std::string> TaskDialog::invalid_params() const {
    std::vector<std::string> names;
    for (const Field& field : fields_) {
        if (!field.error.isEmpty()) {
            names.push_back(field.spec.name);
        }
    }
    return names;
}

Params TaskDialog::params() const {
    Params result;
    for (const Field& field : fields_) {
        result.declare(field.spec);
    }
    for (const Field& field : fields_) {
        result.set(field.spec.name, read(field));
    }
    return result;
}

void TaskDialog::reset_to_defaults() {
    for (Field& field : fields_) {
        load(field, schema_default(field));
        set_error(field, QString());
    }
}

void TaskDialog::set_value(const std::string& param, const ParamValue& value) {
    if (Field* field = find(param)) {
        load(*field, value);
    }
}

QWidget* TaskDialog::editor(const std::string& param) const {
    const Field* field = find(param);
    return field != nullptr ? field->editor : nullptr;
}

TaskDialog::EditorKind TaskDialog::kind(const std::string& param) const {
    const Field* field = find(param);
    return field != nullptr ? field->kind : EditorKind::LineEdit;
}

bool TaskDialog::required(const std::string& param) const {
    const Field* field = find(param);
    return field != nullptr && field->required;
}

bool TaskDialog::advanced(const std::string& param) const {
    const Field* field = find(param);
    return field != nullptr && field->advanced;
}

void TaskDialog::load_remembered() {
    for (Field& field : fields_) {
        if (is_unset(field.spec.default_value)) {
            continue;  // required values are never carried over
        }
        const QString key = settings_key(task_name_, field.spec.name);
        if (!settings_.contains(key)) {
            continue;
        }
        try {
            const ParamValue value =
                parse_param_value(field.spec.type, settings_.value(key).toString().toStdString());
            if (field.kind == EditorKind::ComboBox &&
                static_cast<QComboBox*>(field.editor)->findText(q(std::get<std::string>(value))) < 0) {
                continue;
            }
            load(field, value);
        } catch (const std::exception&) {
            // A stale or hand-edited entry: keep the schema default.
        }
    }
}

void TaskDialog::remember() const {
    for (const Field& field : fields_) {
        if (is_unset(field.spec.default_value)) {
            continue;
        }
        settings_.setValue(settings_key(task_name_, field.spec.name),
                           q(format_param_value(read(field))));
    }
}

void TaskDialog::accept() {
    if (!validate()) {
        return;
    }
    remember();
    QDialog::accept();
}

}  // namespace alphashape::gui
