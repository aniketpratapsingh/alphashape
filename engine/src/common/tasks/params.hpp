#pragma once

// Task parameters: a string-keyed schema, shared by every front end.
//
// This is the central design idea. Parameters are NOT a positional argument list. Each task
// declares its parameters by name, keyed "<Task>/<param>", and the SAME schema
// drives:
//
//   * the Python keyword binder   (chunk.match_photos(keypoint_limit=40000))
//   * the CLI                     (--keypoint-limit 40000)
//   * the XML/project serialiser  (<keypoint_limit>40000</keypoint_limit>)
//   * the network protocol
//
// Because there is one schema, the API and the project file cannot drift: a
// parameter that round-trips through doc.xml is by construction one the task
// accepts, and vice versa. We enforce that as a test (see the parity test).
//
// Unknown keys are rejected with "Invalid argument name: <name>".

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "base/utils/error.hpp"

namespace alphashape {

/// The value types a parameter may hold. Deliberately closed: adding a type
/// means teaching the XML writer, the CLI parser and the Python binder about
/// it, which is the point — one schema, all front ends.
///
/// StringList is the odd one out: it is the only aggregate. Its text form (see
/// parse_param_value/format_param_value) is a JSON array of JSON strings, which
/// keeps the doc.xml meta, the CLI's --set and the XML reader all expressing a
/// list the same way, with no separator character to escape.
using ParamValue =
    std::variant<std::monostate, bool, std::int64_t, double, std::string, std::vector<std::string>>;

enum class ParamType { Bool, Int, Float, String, StringList };

/// Declaration of one parameter: name, type, default, and documentation, plus optional
/// value constraints.
///
/// The constraints exist so a front end can offer an editor that cannot produce a value the
/// task would reject, without knowing the task by name: an interactive dialog builds a combo
/// box for a closed value set and a spin box with the declared bounds. They are metadata
/// only — Params never enforces them, so a task's own argument checks remain the single
/// authority on what is acceptable, and a declaration that omits them keeps the old
/// permissive behaviour.
struct ParamSpec {
    std::string name;          ///< bare name, e.g. "keypoint_limit"
    ParamType type = ParamType::Int;
    ParamValue default_value;  ///< monostate if the parameter is required
    std::string doc;           ///< surfaced as the Python docstring

    /// The accepted values of a String parameter, in display order. Empty means "not a
    /// closed set" (a front end may still scrape a list out of `doc`).
    std::vector<std::string> choices;

    /// Inclusive bounds of an Int or Float parameter; monostate on either side means that
    /// side is unbounded. The type must match `type`.
    ParamValue minimum;
    ParamValue maximum;

    /// An Int parameter that must be a power of two (a resolution downscale factor).
    /// `minimum` / `maximum` still apply.
    bool power_of_two = false;

    ParamSpec() = default;

    /// The four-field form every declaration uses. It is a constructor rather than an
    /// aggregate so callers may omit the constraint fields without the compiler warning about
    /// the ones they left out, and then set the constraints they need on the returned object.
    ParamSpec(std::string param_name, ParamType param_type, ParamValue default_val,
              std::string documentation)
        : name(std::move(param_name)),
          type(param_type),
          default_value(std::move(default_val)),
          doc(std::move(documentation)) {}

    /// Full schema key, e.g. "DetectFeatures/keypoint_limit".
    std::string qualified(const std::string& task_name) const {
        return task_name + "/" + name;
    }
};

/// A set of parameter values bound for one task invocation.
class Params {
public:
    Params() = default;

    /// Declare the schema. Called once per task type at registration.
    void declare(const ParamSpec& spec);

    /// True if a parameter with this bare name is declared.
    bool declares(const std::string& name) const;

    /// By reference: callers take pointers/references into the result, and a
    /// by-value return here once produced a dangling pointer in a front end.
    const std::vector<ParamSpec>& specs() const { return specs_; }

    /// Set a value by bare name. Throws InvalidArgumentError if undeclared,
    /// and ValueError if the value's type does not match the declaration.
    void set(const std::string& name, ParamValue value);

    /// Read a value, falling back to the declared default. Throws
    /// InvalidArgumentError if undeclared. Throws ValueError if required
    /// (no default) and unset.
    ParamValue get(const std::string& name) const;

    bool get_bool(const std::string& name) const;
    std::int64_t get_int(const std::string& name) const;
    double get_float(const std::string& name) const;
    std::string get_string(const std::string& name) const;
    std::vector<std::string> get_string_list(const std::string& name) const;

    /// Names of parameters the caller supplied explicitly (as opposed to
    /// values the schema defaulted). Useful for UIs and diagnostics — e.g. to
    /// show which fields the user touched. Note this is NOT what the project
    /// file records: the file keeps the effective value of every declared
    /// parameter (see record_task_params).
    std::vector<std::string> explicitly_set() const;

    bool is_set(const std::string& name) const;

private:
    const ParamSpec* find_spec(const std::string& name) const;

    std::vector<ParamSpec> specs_;
    std::map<std::string, ParamValue> values_;
};

/// Parse a string into the declared type. Used by the CLI and by the XML
/// reader, so both accept exactly the same spelling.
ParamValue parse_param_value(ParamType type, const std::string& text);

/// Render a value back to a string. Inverse of parse_param_value for the
/// types we support; used by the XML writer and the CLI's --dump.
std::string format_param_value(const ParamValue& value);

}  // namespace alphashape
