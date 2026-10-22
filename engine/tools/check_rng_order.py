#!/usr/bin/env python3
"""check_rng_order.py -- flag random draws whose order the compiler chooses.

C++ leaves the evaluation order of *function arguments* unspecified: clang
evaluates them left to right, g++ right to left. So

    Eigen::Vector3d(f(rng), f(rng), f(rng))
    sample(random_depth(rng), random_normal(rng))
    h(g(a(rng), b(rng)), c(rng))

consume one engine's bit stream in opposite orders on the two platforms. The
same seed then selects a different scene, and seeded reconstructions diverge
between macOS and Linux (see LIMITATIONS.md, "Argument evaluation order"). Every
draw must go into a named local first.

Braced initialisers are *not* a violation: their clauses are sequenced left to
right ([dcl.init.list]/4), so `P p{f(), f()}` is well defined. The check splits
each braced list into its elements and counts the draws per unordered group, so
it reports exactly the calls whose arguments hold two or more draws.

Two further rules, both about reproducibility rather than order:

  * std::*_distribution is specified only by its output range, so libc++ and
    libstdc++ map engine bits to values differently. Draw through
    base/utils/portable_rng.hpp instead (TESTING.md, "Numeric consistency").
  * std::shuffle's permutation is implementation-defined for the same reason;
    use partial Fisher-Yates over rng::index_below.

A statement that legitimately makes several draws (a lambda that draws when
*called*, a helper taking two engines) is marked with a `// rng-order: ok`
comment, on the statement itself or the line above it.

Usage:
    tools/check_rng_order.py [ROOT ...]

    ROOTs default to src/ and tests/ beside this script's parent.
Exit status: 0 clean, 1 findings, 2 usage error.
"""

import re
import sys
from pathlib import Path

MARKER = "rng-order: ok"

# Identifier roots that denote a random engine. The `\w*` suffix catches
# rng_d / rng_n / gen2 / generator_b. Matching by name keeps this tool
# independent of the code's type definitions; a local named `gen` that holds
# something else does not exist in this tree, and the marker covers a future one.
# A single `g` is deliberately absent: fixture lambdas name their parameter `g`
# and declaring one draws nothing.
ENGINE = re.compile(r"\b(?:rng|rnd|generator|engine|gen|prng)\w*\b")

# A bare engine draw: rng() or engine.next(). Drawing through a distribution or
# helper is the call-with-engine rule below.
RAW_DRAW = re.compile(
    r"\b(?:rng|rnd|generator|engine|gen|prng)\w*\s*\(\s*\)"      # rng()
    r"|\b(?:rng|rnd|generator|engine|gen|prng)\w*\.next\s*\("    # engine.next(
)

# Calls that take an engine but are not draws: they own the draw order inside
# (scene builders, index samplers, algorithm samplers).
NOT_A_DRAW = {
    "make_scene", "make_descriptor", "make_rig", "make_arc_scene", "make_cloud",
    "distinct_indices", "shuffle", "sample", "seed", "reset", "index_below",
}

# Standard-library distributions are also not portable: they are specified only
# by their output range, so libc++ and libstdc++ map engine bits to values
# differently. Draw through base/utils/portable_rng.hpp instead (TESTING.md,
# "Numeric consistency"). The portable header is the implementation and is
# exempt.
DISTRIBUTION = re.compile(
    r"\bstd::(?:uniform_real_distribution|uniform_int_distribution|"
    r"normal_distribution|bernoulli_distribution|discrete_distribution|"
    r"poisson_distribution|exponential_distribution|gamma_distribution|"
    r"weibull_distribution|geometric_distribution|binomial_distribution|"
    r"cauchy_distribution|student_t_distribution|fisher_f_distribution|"
    r"extreme_value_distribution|piecewise_constant_distribution|"
    r"piecewise_linear_distribution|lognormal_distribution|chi_squared_distribution)"
)
# std::shuffle's algorithm is implementation-defined too: the standard only
# says "a random permutation". libc++ and libstdc++ differ, so one seed gives a
# different permutation. Partial Fisher-Yates over rng::index_below is portable.
SHUFFLE = re.compile(r"\bstd::(?:shuffle|random_shuffle)\s*\(")
EXEMPT = {"portable_rng.hpp"}

STRING_OR_COMMENT = re.compile(
    r'"(?:\\.|[^"\\])*"'   # string literal
    r"|'(?:\\.|[^'\\])*'"  # character literal
    r"|//[^\n]*"           # line comment
    r"|/\*.*?\*/",         # block comment
    re.DOTALL,
)

CALL = re.compile(r"\b([A-Za-z_]\w*)\s*(?=\()")


def blank_ignorable(source: str) -> str:
    """Blanks comments and preprocessor lines, keeping offsets and newlines."""
    def blank(match: re.Match) -> str:
        text = match.group(0)
        if text.startswith(('"', "'")):
            return text
        return re.sub(r"[^\n]", " ", text)

    source = STRING_OR_COMMENT.sub(blank, source)
    return re.sub(r"(?m)^[ \t]*#[^\n]*", lambda m: " " * len(m.group(0)), source)


def matching_paren(text: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def engine_calls(text: str):
    """Yields (start, end) spans of calls that take an engine in their arguments.

    A call is a draw site; the calls above it are not, so counting only the
    innermost ones counts the draws. `push_back(flip_bits(a, 3, rng))` is one
    draw, while `Vec3(u(rng), u(rng))` is two.
    """
    spans = [match.span() for match in RAW_DRAW.finditer(text)]
    for match in CALL.finditer(text):
        if match.group(1) in NOT_A_DRAW:
            continue
        close_index = matching_paren(text, match.end())
        if close_index < 0:
            continue
        if ENGINE.search(text[match.end() + 1 : close_index]):
            spans.append((match.start(), close_index))
    return spans


def count_draws(text: str) -> int:
    """Counts the innermost engine draws that happen when this expression runs."""
    spans = engine_calls(text)
    # A span that encloses another is only the expression around a draw; the
    # enclosed ones are the draws.
    return sum(
        1
        for start, end in spans
        if not any(
            start < other_start and other_end <= end
            for other_start, other_end in spans
            if (other_start, other_end) != (start, end)
        )
    )


def split_top_level(content: str, separator: str) -> list[str]:
    """Splits on `separator` where it is not nested in brackets or braces."""
    parts: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(content):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(content[start:index])
            start = index + 1
    parts.append(content[start:])
    return parts


def max_unordered_draws(text: str) -> int:
    """Largest number of draws inside one *unordered* part of an expression.

    Function arguments are unordered with respect to each other, but the clauses
    of a braced initialiser list are sequenced left to right ([dcl.init.list]/4),
    so `P p{f(), f()}` is well defined while `g(f(), f())` is not. Each braced
    list is therefore split into its elements recursively; the enclosing text
    (the parts around the list) forms one more unordered group.
    """
    brace = -1
    depth = 0
    for index, char in enumerate(text):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth -= 1
        elif char == "{":
            if depth == 0:
                brace = index
                break
            depth += 1
        elif char == "}":
            depth -= 1
    if brace < 0:
        return count_draws(text)

    # Find the matching close brace for this list.
    depth = 0
    close = -1
    for index in range(brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                close = index
                break
    if close < 0:
        return count_draws(text)  # unterminated: fall back

    outer = text[:brace] + " " * (close - brace + 1) + text[close + 1 :]
    worst = count_draws(outer)
    for element in split_top_level(text[brace + 1 : close], ","):
        worst = max(worst, max_unordered_draws(element))
    return worst


def first_line_of(source: str, offset: int) -> int:
    return source.count("\n", 0, offset) + 1


def marked(lines: list[str], line: int) -> bool:
    if 1 <= line <= len(lines) and MARKER in lines[line - 1]:
        return True
    return line >= 2 and MARKER in lines[line - 2]


def is_block_brace(text: str, index: int) -> bool:
    """True if text[index] == '{' opens a block rather than an initialiser."""
    before = text[:index].rstrip()
    if not before:
        return True
    if before.endswith((")", "}", ";", "else", "do", "try", ":")):
        return True
    # `struct X {`, `namespace n {`, `class X {` open blocks too.
    return bool(re.search(r"\b(?:struct|class|namespace|union|enum)\b[^;{}]*$", before))


def check_file(path: Path) -> list[str]:
    findings: list[str] = []
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        return [f"{path}: cannot read: {error}"]

    code = blank_ignorable(source)
    lines = source.splitlines()

    # A statement runs from the previous statement/block boundary to a ';' at
    # paren/bracket depth 0. A statement's own leading brace (the body of a
    # function, test case or lambda) starts a new statement; a brace that opens
    # an initialiser list inside an expression does not, so
    # `T v{a(rng), b(rng)}` stays one statement.
    start = 0
    paren = 0
    stack: list[bool] = []  # per brace: was it a block?
    index = 0
    while index < len(code):
        char = code[index]
        if char in "([":
            paren += 1
        elif char in ")]":
            paren -= 1
        elif paren == 0 and char == "{":
            segment = code[start:index].strip()
            # A brace that opens right after `=` / `,` / `(` / `{` (or at the
            # start of a statement) initialises; everything else closes a
            # declarator or a control statement and opens a block.
            opens_list = not segment or segment[-1] in "=,({"
            block = not opens_list and is_block_brace(code, index)
            stack.append(block)
            if block:
                start = index + 1
        elif paren == 0 and char == "}":
            # Only a block's closing brace ends a statement; an initialiser
            # list's does not, so its elements stay in the statement being
            # counted.
            if stack and stack.pop():
                start = index + 1
        elif paren == 0 and char == ";":
            segment = code[start:index]
            stripped = segment.strip()
            line = first_line_of(code, start + len(segment) - len(segment.lstrip()))
            start = index + 1
            header = re.match(r"^(?:for|while|if|switch)\s*\(", stripped)
            if header:
                # Only the condition belongs to this statement; the body is a
                # separate one. A draw in a loop condition is one per test,
                # which is already sequenced by the loop.
                continue
            draws = max_unordered_draws(stripped)
            if draws >= 2 and stripped and not marked(lines, line):
                findings.append(
                    f"{path}:{line}: {draws} engine draws in one statement, so "
                    f"their order is the compiler's to choose; move each draw "
                    f"into a named local (or mark it '// {MARKER}'): "
                    f"{stripped.splitlines()[0][:110]}"
                )
        index += 1

    for match in DISTRIBUTION.finditer(code):
        line = first_line_of(code, match.start())
        if marked(lines, line):
            continue
        findings.append(
            f"{path}:{line}: {match.group(0)} is implementation-defined "
            f"across libc++/libstdc++; use base/utils/portable_rng.hpp"
        )
    for match in SHUFFLE.finditer(code):
        line = first_line_of(code, match.start())
        if marked(lines, line):
            continue
        findings.append(
            f"{path}:{line}: {match.group(0)}... is implementation-defined "
            f"across libc++/libstdc++; use Fisher-Yates over rng::index_below"
        )
    return findings


def source_files(root: Path):
    if root.is_file():
        yield root
        return
    for path in sorted(root.rglob("*")):
        if path.suffix in (".cpp", ".hpp", ".cc", ".h") and not any(
            part.startswith("build") for part in path.parts
        ):
            yield path


def main(argv: list[str]) -> int:
    here = Path(__file__).resolve().parent
    args = [arg for arg in argv[1:] if arg != "--tests"]
    roots = [Path(arg) for arg in args] or [here.parent / "src", here.parent / "tests"]
    missing = [str(root) for root in roots if not root.exists()]
    if missing:
        print(f"check_rng_order: no such path: {', '.join(missing)}", file=sys.stderr)
        return 2

    findings: list[str] = []
    scanned = 0
    for root in roots:
        for path in source_files(root):
            if path.name in EXEMPT:
                continue
            scanned += 1
            findings.extend(check_file(path))

    if findings:
        print("\n".join(findings))
        print(
            f"\ncheck_rng_order: {len(findings)} finding(s) in {scanned} file(s). "
            f"Seeded scenes must reproduce across clang/libc++ and g++/libstdc++; "
            f"draw into named locals.",
            file=sys.stderr,
        )
        return 1
    print(f"check_rng_order: clean ({scanned} file(s) scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
