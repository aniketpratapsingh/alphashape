# Run a Python test with an interpreter matching the one CMake configured with.
#
# add_test() bakes the interpreter's absolute path in at configure time, but CI
# configures on one runner and unpacks the build tree on another. GitHub rolls
# its images independently, so the build job's Homebrew Python 3.15 can be gone
# (or an incompatible minor version) where the tests run, and ctest then reports
# "Unable to find executable" / "Not Run" for every Python entry.
#
# This script is the test command instead of the raw interpreter: it keeps the
# configured executable when it still exists and is the expected version, and
# otherwise looks for a matching interpreter on PATH. When none exists it fails
# with the version it needed rather than a bare path. The module is per minor
# version, so EXPECTED_VERSION is what makes a found interpreter usable.
#
# Inputs (-D):
#   TEST_SCRIPT       .py to run
#   EXPECTED_VERSION  major.minor the module was built for ("" skips the check)
#   CONFIGURED_PYTHON interpreter CMake found at configure time (may be gone)
#   PYTHONPATH        value for the child's PYTHONPATH ("" for none)
#   TEST_ARGS         semicolon-separated extra arguments ("" for none)

function(_alphashape_python_version exe out)
    execute_process(COMMAND "${exe}" -c "import sys; sys.stdout.write('%d.%d' % sys.version_info[:2])"
        RESULT_VARIABLE rc OUTPUT_VARIABLE value ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(rc EQUAL 0)
        set(${out} "${value}" PARENT_SCOPE)
    else()
        set(${out} "" PARENT_SCOPE)
    endif()
endfunction()

set(candidates "")
if(CONFIGURED_PYTHON)
    list(APPEND candidates "${CONFIGURED_PYTHON}")
endif()
if(EXPECTED_VERSION)
    list(APPEND candidates "python${EXPECTED_VERSION}")
endif()
list(APPEND candidates python3 python)

set(chosen "")
set(tried "")
foreach(candidate IN LISTS candidates)
    if(chosen)
        break()
    endif()
    # find_program() skips the search when its variable is already set, and in
    # script mode it lands in both stores, so clear both between candidates.
    unset(_exe)
    unset(_exe CACHE)
    find_program(_exe "${candidate}")
    if(NOT _exe)
        continue()
    endif()
    _alphashape_python_version("${_exe}" found)
    if(found)
        list(APPEND tried "${_exe} (${found})")
        if(NOT EXPECTED_VERSION OR found STREQUAL EXPECTED_VERSION)
            set(chosen "${_exe}")
        endif()
    endif()
endforeach()

if(NOT chosen)
    message(FATAL_ERROR
        "no Python ${EXPECTED_VERSION} interpreter to run ${TEST_SCRIPT};\n"
        "tried: ${tried}\n"
        "The module is built for one Python minor version, so install that one.")
endif()

if(PYTHONPATH)
    execute_process(COMMAND "${CMAKE_COMMAND}" -E env "PYTHONPATH=${PYTHONPATH}"
        "${chosen}" "${TEST_SCRIPT}" ${TEST_ARGS} RESULT_VARIABLE rc)
else()
    execute_process(COMMAND "${chosen}" "${TEST_SCRIPT}" ${TEST_ARGS} RESULT_VARIABLE rc)
endif()
# String comparison, not `EQUAL 0`: a process killed by a signal makes CMake
# report a phrase ("Segmentation fault"), which numeric comparison treats as 0.
if(NOT "${rc}" STREQUAL "0")
    message(FATAL_ERROR "${TEST_SCRIPT} failed (result: ${rc})")
endif()
