# Windows install layout and the NSIS installer (M29 WP-29.7).
#
# Included from engine/CMakeLists.txt only when ALPHASHAPE_BUILD_INSTALLER=ON,
# which the packaging step turns on in a build tree that is already compiled, so
# the default lanes are untouched.
#
# Layout under the install directory (%LOCALAPPDATA%\Programs\alphashape for the
# default per-user install, %ProgramFiles%\alphashape for a per-machine one):
#
#   bin\alphashape.exe          GUI, with Qt's DLLs and plugins beside it
#   bin\alphashape-cli.exe      command line
#   bin\*.dll                   vcpkg runtime, the VC++ runtime, libomp, Qt
#   bin\platforms\q*.dll        Windows, offscreen and minimal platform plugins
#   python\alphashape.pyd       Python module, with the same DLLs beside it
#   share\proj\                 proj.db and a size-capped grid set
#   LICENSES\                   GPL, Qt (LGPL-3) and bundled-font (OFL) licences
#
# Components: core (hidden, always installed: DLLs, PROJ data, licences), gui,
# cli, python. "Add to PATH" is a section in NSIS.template.in rather than a
# component, because NSIS cannot show a component that owns no file.

include(GNUInstallDirs)

set(ALPHASHAPE_PACKAGING_DIR "${PROJECT_SOURCE_DIR}/../packaging/windows")
set(ALPHASHAPE_REPO_ROOT "${PROJECT_SOURCE_DIR}/..")

# The installer exists to ship the desktop app. That target is optional, and its
# absence would otherwise fail much later with a message about a missing Qt
# version or an unset windeployqt, so it is diagnosed here instead.
if(NOT ALPHASHAPE_BUILD_GUI)
    message(FATAL_ERROR
        "alphashape: ALPHASHAPE_BUILD_INSTALLER=ON needs -DALPHASHAPE_BUILD_GUI=ON; the "
        "installer lays out bin\\alphashape.exe and its notice file names the Qt it ships.")
endif()

# ALPHASHAPE_PACKAGE_VERSION lets a release stamp its tag (v0.2.0) into the
# installer while the source tree keeps its own version. "dev" is what the
# reusable workflow passes for a plain CI run, and it belongs in no installer's
# DisplayVersion.
if(NOT ALPHASHAPE_PACKAGE_VERSION OR ALPHASHAPE_PACKAGE_VERSION STREQUAL "dev")
    set(ALPHASHAPE_PACKAGE_VERSION "${PROJECT_VERSION}")
endif()

# The numeric triple behind the package version. A release version arrives as
# "v0.2.0" and a nightly as "0.1.0-nightly", neither of which an EXE's
# FILEVERSION or NSIS's VIProductVersion accepts: strip the tag's leading v,
# take the numeric prefix, and pad the missing fields with zeros. The same
# numbers go into the GUI's version resource and into CPack's
# CPACK_PACKAGE_VERSION_MAJOR/MINOR/PATCH, which the installer template uses.
string(REGEX REPLACE "^[vV]" "" _alphashape_version_numeric "${ALPHASHAPE_PACKAGE_VERSION}")
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)*" _alphashape_version_numeric "${_alphashape_version_numeric}")
if(NOT _alphashape_version_numeric)
    set(_alphashape_version_numeric "${PROJECT_VERSION}")
endif()
string(REPLACE "." ";" _alphashape_version_parts "${_alphashape_version_numeric}")
set(ALPHASHAPE_VERSION_QUADRUPLE "")
set(_alphashape_version_fields "")
foreach(_part RANGE 0 3)
    list(LENGTH _alphashape_version_parts _count)
    if(_part LESS _count)
        list(GET _alphashape_version_parts ${_part} _value)
    else()
        set(_value 0)
    endif()
    list(APPEND _alphashape_version_fields "${_value}")
    string(APPEND ALPHASHAPE_VERSION_QUADRUPLE "${_value},")
endforeach()
string(REGEX REPLACE ",$" "" ALPHASHAPE_VERSION_QUADRUPLE "${ALPHASHAPE_VERSION_QUADRUPLE}")
list(GET _alphashape_version_fields 0 ALPHASHAPE_VERSION_MAJOR)
list(GET _alphashape_version_fields 1 ALPHASHAPE_VERSION_MINOR)
list(GET _alphashape_version_fields 2 ALPHASHAPE_VERSION_PATCH)
unset(_value)

# The installer's and the shortcuts' icon, rendered from the design-token
# geometry by packaging/windows/make-ico.py. Generated rather than committed so
# it cannot drift from the SVG the other platforms use, and generated with the
# Python CMake already found for the build (no ImageMagick on the runner).
find_package(Python3 COMPONENTS Interpreter QUIET)
if(NOT Python3_Interpreter_FOUND)
    message(FATAL_ERROR "alphashape: the installer icon needs a Python 3 interpreter; "
                        "install-qt/msvc-dev-cmd do not supply one, so add actions/setup-python")
endif()
set(ALPHASHAPE_ICON "${CMAKE_BINARY_DIR}/windows/alphashape.ico")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/windows")
execute_process(
    COMMAND "${Python3_EXECUTABLE}" "${ALPHASHAPE_PACKAGING_DIR}/make-ico.py" "${ALPHASHAPE_ICON}"
    RESULT_VARIABLE _alphashape_icon_rc
    OUTPUT_QUIET)
if(NOT _alphashape_icon_rc EQUAL 0 OR NOT EXISTS "${ALPHASHAPE_ICON}")
    message(FATAL_ERROR "alphashape: make-ico.py failed (${_alphashape_icon_rc})")
endif()
message(STATUS "  installer icon ... ${ALPHASHAPE_ICON}")

# The module is not in bin: a .pyd's load-time DLL search covers its own
# directory (and System32), not the directory above it.
set(ALPHASHAPE_WINDOWS_PYTHON_DIR "python")

# PROJ grids: at most 32 MiB each and 64 MiB in total, smallest first. That
# keeps the worldwide geoid models and drops the continental grids, the same
# trade the macOS bundle makes, while still bounding the installer size from
# above whatever proj-data happens to ship.
set(ALPHASHAPE_PROJ_GRID_MAX_BYTES 33554432)
set(ALPHASHAPE_PROJ_TOTAL_MAX_BYTES 67108864)

# ---- binaries ---------------------------------------------------------------------------------
if(ALPHASHAPE_BUILD_CLI)
    install(TARGETS alphashape-cli RUNTIME DESTINATION bin COMPONENT cli)
endif()

# The executable's own icon and version resource, so Explorer, the task bar and
# a shortcut made by hand all show the product mark rather than the generic
# application icon. Only the installed copy is renamed; the build tree keeps
# alphashape-gui so the tests and the other platforms' packaging scripts go on
# naming one file.
if(ALPHASHAPE_BUILD_GUI AND TARGET alphashape-gui)
    set(ALPHASHAPE_WINDOWS_RC "${CMAKE_BINARY_DIR}/windows/alphashape.rc")
    file(WRITE "${ALPHASHAPE_WINDOWS_RC}"
         "IDI_ICON1 ICON DISCARDABLE \"${ALPHASHAPE_ICON}\"\n"
         "VS_VERSION_INFO VERSIONINFO\n"
         "FILEVERSION ${ALPHASHAPE_VERSION_QUADRUPLE}\n"
         "PRODUCTVERSION ${ALPHASHAPE_VERSION_QUADRUPLE}\n"
         "BEGIN\n"
         "  BLOCK \"StringFileInfo\"\n"
         "  BEGIN\n"
         "    BLOCK \"040904b0\"\n"
         "    BEGIN\n"
         "      VALUE \"FileDescription\", \"alphashape\"\n"
         "      VALUE \"FileVersion\", \"${ALPHASHAPE_PACKAGE_VERSION}\"\n"
         "      VALUE \"ProductName\", \"alphashape\"\n"
         "      VALUE \"ProductVersion\", \"${ALPHASHAPE_PACKAGE_VERSION}\"\n"
         "      VALUE \"CompanyName\", \"alphashape contributors\"\n"
         "      VALUE \"LegalCopyright\", \"GPL-3.0-or-later\"\n"
         "    END\n"
         "  END\n"
         "END\n")
    target_sources(alphashape-gui PRIVATE "${ALPHASHAPE_WINDOWS_RC}")

    install(PROGRAMS $<TARGET_FILE:alphashape-gui>
            DESTINATION bin RENAME alphashape.exe COMPONENT gui)
endif()

if(ALPHASHAPE_BUILD_PYTHON AND TARGET alphashape_python)
    # A Python extension module is a MODULE library. On Windows CMake places the
    # .pyd through RUNTIME, elsewhere through LIBRARY, so both destinations are
    # given (install(TARGETS) rejects a module target that names only one).
    # The file keeps the name CMake gave it -- alphashape.cp314-win_amd64.pyd --
    # because Python recognises the ".cpXXX-win_amd64" tag and still imports it
    # as `alphashape`; renaming it to alphashape.pyd would also work but would
    # make the interpreter it was built for invisible.
    install(TARGETS alphashape_python
            RUNTIME DESTINATION ${ALPHASHAPE_WINDOWS_PYTHON_DIR}
            LIBRARY DESTINATION ${ALPHASHAPE_WINDOWS_PYTHON_DIR}
            COMPONENT python)
    install(FILES ${ALPHASHAPE_PACKAGING_DIR}/python-usage.txt
            DESTINATION ${ALPHASHAPE_WINDOWS_PYTHON_DIR} COMPONENT python)
endif()

# ---- app-local runtime: vcpkg DLLs, the VC++ runtime, libomp ------------------------------------
# Windows has no rpath and no "bin" directory on the search path: every DLL a
# binary needs must sit beside it. Nothing is installed system-wide, so the
# installer neither needs a redistributable nor writes outside its own tree.
if(NOT ALPHASHAPE_RUNTIME_DLL_DIR)
    if(VCPKG_INSTALLED_DIR AND VCPKG_TARGET_TRIPLET)
        file(TO_CMAKE_PATH "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin"
             ALPHASHAPE_RUNTIME_DLL_DIR)
    endif()
endif()
if(NOT EXISTS "${ALPHASHAPE_RUNTIME_DLL_DIR}")
    message(FATAL_ERROR
        "alphashape: ALPHASHAPE_BUILD_INSTALLER=ON but the vcpkg runtime directory "
        "'${ALPHASHAPE_RUNTIME_DLL_DIR}' does not exist. Set ALPHASHAPE_RUNTIME_DLL_DIR to "
        "the directory holding the x64-windows runtime DLLs.")
endif()

# The Visual C++ runtime is copied app-locally: no vc_redist, no system write.
# msvc-dev-cmd exports VCToolsRedistDir; the glob survives a toolset rename.
# The value arrives with backslashes ("C:\Program Files\..."), and CMake writes
# a literal backslash into cmake_install.cmake, where "\P" is an invalid escape
# and the generated script would be rejected at install time, so the path is
# normalised to CMake's forward-slash form first.
if(DEFINED ENV{VCToolsRedistDir})
    file(TO_CMAKE_PATH "$ENV{VCToolsRedistDir}" ALPHASHAPE_VC_RUNTIME_ROOT)
    file(GLOB ALPHASHAPE_VC_RUNTIME_DLLS
         "${ALPHASHAPE_VC_RUNTIME_ROOT}/x64/Microsoft.VC*.CRT/*.dll")
else()
    set(ALPHASHAPE_VC_RUNTIME_DLLS "")
endif()
if(NOT ALPHASHAPE_VC_RUNTIME_DLLS)
    message(FATAL_ERROR
        "alphashape: cannot find the Visual C++ runtime under "
        "$ENV{VCToolsRedistDir}/x64/Microsoft.VC*.CRT. Configure from a developer "
        "environment (after ilammy/msvc-dev-cmd) so VCToolsRedistDir is set.")
endif()

# clang-cl links LLVM's libomp, which sits in the LLVM toolchain's bin. Same
# backslash caveat as the redist directory above.
get_filename_component(ALPHASHAPE_COMPILER_BIN "${CMAKE_CXX_COMPILER}" DIRECTORY)
file(TO_CMAKE_PATH "${ALPHASHAPE_COMPILER_BIN}" ALPHASHAPE_COMPILER_BIN)
if(EXISTS "${ALPHASHAPE_COMPILER_BIN}/libomp.dll")
    set(ALPHASHAPE_LIBOMP_DLL "${ALPHASHAPE_COMPILER_BIN}/libomp.dll")
else()
    set(ALPHASHAPE_LIBOMP_DLL "")
    message(WARNING "alphashape: no libomp.dll beside ${CMAKE_CXX_COMPILER}; an OpenMP "
                    "build would fail to start on a machine without LLVM installed")
endif()

# vcpkg's build-time-only ports can leave a DLL in the same bin directory that
# nothing the application starts links. They are excluded by name rather than
# by "whatever vcpkg happens to install", because the glob has to keep working
# for the ports that are real runtime dependencies.
#
#   SPIRV-Tools-shared.dll  the shader *tool* port builds it beside spirv-val;
#                           the engine embeds SPIR-V at build time and never
#                           loads the library at run time.
#   vulkan-1.dll            the Vulkan loader. Windows supplies one with the
#                           graphics driver, and a private copy would be found
#                           first (the application directory precedes System32
#                           in the DLL search order) and could not see the
#                           driver's ICDs. The engine opens it by name at run
#                           time and falls back to the CPU when it is absent.
set(ALPHASHAPE_TOOL_ONLY_DLL_PATTERNS
    FILES_MATCHING PATTERN "*.dll"
    PATTERN "SPIRV-Tools-shared.dll" EXCLUDE
    PATTERN "vulkan-1.dll" EXCLUDE)

install(DIRECTORY "${ALPHASHAPE_RUNTIME_DLL_DIR}/" DESTINATION bin COMPONENT core
        ${ALPHASHAPE_TOOL_ONLY_DLL_PATTERNS})
install(FILES ${ALPHASHAPE_VC_RUNTIME_DLLS} DESTINATION bin COMPONENT core)
if(ALPHASHAPE_LIBOMP_DLL)
    install(FILES "${ALPHASHAPE_LIBOMP_DLL}" DESTINATION bin COMPONENT core)
endif()

# The Python module needs the same DLLs in its own directory.
if(ALPHASHAPE_BUILD_PYTHON AND TARGET alphashape_python)
    install(DIRECTORY "${ALPHASHAPE_RUNTIME_DLL_DIR}/"
            DESTINATION ${ALPHASHAPE_WINDOWS_PYTHON_DIR} COMPONENT python
            ${ALPHASHAPE_TOOL_ONLY_DLL_PATTERNS})
    install(FILES ${ALPHASHAPE_VC_RUNTIME_DLLS}
            DESTINATION ${ALPHASHAPE_WINDOWS_PYTHON_DIR} COMPONENT python)
    if(ALPHASHAPE_LIBOMP_DLL)
        install(FILES "${ALPHASHAPE_LIBOMP_DLL}"
                DESTINATION ${ALPHASHAPE_WINDOWS_PYTHON_DIR} COMPONENT python)
    endif()
endif()

# ---- Qt (windeployqt, at install time) ---------------------------------------------------------
# Running windeployqt inside the install tree rather than in the build tree is
# what makes CPack package its output: CPack stages the install, then builds the
# file lists from what is on disk. It also means the Qt DLLs and plugins are
# attributed to the gui component, so deselecting the GUI leaves no Qt behind.
if(ALPHASHAPE_BUILD_GUI AND TARGET alphashape-gui)
    # windeployqt.exe sits in <qt>/bin, which is where qmake is.
    get_target_property(_alphashape_qmake Qt6::qmake IMPORTED_LOCATION)
    if(NOT _alphashape_qmake)
        find_program(ALPHASHAPE_WINDEPLOYQT windeployqt HINTS "${QT_ROOT_DIR}/bin" PATH_SUFFIXES bin)
    else()
        get_filename_component(ALPHASHAPE_QT_BIN "${_alphashape_qmake}" DIRECTORY)
        set(ALPHASHAPE_WINDEPLOYQT "${ALPHASHAPE_QT_BIN}/windeployqt.exe")
    endif()
    if(NOT ALPHASHAPE_WINDEPLOYQT OR NOT EXISTS "${ALPHASHAPE_WINDEPLOYQT}")
        message(FATAL_ERROR
            "alphashape: windeployqt.exe not found (looked at '${ALPHASHAPE_WINDEPLOYQT}'); the "
            "GUI would be installed without Qt. Point CMAKE_PREFIX_PATH at a full Qt install.")
    endif()
    get_filename_component(ALPHASHAPE_WINDEPLOYQT "${ALPHASHAPE_WINDEPLOYQT}" ABSOLUTE)
    # Qt6_DIR is <qt>/lib/cmake/Qt6.
    get_filename_component(ALPHASHAPE_QT_PREFIX "${Qt6_DIR}/../../.." ABSOLUTE)
    set(ALPHASHAPE_QT_PLUGINS "${ALPHASHAPE_QT_PREFIX}/plugins")

    # --release: the build is Release, and the debug Qt DLLs are not shipped.
    # --no-translations: no page of the GUI is translated yet.
    # --no-opengl-sw: the Widgets GUI never needs the software OpenGL fallback,
    #   which is a 20 MB single file.
    # --no-compiler-runtime: the VC++ runtime is installed by the rules above.
    install(CODE "
        set(_bin \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/bin\")
        message(STATUS \"windeployqt: \${_bin}/alphashape.exe\")
        # windeployqt links Qt6Core itself, so its own directory must be on the
        # provider path; the Qt install is not on the runner's PATH.
        set(ENV{PATH} \"${ALPHASHAPE_QT_BIN};\$ENV{PATH}\")
        execute_process(
            COMMAND \"${ALPHASHAPE_WINDEPLOYQT}\" --release --no-translations
                    --no-opengl-sw --no-compiler-runtime --dir \"\${_bin}\"
                    \"\${_bin}/alphashape.exe\"
            RESULT_VARIABLE _rc)
        if(NOT _rc EQUAL 0)
            message(FATAL_ERROR \"windeployqt failed (\${_rc})\")
        endif()
        # windeployqt ships only the platform plugin of the machine it runs on
        # (qwindows). The offscreen one is what any headless run needs: the
        # --offscreen-smoke path, CI, and a user connecting over a service.
        # minimal is the generic fallback, copied when this Qt kit has it.
        foreach(_plugin qoffscreen qminimal)
            set(_src \"${ALPHASHAPE_QT_PLUGINS}/platforms/\${_plugin}.dll\")
            if(EXISTS \"\${_src}\")
                file(COPY \"\${_src}\" DESTINATION \"\${_bin}/platforms\")
                message(STATUS \"  added platform plugin \${_plugin}.dll\")
            elseif(_plugin STREQUAL \"qoffscreen\")
                message(FATAL_ERROR
                    \"alphashape: \${_src} is missing; headless runs could not start\")
            endif()
        endforeach()" COMPONENT gui)
endif()

# ---- PROJ data ---------------------------------------------------------------------------------
# <install>\share\proj is step 2 of the search in model/crs/proj_context.cpp: the
# executable in bin\ reaches it as ../share/proj, and the module in python\
# reaches it the same way through its own image directory.
set(ALPHASHAPE_PROJ_DATA_SRC "" CACHE PATH
    "Directory holding proj.db to bundle (defaults to the vcpkg share/proj)")
if(NOT ALPHASHAPE_PROJ_DATA_SRC AND VCPKG_INSTALLED_DIR AND VCPKG_TARGET_TRIPLET)
    file(TO_CMAKE_PATH "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share/proj"
         ALPHASHAPE_PROJ_DATA_SRC)
endif()
if(NOT EXISTS "${ALPHASHAPE_PROJ_DATA_SRC}/proj.db")
    message(FATAL_ERROR
        "alphashape: no proj.db under ALPHASHAPE_PROJ_DATA_SRC "
        "('${ALPHASHAPE_PROJ_DATA_SRC}'); the installed copy could not do a single "
        "coordinate transform.")
endif()

file(GLOB ALPHASHAPE_PROJ_DB_FILES
     "${ALPHASHAPE_PROJ_DATA_SRC}/*.db" "${ALPHASHAPE_PROJ_DATA_SRC}/*.ini"
     "${ALPHASHAPE_PROJ_DATA_SRC}/*.json")
install(FILES ${ALPHASHAPE_PROJ_DB_FILES} DESTINATION share/proj COMPONENT core)

# Grids smallest first until the budget is spent. Sorting by size (name breaks
# ties, so the choice is reproducible) favours many small regional grids over
# one continental file.
file(GLOB ALPHASHAPE_PROJ_GRID_FILES
     "${ALPHASHAPE_PROJ_DATA_SRC}/*.tif" "${ALPHASHAPE_PROJ_DATA_SRC}/*.gtx")
set(ALPHASHAPE_PROJ_GRID_SELECTED "")
set(ALPHASHAPE_PROJ_GRID_BYTES 0)
foreach(_grid IN LISTS ALPHASHAPE_PROJ_GRID_FILES)
    file(SIZE "${_grid}" _grid_size)
    if(_grid_size GREATER ALPHASHAPE_PROJ_GRID_MAX_BYTES)
        continue()
    endif()
    # if() compares, it does not compute: the remaining budget needs math().
    math(EXPR _grid_budget "${ALPHASHAPE_PROJ_TOTAL_MAX_BYTES} - ${ALPHASHAPE_PROJ_GRID_BYTES}")
    if(_grid_size GREATER _grid_budget)
        continue()
    endif()
    list(APPEND ALPHASHAPE_PROJ_GRID_SELECTED "${_grid}")
    math(EXPR ALPHASHAPE_PROJ_GRID_BYTES "${ALPHASHAPE_PROJ_GRID_BYTES} + ${_grid_size}")
endforeach()
list(LENGTH ALPHASHAPE_PROJ_GRID_SELECTED ALPHASHAPE_PROJ_GRID_COUNT)
math(EXPR ALPHASHAPE_PROJ_GRID_MIB "${ALPHASHAPE_PROJ_GRID_BYTES} / 1048576")
if(ALPHASHAPE_PROJ_GRID_COUNT EQUAL 0)
    # vcpkg's proj port ships proj.db but no grid files. That is workable --
    # transforms that need no grid work, and the ones that do fail offline with
    # PROJ's "grid not found" rather than silently downloading -- but it is worth
    # saying out loud, because a grid-less install cannot do a geoid transform.
    message(STATUS
        "  PROJ data ........ proj.db only, no grids in ${ALPHASHAPE_PROJ_DATA_SRC} "
        "(geoid transforms will need ALPHASHAPE_PROJ_DATA pointing at a full proj-data)")
else()
    message(STATUS
        "  PROJ data ........ ${ALPHASHAPE_PROJ_GRID_COUNT} grids, ${ALPHASHAPE_PROJ_GRID_MIB} MiB "
        "(${ALPHASHAPE_PROJ_GRID_MAX_BYTES} B per file, ${ALPHASHAPE_PROJ_TOTAL_MAX_BYTES} B total)")
endif()
install(FILES ${ALPHASHAPE_PROJ_GRID_SELECTED} DESTINATION share/proj COMPONENT core)

# ---- documentation and licences ----------------------------------------------------------------
install(FILES ${ALPHASHAPE_REPO_ROOT}/README.md
              ${ALPHASHAPE_REPO_ROOT}/LICENSE
              ${PROJECT_SOURCE_DIR}/LIMITATIONS.md
              ${ALPHASHAPE_PACKAGING_DIR}/README.md
              ${ALPHASHAPE_PACKAGING_DIR}/README-smartscreen.md
        DESTINATION LICENSES COMPONENT core)

string(TIMESTAMP ALPHASHAPE_RELEASE_YEAR "%Y" UTC)
# Qt's version for the notices, taken from the package that found it so a Qt
# upgrade cannot leave a stale number behind. The notices template names these
# three exactly: a Qt-version line rendered as "Qt " with no number is a licence
# notice that does not say which Qt it is about, so an unset version is a hard
# error rather than an empty substitution. (It also catches a template that
# drifts back to a bare @QT_VERSION@, which CPack does not define.)
set(ALPHASHAPE_QT_VERSION "${Qt6_VERSION}")
if(NOT ALPHASHAPE_QT_VERSION MATCHES "^[0-9]+\\.[0-9]+")
    message(FATAL_ERROR
        "alphashape: the third-party notices need Qt's version, but Qt6_VERSION is "
        "'${Qt6_VERSION}'. Configure with the GUI enabled (ALPHASHAPE_BUILD_GUI=ON) so "
        "find_package(Qt6) records it.")
endif()
string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" ALPHASHAPE_QT_VERSION_MAJOR
       "${ALPHASHAPE_QT_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*$" "\\1" ALPHASHAPE_QT_VERSION_MINOR
       "${ALPHASHAPE_QT_VERSION}")
configure_file(${ALPHASHAPE_PACKAGING_DIR}/THIRD-PARTY-NOTICES.txt.in
               ${CMAKE_BINARY_DIR}/windows/THIRD-PARTY-NOTICES.txt @ONLY)
# A placeholder that survived configuration (a misspelt name, or a CPack
# variable used where a CMake one belongs) would ship a literal "@...@" in a
# licence file. Fail at configure time instead.
file(READ ${CMAKE_BINARY_DIR}/windows/THIRD-PARTY-NOTICES.txt _alphashape_notices)
if(_alphashape_notices MATCHES "@[A-Za-z_][A-Za-z0-9_]*@")
    message(FATAL_ERROR
        "alphashape: THIRD-PARTY-NOTICES.txt still contains ${CMAKE_MATCH_0}; the "
        "template names a variable configure_file did not set.")
endif()
unset(_alphashape_notices)
install(FILES ${CMAKE_BINARY_DIR}/windows/THIRD-PARTY-NOTICES.txt
        DESTINATION LICENSES COMPONENT core)

# Qt is LGPL-3: whoever receives the DLLs must receive that licence text, so a
# Qt install that cannot supply it is a hard error rather than a smaller
# installer. aqtinstall (what install-qt-action downloads) unpacks Qt's LICENSES
# directory beside the version's architecture directory; a distro or a hand-built
# Qt keeps it in the prefix. Both are searched, then the vendored copy under
# packaging/licenses that ships with this repository.
set(ALPHASHAPE_QT_LICENSE_DIR "${ALPHASHAPE_QT_PREFIX}/LICENSES" CACHE PATH
    "Qt's LICENSES directory, copied into LICENSES/Qt")
set(ALPHASHAPE_VENDORED_LICENSE_DIR "${ALPHASHAPE_PACKAGING_DIR}/../licenses")
foreach(_license LGPL-3.0-only.txt GPL-3.0-only.txt LGPL-2.1-only.txt GPL-2.0-only.txt)
    set(_license_source "")
    foreach(_candidate "${ALPHASHAPE_QT_LICENSE_DIR}/${_license}"
                       "${ALPHASHAPE_QT_PREFIX}/../LICENSES/${_license}"
                       "${ALPHASHAPE_QT_PREFIX}/../../LICENSES/${_license}"
                       "${ALPHASHAPE_VENDORED_LICENSE_DIR}/${_license}")
        if(EXISTS "${_candidate}")
            set(_license_source "${_candidate}")
            break()
        endif()
    endforeach()
    if(NOT _license_source)
        # LGPL-3.0-only is the licence Qt itself is under; the others only cover
        # modules this installer does not link, so they stay optional.
        if(_license STREQUAL "LGPL-3.0-only.txt")
            message(FATAL_ERROR
                "alphashape: no ${_license} for the shipped Qt. Looked under "
                "'${ALPHASHAPE_QT_PREFIX}' and in '${ALPHASHAPE_VENDORED_LICENSE_DIR}'. "
                "Ship the text in packaging/licenses/ rather than producing an installer "
                "without it.")
        endif()
        continue()
    endif()
    install(FILES "${_license_source}" DESTINATION LICENSES/Qt COMPONENT gui)
endforeach()

# ---- third-party licence texts ----------------------------------------------------------------
# Every port whose runtime DLL is installed above has its licence text beside it
# in vcpkg's share/<port>/copyright. Copying the whole set rather than a hand
# list means a DLL cannot be added without its text: the glob follows whatever
# vcpkg actually installed, and the smoke test asserts the ones known to be
# shipped today are still there.
set(ALPHASHAPE_VCPKG_SHARE_DIR "")
if(VCPKG_INSTALLED_DIR AND VCPKG_TARGET_TRIPLET)
    file(TO_CMAKE_PATH "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/share"
         ALPHASHAPE_VCPKG_SHARE_DIR)
endif()
file(GLOB ALPHASHAPE_THIRD_PARTY_COPYRIGHTS "${ALPHASHAPE_VCPKG_SHARE_DIR}/*/copyright")
list(LENGTH ALPHASHAPE_THIRD_PARTY_COPYRIGHTS ALPHASHAPE_THIRD_PARTY_COUNT)
if(ALPHASHAPE_THIRD_PARTY_COUNT EQUAL 0)
    message(FATAL_ERROR
        "alphashape: no share/<port>/copyright under '${ALPHASHAPE_VCPKG_SHARE_DIR}'; the "
        "installer would ship third-party DLLs without their licence texts.")
endif()
foreach(_copyright IN LISTS ALPHASHAPE_THIRD_PARTY_COPYRIGHTS)
    get_filename_component(_port_dir "${_copyright}" DIRECTORY)
    get_filename_component(_port "${_port_dir}" NAME)
    install(FILES "${_copyright}"
            DESTINATION "LICENSES/third-party" RENAME "${_port}.txt" COMPONENT core)
endforeach()
message(STATUS "  licences ......... ${ALPHASHAPE_THIRD_PARTY_COUNT} third-party texts")

# libomp is not a vcpkg port here: it comes from the LLVM install beside
# clang-cl, whose principal licence text is <llvm>/LICENSE.TXT (Apache-2.0 with
# the LLVM exception). The file name differs between LLVM releases and an
# install may carry individual licences instead, so candidates are probed and
# the vendored Apache-2.0 text is the fallback.
set(ALPHASHAPE_LIBOMP_LICENSE "")
get_filename_component(ALPHASHAPE_LLVM_PREFIX "${ALPHASHAPE_COMPILER_BIN}" DIRECTORY)
foreach(_candidate
        "${ALPHASHAPE_LLVM_PREFIX}/LICENSE.TXT"
        "${ALPHASHAPE_LLVM_PREFIX}/LICENSE.txt"
        "${ALPHASHAPE_LLVM_PREFIX}/licenses/Apache-2.0.txt"
        "${ALPHASHAPE_VENDORED_LICENSE_DIR}/Apache-2.0.txt")
    if(EXISTS "${_candidate}")
        set(ALPHASHAPE_LIBOMP_LICENSE "${_candidate}")
        break()
    endif()
endforeach()
if(NOT ALPHASHAPE_LIBOMP_LICENSE)
    message(FATAL_ERROR
        "alphashape: libomp.dll is shipped but no Apache-2.0 licence text was found under "
        "'${ALPHASHAPE_LLVM_PREFIX}' or '${ALPHASHAPE_VENDORED_LICENSE_DIR}'.")
endif()
install(FILES "${ALPHASHAPE_LIBOMP_LICENSE}"
        DESTINATION LICENSES/third-party RENAME "libomp.txt" COMPONENT core)

# The fonts are compiled into alphashape.exe; their OFL texts must ship with it.
install(FILES ${PROJECT_SOURCE_DIR}/src/apps/gui/fonts/OFL-Inter.txt
              ${PROJECT_SOURCE_DIR}/src/apps/gui/fonts/OFL-JetBrainsMono.txt
        DESTINATION LICENSES/Fonts COMPONENT core)

# ---- CPack: NSIS --------------------------------------------------------------------------------
set(CPACK_GENERATOR "NSIS")
set(CPACK_PACKAGE_NAME "alphashape")
set(CPACK_PACKAGE_VERSION "${ALPHASHAPE_PACKAGE_VERSION}")
# NSIS's VIProductVersion takes a four-number quadruple and CPack does not
# derive one, so the numbers the GUI's version resource already uses are given
# to CPack as well; the template names them.
set(CPACK_PACKAGE_VERSION_MAJOR "${ALPHASHAPE_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${ALPHASHAPE_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${ALPHASHAPE_VERSION_PATCH}")
set(CPACK_PACKAGE_VENDOR "alphashape")
set(CPACK_PACKAGE_CONTACT "Aniket Singh <16048767+aniketpratapsingh@users.noreply.github.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Open-source photogrammetry engine")
set(CPACK_PACKAGE_DESCRIPTION
    "alphashape turns overlapping photographs into georeferenced point clouds, meshes, textured models, DEMs and orthomosaics. This installer contains the desktop app, the command-line app and the Python module. GPU compute uses Vulkan and falls back to the CPU.")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/aniketpratapsingh/alphashape")
set(CPACK_PACKAGE_EXECUTABLES "alphashape" "alphashape")
set(CPACK_RESOURCE_FILE_LICENSE "${ALPHASHAPE_REPO_ROOT}/LICENSE")
set(CPACK_RESOURCE_FILE_README "${ALPHASHAPE_REPO_ROOT}/README.md")
set(CPACK_PACKAGE_FILE_NAME
    "alphashape-${ALPHASHAPE_PACKAGE_VERSION}-windows-x86_64")
# CPack stages the install under $CPACK_PACKAGE_DIRECTORY/_CPack_Packages and
# writes the installer to $CPACK_OUTPUT_FILE_PREFIX. Both are given here so a
# `cpack --config build/CPackConfig.cmake` run from anywhere leaves the source
# tree alone (the default for the staging directory is the current directory).
set(CPACK_PACKAGE_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CPACK_OUTPUT_FILE_PREFIX "${CMAKE_BINARY_DIR}/dist")

# No version in the directory or the registry key: an upgrade then replaces the
# previous install instead of stacking beside it.
set(CPACK_PACKAGE_INSTALL_DIRECTORY "alphashape")
set(CPACK_PACKAGE_INSTALL_REGISTRY_KEY "alphashape")
set(CPACK_NSIS_DISPLAY_NAME "alphashape")
set(CPACK_NSIS_PACKAGE_NAME "alphashape ${ALPHASHAPE_PACKAGE_VERSION}")
set(CPACK_NSIS_URL_INFO_ABOUT "${CPACK_PACKAGE_HOMEPAGE_URL}")
set(CPACK_NSIS_HELP_LINK "${CPACK_PACKAGE_HOMEPAGE_URL}/issues")
set(CPACK_NSIS_CONTACT "${CPACK_PACKAGE_CONTACT}")
# Written into NSIS as "$INSTDIR\<this>", so it is a forward-slash relative path:
# a backslash in a CMake string is an escape ("\a" is not one) and the generated
# CPackConfig.cmake would be rejected.
set(CPACK_NSIS_INSTALLED_ICON_NAME "bin/alphashape.exe")
# Installer, uninstaller and shortcut icon: the rendered product mark.
set(CPACK_NSIS_MUI_ICON "${ALPHASHAPE_ICON}")
set(CPACK_NSIS_MUI_UNIICON "${ALPHASHAPE_ICON}")
set(CPACK_PACKAGE_ICON "${ALPHASHAPE_ICON}")

# The template is a custom one: CPack looks for NSIS.template.in on
# CPACK_MODULE_PATH, which cpack installs as CMAKE_MODULE_PATH while it loads the
# configuration (cmCPackGenerator::FindTemplate / cpack.cxx). Everything else in
# that directory would be found first, so the project's own module path is kept
# after it.
set(CPACK_MODULE_PATH "${ALPHASHAPE_PACKAGING_DIR};${CPACK_MODULE_PATH}")
set(CPACK_NSIS_COMPRESSOR "lzma")

# Installed but not offered on the components page: the DLLs, PROJ data and
# licences every other component needs.
set(CPACK_COMPONENT_CORE_DISPLAY_NAME "Runtime")
set(CPACK_COMPONENT_CORE_DESCRIPTION
    "Shared libraries, coordinate-system data and licences (required)")
set(CPACK_COMPONENT_CORE_REQUIRED ON)
set(CPACK_COMPONENT_CORE_HIDDEN ON)

set(CPACK_COMPONENT_GUI_DISPLAY_NAME "Desktop app")
set(CPACK_COMPONENT_GUI_DESCRIPTION "The alphashape window, with Qt")
set(CPACK_COMPONENT_CLI_DISPLAY_NAME "Command-line app")
set(CPACK_COMPONENT_CLI_DESCRIPTION "alphashape-cli, for scripts and batch runs")
set(CPACK_COMPONENT_PYTHON_DISPLAY_NAME "Python module")
set(CPACK_COMPONENT_PYTHON_DESCRIPTION
    "alphashape.pyd, for the Python 3 interpreter the build used")

include(CPack)
