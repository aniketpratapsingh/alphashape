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

# ALPHASHAPE_PACKAGE_VERSION lets a release stamp its tag (v0.2.0) into the
# installer while the source tree keeps its own version. "dev" is what the
# reusable workflow passes for a plain CI run, and it belongs in no installer's
# DisplayVersion.
if(NOT ALPHASHAPE_PACKAGE_VERSION OR ALPHASHAPE_PACKAGE_VERSION STREQUAL "dev")
    set(ALPHASHAPE_PACKAGE_VERSION "${PROJECT_VERSION}")
endif()

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
    # The resource's FILEVERSION is a numeric quadruple; a release version like
    # "v0.2.0" or "0.1.0-nightly" is not. Strip the tag's leading v, take the
    # numeric prefix and pad to four fields.
    string(REGEX REPLACE "^[vV]" "" _version_numeric "${ALPHASHAPE_PACKAGE_VERSION}")
    string(REGEX MATCH "^[0-9]+(\\.[0-9]+)*" _version_numeric "${_version_numeric}")
    if(NOT _version_numeric)
        set(_version_numeric "${PROJECT_VERSION}")
    endif()
    string(REPLACE "." ";" _version_parts "${_version_numeric}")
    set(_version_rc "")
    foreach(_part RANGE 0 3)
        list(LENGTH _version_parts _count)
        if(_part LESS _count)
            list(GET _version_parts ${_part} _value)
        else()
            set(_value 0)
        endif()
        string(APPEND _version_rc "${_value},")
    endforeach()
    string(REGEX REPLACE ",$" "" _version_rc "${_version_rc}")

    set(ALPHASHAPE_WINDOWS_RC "${CMAKE_BINARY_DIR}/windows/alphashape.rc")
    file(WRITE "${ALPHASHAPE_WINDOWS_RC}"
         "IDI_ICON1 ICON DISCARDABLE \"${ALPHASHAPE_ICON}\"\n"
         "VS_VERSION_INFO VERSIONINFO\n"
         "FILEVERSION ${_version_rc}\n"
         "PRODUCTVERSION ${_version_rc}\n"
         "BEGIN\n"
         "  BLOCK \"StringFileInfo\"\n"
         "  BEGIN\n"
         "    BLOCK \"040904b0\"\n"
         "    BEGIN\n"
         "      VALUE \"FileDescription\", \"alphashape\"\n"
         "      VALUE \"FileVersion\", \"${ALPHASHAPE_PACKAGE_VERSION}\"\n"
         "      VALUE \"ProductName\", \"alphashape\"\n"
         "      VALUE \"ProductVersion\", \"${ALPHASHAPE_PACKAGE_VERSION}\"\n"
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

install(DIRECTORY "${ALPHASHAPE_RUNTIME_DLL_DIR}/" DESTINATION bin COMPONENT core
        FILES_MATCHING PATTERN "*.dll")
install(FILES ${ALPHASHAPE_VC_RUNTIME_DLLS} DESTINATION bin COMPONENT core)
if(ALPHASHAPE_LIBOMP_DLL)
    install(FILES "${ALPHASHAPE_LIBOMP_DLL}" DESTINATION bin COMPONENT core)
endif()

# The Python module needs the same DLLs in its own directory.
if(ALPHASHAPE_BUILD_PYTHON AND TARGET alphashape_python)
    install(DIRECTORY "${ALPHASHAPE_RUNTIME_DLL_DIR}/"
            DESTINATION ${ALPHASHAPE_WINDOWS_PYTHON_DIR} COMPONENT python
            FILES_MATCHING PATTERN "*.dll")
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
# upgrade cannot leave a stale number behind.
set(ALPHASHAPE_QT_VERSION "${Qt6_VERSION}")
string(REGEX REPLACE "^([0-9]+)\\..*$" "\\1" ALPHASHAPE_QT_VERSION_MAJOR
       "${ALPHASHAPE_QT_VERSION}")
string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*$" "\\1" ALPHASHAPE_QT_VERSION_MINOR
       "${ALPHASHAPE_QT_VERSION}")
configure_file(${ALPHASHAPE_PACKAGING_DIR}/THIRD-PARTY-NOTICES.txt.in
               ${CMAKE_BINARY_DIR}/windows/THIRD-PARTY-NOTICES.txt @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/windows/THIRD-PARTY-NOTICES.txt
        DESTINATION LICENSES COMPONENT core)

# Qt is LGPL-3 and links against GPL-3 code paths only when asked to; the licence
# texts Qt does not have windeployqt copy are copied here. A Qt install without
# its LICENSES directory (some binary-only packs) is tolerated, the rest is not.
set(ALPHASHAPE_QT_LICENSE_DIR "${ALPHASHAPE_QT_PREFIX}/../LICENSES" CACHE PATH
    "Qt's LICENSES directory, copied into LICENSES/Qt")
foreach(_license LGPL-3.0-only.txt GPL-3.0-only.txt GPL-2.0-only.txt)
    if(EXISTS "${ALPHASHAPE_QT_LICENSE_DIR}/${_license}")
        install(FILES "${ALPHASHAPE_QT_LICENSE_DIR}/${_license}"
                DESTINATION LICENSES/Qt COMPONENT gui)
    endif()
endforeach()

# The fonts are compiled into alphashape.exe; their OFL texts must ship with it.
install(FILES ${PROJECT_SOURCE_DIR}/src/apps/gui/fonts/OFL-Inter.txt
              ${PROJECT_SOURCE_DIR}/src/apps/gui/fonts/OFL-JetBrainsMono.txt
        DESTINATION LICENSES/Fonts COMPONENT core)

# ---- CPack: NSIS --------------------------------------------------------------------------------
set(CPACK_GENERATOR "NSIS")
set(CPACK_PACKAGE_NAME "alphashape")
set(CPACK_PACKAGE_VERSION "${ALPHASHAPE_PACKAGE_VERSION}")
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
