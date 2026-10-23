# alphashape-gui: the Qt 6 desktop front end. Only configured with ALPHASHAPE_BUILD_GUI.
#
# The code lives in a static library so the tests link exactly what the executable runs.
# Qt is linked dynamically (LGPL-3).

set(CMAKE_AUTOMOC ON)

set(ALPHASHAPE_DESIGN_DIR ${PROJECT_SOURCE_DIR}/../design/system)

add_library(alphashape_gui_lib STATIC
    command_line.cpp
    app.cpp
    job_manager.cpp
    main_window.cpp
    viewport_placeholder.cpp
    workspace_model.cpp
    workspace_pane.cpp
    properties_pane.cpp
    theme/theme.cpp)
target_include_directories(alphashape_gui_lib PUBLIC ${ALPHASHAPE_INCLUDE_DIR})
target_link_libraries(alphashape_gui_lib
    PUBLIC Qt6::Widgets alphashape_algorithms alphashape_core
    PRIVATE alphashape_warnings)

# The engine uses `slots` and `signals` as ordinary identifiers (core/assets.hpp), which Qt's
# keyword macros would break, so Qt code here spells them Q_SLOTS / Q_SIGNALS / Q_EMIT.
target_compile_definitions(alphashape_gui_lib PUBLIC QT_NO_KEYWORDS)
target_compile_definitions(alphashape_gui_lib PRIVATE ALPHASHAPE_GUI_VERSION="${PROJECT_VERSION}")

# The style sheets are generated from design/system/tokens.json by tools/theme/compile_qss.py
# and checked in; they are embedded so the binary has no files to find at run time.
qt_add_resources(alphashape_gui_lib alphashape_theme
    PREFIX /alphashape/theme
    BASE ${ALPHASHAPE_DESIGN_DIR}
    FILES
        ${ALPHASHAPE_DESIGN_DIR}/alphashape-dark.qss
        ${ALPHASHAPE_DESIGN_DIR}/alphashape-light.qss)

# Bundled fonts (SIL Open Font License 1.1; the licence texts ship beside them).
qt_add_resources(alphashape_gui_lib alphashape_fonts
    PREFIX /alphashape/fonts
    BASE ${CMAKE_CURRENT_SOURCE_DIR}/fonts
    FILES
        fonts/Inter-Variable.ttf
        fonts/JetBrainsMono-Variable.ttf
        fonts/OFL-Inter.txt
        fonts/OFL-JetBrainsMono.txt)

add_executable(alphashape-gui main.cpp)
target_link_libraries(alphashape-gui PRIVATE alphashape_gui_lib alphashape_warnings)
target_compile_definitions(alphashape-gui PRIVATE ALPHASHAPE_GUI_VERSION="${PROJECT_VERSION}")
# Console subsystem on every platform, including the installed Windows copy: --help,
# --version, --offscreen-smoke and diagnostics all write to stdout, and the packaging
# smoke tests read it. A GUI-subsystem build (no console window beside the app) is a
# separate change: it needs the WinMain entry point, and every test that captures this
# executable's output would have to stop relying on it. See engine/LIMITATIONS.md.
