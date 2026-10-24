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
    task_dialog.cpp
    workflow.cpp
    main_window.cpp
    workspace_model.cpp
    workspace_pane.cpp
    properties_pane.cpp
    viewer/camera_controller.cpp
    viewer/scene.cpp
    viewer/viewport_renderer.cpp
    viewer/viewport_widget.cpp
    theme/theme.cpp)
target_include_directories(alphashape_gui_lib PUBLIC ${ALPHASHAPE_INCLUDE_DIR})
target_link_libraries(alphashape_gui_lib
    PUBLIC Qt6::Widgets alphashape_algorithms alphashape_core
    PRIVATE alphashape_warnings Qt6::GuiPrivate)

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

# The viewport shaders. GLSL 440 in, .qsb out: qsb (Qt Shader Tools) translates them to SPIR-V,
# HLSL and MSL at build time, so the renderer loads one resource per stage on every backend.
find_package(Qt6 REQUIRED COMPONENTS ShaderTools)
qt_add_shaders(alphashape_gui_lib alphashape_shaders
    PREFIX /alphashape/shaders
    BASE ${CMAKE_CURRENT_SOURCE_DIR}/viewer/shaders
    FILES
        viewer/shaders/mesh.vert
        viewer/shaders/mesh.frag
        viewer/shaders/points.vert
        viewer/shaders/points.frag
        viewer/shaders/lines.vert
        viewer/shaders/lines.frag)

add_executable(alphashape-gui main.cpp)
target_link_libraries(alphashape-gui PRIVATE alphashape_gui_lib alphashape_warnings)
target_compile_definitions(alphashape-gui PRIVATE ALPHASHAPE_GUI_VERSION="${PROJECT_VERSION}")
# Console subsystem on Windows for now: --help, --version and the smoke run write to stdout.
# The installer WP switches the shipped binary to a windowed one.
