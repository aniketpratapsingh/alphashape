# GUI tests: Theme.*, GuiShell.*, GuiApp.*, CommandLine.*, Jobs.*, TaskDialog.*, Workflow.*,
# MainWindow.*, WorkspaceTree.*, PhotosPane.*, LogPane.*, Undo.*, ViewerCamera.* and ViewerScene.*
# (MainWindow on the harness in harness.hpp), plus the viewport's renderer in its own binary
# (ViewerRender.*, see below). Registered with ctest like the engine suite, so they run in the CI
# shard lanes. QT_QPA_PLATFORM=offscreen: no display needed.

find_package(GTest REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CMAKE_AUTOMOC OFF)

# Everything here needs no display: QT_QPA_PLATFORM=offscreen, set per test by
# gtest_discover_tests below. The camera and scene suites are pure data and run everywhere.
add_executable(alphashape_gui_tests
    test_main.cpp
    harness.cpp
    test_main_window.cpp
    test_command_line.cpp
    test_theme.cpp
    test_gui_app.cpp
    test_jobs.cpp
    test_task_dialog.cpp
    test_workflow.cpp
    test_shell.cpp
    test_workspace.cpp
    test_photos_log.cpp
    test_viewer_camera.cpp
    test_viewer_scene.cpp)
target_link_libraries(alphashape_gui_tests PRIVATE alphashape_gui_lib Qt6::Test GTest::gtest
                                              alphashape_warnings)
add_dependencies(alphashape_gui_tests alphashape-gui)

# The render tests are the one GUI suite that needs a platform integration with a graphics API
# (the offscreen plugin has none) and the RHI headers (Qt6::GuiPrivate, which the library links
# PRIVATE and therefore does not re-export). They live in their own binary because a process can
# only have one QGuiApplication and one QPA platform, and its main() re-executes itself under
# xvfb-run on a display-less Linux host. It never fails for a missing GPU: it falls back to
# QRhi's Null backend and says so.
add_executable(alphashape_gui_render_tests
    viewer_render_main.cpp
    test_viewer_render.cpp)
target_link_libraries(alphashape_gui_render_tests PRIVATE alphashape_gui_lib GTest::gtest
                                                         Qt6::GuiPrivate alphashape_warnings)

file(TO_CMAKE_PATH "${Python3_EXECUTABLE}" _alphashape_python)
file(TO_CMAKE_PATH "${PROJECT_SOURCE_DIR}/.." _alphashape_repo_root)
foreach(_target alphashape_gui_tests alphashape_gui_render_tests)
    target_compile_definitions(${_target} PRIVATE
        ALPHASHAPE_GUI_EXE="$<TARGET_FILE:alphashape-gui>"
        ALPHASHAPE_GUI_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
        ALPHASHAPE_REPO_ROOT="${_alphashape_repo_root}"
        ALPHASHAPE_PYTHON="${_alphashape_python}")
endforeach()

include(GoogleTest)
gtest_discover_tests(alphashape_gui_tests
    PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 120)

# QT_QPA_PLATFORM is deliberately NOT set here: the render tests need whatever real platform the
# host has, and their main() deals with a missing display itself.
gtest_discover_tests(alphashape_gui_render_tests
    PROPERTIES TIMEOUT 180)
