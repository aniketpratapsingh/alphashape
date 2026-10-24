# GUI tests: Theme.*, GuiShell.*, GuiApp.*, CommandLine.*, Jobs.*, TaskDialog.*, Workflow.*,
# StartScreen.*, ExportMenu.*, MainWindow.*, WorkspaceTree.*, PhotosPane.*, LogPane.*, Undo.*,
# ViewerCamera.* and ViewerScene.* (MainWindow on the harness in harness.hpp), plus two binaries
# of their own: the viewport's renderer (ViewerRender.*, see below) and the Phase 1 exit check
# (GuiPipelineEndToEnd.*, also below). Registered with ctest like the engine suite, so they run
# in the CI shard lanes. QT_QPA_PLATFORM=offscreen: no display needed.

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
    test_start_screen.cpp
    test_export_menu.cpp
    test_shell.cpp
    test_workspace.cpp
    test_photos_log.cpp
    test_viewer_camera.cpp
    test_viewer_scene.cpp)
target_link_libraries(alphashape_gui_tests PRIVATE alphashape_gui_lib Qt6::Test GTest::gtest
                                              alphashape_warnings)
add_dependencies(alphashape_gui_tests alphashape-gui)

# The Phase 1 exit check: one window, one project, the whole pipeline through the Workflow menu
# and the JobManager. Its own binary because it needs a timeout of its own (a per-test property
# is not available on a gtest_discover_tests() call, and 120 s would turn a slow CI runner into a
# failure). Same offscreen platform and the same harness as the suite above.
add_executable(alphashape_gui_e2e_tests
    test_main.cpp
    harness.cpp
    test_pipeline_e2e.cpp)
target_link_libraries(alphashape_gui_e2e_tests PRIVATE alphashape_gui_lib Qt6::Test
                                                       GTest::gtest alphashape_warnings)
add_dependencies(alphashape_gui_e2e_tests alphashape-gui)

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
foreach(_target alphashape_gui_tests alphashape_gui_e2e_tests alphashape_gui_render_tests)
    target_compile_definitions(${_target} PRIVATE
        ALPHASHAPE_GUI_EXE="$<TARGET_FILE:alphashape-gui>"
        ALPHASHAPE_GUI_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
        ALPHASHAPE_REPO_ROOT="${_alphashape_repo_root}"
        ALPHASHAPE_PYTHON="${_alphashape_python}")
endforeach()

include(GoogleTest)
gtest_discover_tests(alphashape_gui_tests
    PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 120)

# The pipeline runs the dense stage end to end once, so it gets a timeout that a slow shared
# runner cannot trip. The test itself waits a bounded time per stage and reports the stage name,
# so a real hang is diagnosed by the test, not by ctest killing the process silently.
gtest_discover_tests(alphashape_gui_e2e_tests
    PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 1800
    DISCOVERY_TIMEOUT 120)

# QT_QPA_PLATFORM is deliberately NOT set here: the render tests need whatever real platform the
# host has, and their main() deals with a missing display itself.
gtest_discover_tests(alphashape_gui_render_tests
    PROPERTIES TIMEOUT 180)
