# GUI tests: Theme.*, GuiShell.*, GuiApp.*, CommandLine.*, Jobs.*, TaskDialog.*, Workflow.*,
# MainWindow.*, WorkspaceTree.* (MainWindow on the
# harness in harness.hpp). Registered with ctest like the engine
# suite, so they run in the CI shard lanes. QT_QPA_PLATFORM=offscreen: no display needed.
# harness in harness.hpp). Registered with ctest like the engine
# suite, so they run in the CI shard lanes. QT_QPA_PLATFORM=offscreen: no display needed.

find_package(GTest REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(CMAKE_AUTOMOC OFF)

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
    test_workspace.cpp)
target_link_libraries(alphashape_gui_tests PRIVATE alphashape_gui_lib Qt6::Test GTest::gtest
                                              alphashape_warnings)
add_dependencies(alphashape_gui_tests alphashape-gui)

file(TO_CMAKE_PATH "${Python3_EXECUTABLE}" _alphashape_python)
file(TO_CMAKE_PATH "${PROJECT_SOURCE_DIR}/.." _alphashape_repo_root)
target_compile_definitions(alphashape_gui_tests PRIVATE
    ALPHASHAPE_GUI_EXE="$<TARGET_FILE:alphashape-gui>"
    ALPHASHAPE_GUI_DATA_DIR="${PROJECT_SOURCE_DIR}/data"
    ALPHASHAPE_REPO_ROOT="${_alphashape_repo_root}"
    ALPHASHAPE_PYTHON="${_alphashape_python}")

include(GoogleTest)
gtest_discover_tests(alphashape_gui_tests
    PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen" TIMEOUT 120)
