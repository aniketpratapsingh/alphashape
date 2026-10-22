if(ALPHASHAPE_BUILD_CLI)
    add_executable(alphashape-cli main.cpp)
    target_link_libraries(alphashape-cli PRIVATE alphashape_algorithms)
endif()
