# Install layout and Linux packages (.deb via CPack; the AppImage is assembled from the same
# install tree by packaging/linux/build-appimage.sh).
#
# Layout (CMAKE_INSTALL_PREFIX=/usr for the .deb):
#   bin/alphashape-cli, bin/alphashape-gui
#   lib/python3/dist-packages/alphashape*.so      Debian's system-Python module directory
#   lib/alphashape/qt/{lib,plugins}               private Qt 6.8 (ALPHASHAPE_BUNDLE_QT, .deb only)
#   share/applications, share/icons, share/metainfo
#   share/doc/alphashape                          README, licences, limitations
#   share/proj                                    PROJ data (ALPHASHAPE_BUNDLE_PROJ_DATA, AppImage)
# Shaders are embedded in the binaries (src/gpu), so there are no runtime shader files.

include(GNUInstallDirs)

set(ALPHASHAPE_PACKAGING_DIR "${PROJECT_SOURCE_DIR}/../packaging/linux")
set(ALPHASHAPE_REPO_ROOT "${PROJECT_SOURCE_DIR}/..")

option(ALPHASHAPE_BUNDLE_QT
    "Install a private copy of Qt (libs + plugins) under lib/alphashape/qt (Linux .deb)" OFF)
option(ALPHASHAPE_BUNDLE_PROJ_DATA
    "Install proj.db and the PROJ grids to share/proj so the package needs no proj-data (AppImage)" OFF)
set(ALPHASHAPE_PROJ_DATA_SRC "/usr/share/proj" CACHE PATH "Directory holding proj.db to bundle")
set(ALPHASHAPE_PYTHON_INSTALL_DIR "lib/python3/dist-packages" CACHE STRING
    "Where the Python module is installed (relative to the prefix)")

# ---- binaries ---------------------------------------------------------------------------------
if(ALPHASHAPE_BUILD_CLI)
    install(TARGETS alphashape-cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)
endif()

if(ALPHASHAPE_BUILD_GUI AND TARGET alphashape-gui)
    if(ALPHASHAPE_BUNDLE_QT)
        set_target_properties(alphashape-gui PROPERTIES
            INSTALL_RPATH "\$ORIGIN/../lib/alphashape/qt/lib")
    else()
        set_target_properties(alphashape-gui PROPERTIES INSTALL_RPATH "\$ORIGIN/../lib")
    endif()
    install(TARGETS alphashape-gui RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT runtime)

    # Desktop integration. The icon is a scalable SVG in the accent colour of the design tokens.
    install(FILES ${ALPHASHAPE_PACKAGING_DIR}/org.alphashape.app.desktop
            DESTINATION ${CMAKE_INSTALL_DATADIR}/applications COMPONENT runtime)
    install(FILES ${ALPHASHAPE_PACKAGING_DIR}/org.alphashape.app.svg
            DESTINATION ${CMAKE_INSTALL_DATADIR}/icons/hicolor/scalable/apps COMPONENT runtime)
    string(TIMESTAMP ALPHASHAPE_RELEASE_DATE "%Y-%m-%d" UTC)
    configure_file(${ALPHASHAPE_PACKAGING_DIR}/org.alphashape.app.metainfo.xml.in
                   ${CMAKE_BINARY_DIR}/org.alphashape.app.metainfo.xml @ONLY)
    install(FILES ${CMAKE_BINARY_DIR}/org.alphashape.app.metainfo.xml
            DESTINATION ${CMAKE_INSTALL_DATADIR}/metainfo COMPONENT runtime)
endif()

# ---- Python module: lands where the distribution's python3 already looks --------------------------
if(ALPHASHAPE_BUILD_PYTHON AND TARGET alphashape_python)
    install(TARGETS alphashape_python LIBRARY DESTINATION ${ALPHASHAPE_PYTHON_INSTALL_DIR}
            COMPONENT python)
endif()

# ---- documentation and licences ---------------------------------------------------------------------
install(FILES ${ALPHASHAPE_REPO_ROOT}/README.md ${ALPHASHAPE_REPO_ROOT}/LICENSE
              ${PROJECT_SOURCE_DIR}/LIMITATIONS.md
        DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT runtime)
install(FILES ${ALPHASHAPE_PACKAGING_DIR}/copyright
        DESTINATION ${CMAKE_INSTALL_DOCDIR} COMPONENT runtime)

# ---- PROJ data (packages that cannot rely on the distribution's proj-data) -------------------------
if(ALPHASHAPE_BUNDLE_PROJ_DATA)
    # <exe_dir>/../share/proj is step 2 of the data-directory search in model/crs/proj_context.cpp.
    install(DIRECTORY ${ALPHASHAPE_PROJ_DATA_SRC}/ DESTINATION ${CMAKE_INSTALL_DATADIR}/proj
            COMPONENT runtime FILES_MATCHING PATTERN "*.db" PATTERN "*.ini" PATTERN "*.tif"
            PATTERN "*.gtx" PATTERN "*.json")
endif()

# ---- private Qt for the .deb --------------------------------------------------------------------------
if(ALPHASHAPE_BUNDLE_QT AND ALPHASHAPE_BUILD_GUI)
    # Qt6_DIR is <qt>/lib/cmake/Qt6.
    get_filename_component(QT6_INSTALL_PREFIX "${Qt6_DIR}/../../.." ABSOLUTE)
    if(NOT EXISTS "${QT6_INSTALL_PREFIX}/plugins/platforms")
        message(FATAL_ERROR "ALPHASHAPE_BUNDLE_QT: no Qt plugins under ${QT6_INSTALL_PREFIX}")
    endif()
    install(CODE "
        execute_process(
            COMMAND bash \"${ALPHASHAPE_PACKAGING_DIR}/bundle-qt.sh\"
                    \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\" \"${QT6_INSTALL_PREFIX}\"
            RESULT_VARIABLE _bundle_rc)
        if(NOT _bundle_rc EQUAL 0)
            message(FATAL_ERROR \"bundle-qt.sh failed (\${_bundle_rc})\")
        endif()"
        COMPONENT runtime)
endif()

# ---- CPack: .deb for Ubuntu 24.04 ------------------------------------------------------------------
set(CPACK_PACKAGE_NAME "alphashape")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_CONTACT "Aniket Singh <16048767+aniketpratapsingh@users.noreply.github.com>")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Open-source photogrammetry engine")
set(CPACK_PACKAGE_DESCRIPTION
    "alphashape turns overlapping photographs into georeferenced point clouds, meshes, textured\n models, DEMs and orthomosaics. This package contains the command-line app, the Python\n module and the desktop GUI. GPU compute uses Vulkan and falls back to the CPU.")
set(CPACK_PACKAGE_HOMEPAGE_URL "https://github.com/aniketpratapsingh/alphashape")
set(CPACK_RESOURCE_FILE_LICENSE "${ALPHASHAPE_REPO_ROOT}/LICENSE")
set(CPACK_GENERATOR "DEB")
set(CPACK_SET_DESTDIR ON)
set(CPACK_INSTALL_PREFIX "/usr")
set(CPACK_PACKAGING_INSTALL_PREFIX "/usr")

set(CPACK_DEBIAN_FILE_NAME DEB-DEFAULT)
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
set(CPACK_DEBIAN_PACKAGE_SECTION "science")
set(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "${CPACK_PACKAGE_HOMEPAGE_URL}")
set(CPACK_DEBIAN_COMPRESSION_TYPE "xz")
# dpkg-shlibdeps cannot resolve the privately bundled Qt, and the set of distribution libraries is
# small and stable, so the list is explicit. CI audits it: every non-bundled library the binaries
# resolve to must be owned by a package named here (packaging/linux/audit-deb.sh).
set(ALPHASHAPE_DEB_DEPENDS
    "libc6 (>= 2.39)"
    "libstdc++6 (>= 14)"
    "libgcc-s1"
    "libgomp1"
    "libceres4t64"
    "libproj25"
    "proj-data"
    "libjpeg-turbo8"
    "libpng16-16t64"
    "libtiff6"
    "zlib1g"
    "libvulkan1 (>= 1.3)"
    "python3 (>= 3.12)"
    "python3 (<< 3.13)"
    # Libraries of the bundled Qt 6.8 that Qt does not ship itself.
    "libgl1" "libegl1" "libopengl0" "libglx0" "libfontconfig1" "libfreetype6" "libdbus-1-3"
    "libglib2.0-0t64" "libx11-6" "libx11-xcb1" "libxcb1" "libxkbcommon0" "libxkbcommon-x11-0"
    "libxcb-cursor0" "libxcb-icccm4" "libxcb-image0" "libxcb-keysyms1" "libxcb-randr0"
    "libxcb-render-util0" "libxcb-render0" "libxcb-shape0" "libxcb-shm0" "libxcb-sync1"
    "libxcb-xfixes0" "libxcb-xinerama0" "libxcb-xkb1" "libxcb-glx0")
list(JOIN ALPHASHAPE_DEB_DEPENDS ", " CPACK_DEBIAN_PACKAGE_DEPENDS)
set(CPACK_DEBIAN_PACKAGE_RECOMMENDS "mesa-vulkan-drivers")
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS OFF)
set(CPACK_DEBIAN_PACKAGE_GENERATE_SHLIBS OFF)
set(CPACK_STRIP_FILES ON)
include(CPack)
