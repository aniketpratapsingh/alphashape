# macOS app bundle inputs. The bundle itself is assembled from the build tree by
# packaging/macos/build-app.sh (the GUI target is not a MACOSX_BUNDLE target, so the tests keep
# finding the plain executable). This only renders the Info.plist template with the project
# version; the script fills in LSMinimumSystemVersion from the binaries it bundles.
configure_file(${PROJECT_SOURCE_DIR}/../packaging/macos/Info.plist.in
               ${CMAKE_BINARY_DIR}/macos/Info.plist @ONLY)
