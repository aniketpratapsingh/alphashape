# Windows installer (NSIS)

x86-64 only. Unsigned: see `README-smartscreen.md`.

| File | Purpose |
|------|---------|
| `NSIS.template.in` | The installer script CPack configures and runs (`cpack -G NSIS`) |
| `smoke.ps1` | Install, run and uninstall check, executed on a fresh runner |
| `python-usage.txt` | Shipped in `python\`, beside the module |
| `THIRD-PARTY-NOTICES.txt.in` | Rendered into `LICENSES\` at configure time |
| `README-smartscreen.md` | What a SmartScreen warning means and how to verify the download |

Install rules and CPack settings live in `engine/cmake/alphashape_windows_install.cmake`.
The vendored licence texts (`packaging/licenses/`) are the fallback for a Qt or
LLVM install that does not carry its own.

## Installer metadata

The `.exe` carries a version resource (ProductName, FileVersion, CompanyName
"alphashape contributors", LegalCopyright "GPL-3.0-or-later") built from the
package version: the tag's leading `v` is stripped and the numeric prefix
padded to the four fields `VIProductVersion` requires.

## Layout

```
bin\alphashape.exe             GUI (windeployqt's Qt DLLs and plugins beside it)
bin\alphashape-cli.exe         command line
bin\*.dll                      vcpkg runtime, VC++ runtime, libomp, Qt
bin\platforms\qwindows.dll     the Windows platform plugin
bin\platforms\qoffscreen.dll   headless runs (--offscreen-smoke, CI, services)
python\alphashape.cp3XX-win_amd64.pyd   Python module, with the same DLLs
                               beside it; the tag is the interpreter it was
                               built for and Python imports it as `alphashape`
python\*.dll
share\proj\                    proj.db and a size-capped grid set
LICENSES\                      GPL, the notices, this file
LICENSES\Qt\                   Qt's LGPL-3.0 (and GPL-2.0/3.0) texts
LICENSES\third-party\          one <port>.txt per shipped vcpkg port, plus libomp.txt
LICENSES\Fonts\                the bundled fonts' OFL texts
```

`LICENSES\third-party\` is filled at configure time from vcpkg's
`share/<port>/copyright`, one file per port, so a runtime DLL cannot be added
without its licence text; `libomp.txt` comes from the LLVM install beside
clang-cl. The glob is deliberately not narrowed to the ports that ship a DLL:
it also picks up the build-only ports the GPU toolchain pulls in
(`vulkan-headers`, `vulkan-memory-allocator`, `shaderc`, `spirv-tools`, …).
Shipping those texts costs a few kilobytes and is harmless — an extra notice is
never a compliance problem, a missing one is. Qt's LGPL-3.0 text is not
optional: `alphashape_windows_install.cmake` fails the configure if neither the
Qt install nor the vendored copy under `packaging/licenses/` supplies it, and
the shipped `LICENSES\THIRD-PARTY-NOTICES.txt` names the exact Qt version in
the payload.

The GUI is a Windows-subsystem binary, so no console window opens beside it. A
console that started it is adopted (`AttachConsole(ATTACH_PARENT_PROCESS)`), so
`--version` and `--offscreen-smoke` still print there; a redirection passed to
the process is used as given.

`python\` has its own copy of the DLLs because Windows resolves a `.pyd`'s
load-time dependencies from the module's own directory, not from `bin\` or the
install root. For the same reason there is no `path` trick that would make one
copy serve both.

## Install modes

Per-user by default: `%LOCALAPPDATA%\Programs\alphashape`, no administrator
rights, uninstallable without them. The install-mode page offers a per-machine
install into `%ProgramFiles%\alphashape`, which needs elevation; on an
unelevated run NSIS skips the page and installs per-user.

Both modes use `HKCU` or `HKLM` consistently (Start menu, Add/Remove Programs,
PATH), and the uninstaller reads the mode back from the uninstall key whose
`InstallLocation` is its own directory -- not merely the first key it finds, so
a per-user and a per-machine copy can be installed side by side and each
uninstaller removes its own tree.

Unattended:

```
alphashape-0.1.0-windows-x86_64.exe /S /CurrentUser /D=C:\Temp\alphashape
alphashape-0.1.0-windows-x86_64.exe /S /AllUsers /D=C:\Temp\alphashape-machine
```

`/D=` must be last and unquoted, which is an NSIS rule, and it wins over the
directory a previous install recorded, so re-running with a different `/D=`
moves the install rather than landing in the old one. `/AddToPath` ticks the
PATH option that the components page offers. A directory with non-ASCII
characters is supported: paths are UTF-8 end to end.

## Runtime lookups

* **PROJ.** `ALPHASHAPE_PROJ_DATA` wins, then `share\proj` beside the
  executable (`bin\alphashape.exe` -> `..\share\proj`) and beside the module's
  image (`python\alphashape*.pyd` -> `..\share\proj`). `alphashape-cli
  --proj-info` shows the database in use. vcpkg's `proj` ships `proj.db` and no
  grid files, so the installed grid set is normally empty: transforms that need
  a geoid fail offline, by design, until `ALPHASHAPE_PROJ_DATA` names a full
  `proj-data` directory.
* **Qt.** Plugins are found in `bin\` because Qt looks in the application
  directory first; no `qt.conf` is needed.
* **GPU.** Vulkan is off in this build (`ALPHASHAPE_GPU=OFF` on Windows until
  the shader compiler branch lands), so nothing GPU-related is shipped.

## Building the installer in CI

`windows-package` in `.github/workflows/build.yml` does it: the build job
configures with `-DALPHASHAPE_BUILD_INSTALLER=ON -DALPHASHAPE_BUILD_PYTHON=ON`
and runs `cpack -G NSIS`. `windows-package-smoke` then installs the single
`.exe` on a fresh runner and runs `smoke.ps1`.

`smoke.ps1` installs both modes side by side (per-user first, into a directory
with non-ASCII characters, with `/AddToPath`), removes the machine copy and
asserts every part of the per-user copy survived, then removes that too. It
checks the licence texts, that `--proj-info` reports the non-ASCII path intact,
that PATH gains and loses exactly one element in the right hive, and that the
GUI's `--offscreen-smoke` and `--version` still print through a redirection.

## Python version

`alphashape*.pyd` builds against one CPython and is loadable only by that
minor version. The packaging configure is pinned to the interpreter
`actions/setup-python` installs (`-DPython3_ROOT_DIR=$env:pythonLocation`,
`-DPython3_FIND_STRATEGY=LOCATION`, `-DPython3_FIND_REGISTRY=NEVER`) with
pybind11 taken from that interpreter's `pip`, so the shipped module is
`alphashape.cp312-win_amd64.pyd` for a python.org 3.12. vcpkg's `pybind11` and
`python3` ports are deliberately not installed: their CMake wrapper replaces
`Development.Module` with vcpkg's own interpreter, which is how the module
previously ended up tagged for a Python the user would never have.

The tag in the file name names the interpreter and Python imports it as
`alphashape`; the packaging job also writes `dist/PYTHON-ABI.txt` with the exact
version, fails if it disagrees with the interpreter it was pointed at, and the
smoke job reads the file and installs the same minor with
`actions/setup-python`.

A bundled standalone CPython (so the module works with no Python installed at
all) is a later improvement: it means shipping the interpreter, its standard
library and its licence texts, roughly 30 MB, and a `python\` layout that does
not depend on the system installation. Nothing in the current layout blocks it.
