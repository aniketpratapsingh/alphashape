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
LICENSES\                      GPL, Qt (LGPL-3), bundled fonts (OFL), this file
```

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
PATH), and the uninstaller reads the mode back from the registry to find the
right hive.

Unattended:

```
alphashape-0.1.0-windows-x86_64.exe /S /CurrentUser /D=C:\Temp\alphashape
alphashape-0.1.0-windows-x86_64.exe /S /AllUsers /D=C:\Temp\alphashape-machine
```

`/D=` must be last and unquoted, which is an NSIS rule. `/AddToPath` ticks the
PATH option that the components page offers.

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

## Python version

`alphashape*.pyd` builds against one CPython and is loadable only by that
minor version. CMake names it `alphashape.cp314-win_amd64.pyd`, so the tag in
the file name already names the interpreter and Python imports it as
`alphashape`; the packaging job also writes `dist/PYTHON-ABI.txt` with the
exact version, and the smoke job reads that and installs the same minor with
`actions/setup-python`.

A bundled standalone CPython (so the module works with no Python installed at
all) is a later improvement: it means shipping the interpreter, its standard
library and its licence texts, roughly 30 MB, and a `python\` layout that does
not depend on the system installation. Nothing in the current layout blocks it.
