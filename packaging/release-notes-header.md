## alphashape @VERSION@

Everything below is generated from the tagged source tree. Every asset was built on
GitHub-hosted runners, and the published files are the ones a fresh runner re-downloaded
from this release and exercised before it was published.

### Downloads

| OS | File | Notes |
|----|------|-------|
| Linux (Ubuntu 24.04+) | `alphashape-@VERSION@-linux-amd64.deb` | GUI, CLI and the Python module. `sudo apt install ./alphashape-@VERSION@-linux-amd64.deb` |
| Linux (any x86-64, portable) | `alphashape-@VERSION@-linux-x86_64.AppImage` | GUI and CLI, no install. `chmod +x` then run it (`--appimage-extract-and-run` if FUSE is unavailable) |
| macOS (Apple silicon) | `alphashape-@VERSION@-macos-arm64.dmg` | Open the image, drag `alphashape.app` to Applications |
| Windows (x86-64) | `alphashape-@VERSION@-windows-x86_64.exe` | Installer for the desktop app, the CLI and the Python module |
| Command-line archives | `alphashape-@VERSION@-linux-x86_64.tar.gz`, `alphashape-@VERSION@-windows-x86_64.zip` | Unpack and run `alphashape-cli`; no GUI, no install |

### Verify the download

`SHA256SUMS` lists every asset in this release:

```sh
sha256sum -c SHA256SUMS      # Linux
shasum -a 256 -c SHA256SUMS  # macOS
Get-FileHash -Algorithm SHA256 <file>   # Windows
```

Build provenance is attested for the installers (`.deb`, `.AppImage`, `.dmg`, `.exe`):

```sh
gh attestation verify alphashape-@VERSION@-linux-amd64.deb \
  --repo aniketpratapsingh/alphashape
```

### Third-party notices and bill of materials

`NOTICE.txt` lists every third-party component in this release with its licence and
source. `sbom.spdx.json` is the same set as an SPDX 2.3 document.

### Unsigned builds

These builds are **not code-signed**. That is a limitation of the project's current
release setup, not a sign that a file is unsafe — verify it against `SHA256SUMS` and the
provenance attestation above.

* **macOS:** the app is ad-hoc signed and not notarized. On first launch, right-click
  `alphashape.app` and choose **Open**, or clear the quarantine flag:
  `xattr -dr com.apple.quarantine /Applications/alphashape.app`.
* **Windows:** SmartScreen shows "Windows protected your PC". Choose **More info** then
  **Run anyway**. The installer details and the reason are in
  [`packaging/windows/README-smartscreen.md`](https://github.com/aniketpratapsingh/alphashape/blob/@VERSION@/packaging/windows/README-smartscreen.md).

### What does not work yet

This is an early (0.1.x) release. Known gaps — approximate results, unbuilt features,
format limitations — are listed in
[`engine/LIMITATIONS.md`](https://github.com/aniketpratapsingh/alphashape/blob/@VERSION@/engine/LIMITATIONS.md).
Read it before using the output for anything that matters.
