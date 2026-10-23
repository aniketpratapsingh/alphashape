# The SmartScreen warning this installer shows

alphashape's Windows installer is **not code-signed**. Signing needs a
certificate from a commercial certificate authority and a hardware token, which
the project does not have; nothing is paid for to produce a release.

## What you will see

On first download, Windows Defender SmartScreen shows

> **Windows protected your PC**
> Microsoft Defender SmartScreen prevented an unrecognised app from starting.

This is the expected response to any binary whose publisher certificate the
installer does not have. It is not a statement that the file contains malware
and it is not specific to alphashape.

To continue: choose **More info**, check that the app is **alphashape**, then
choose **Run anyway**. The checkbox "Don't run" is the default, so this takes two
deliberate clicks.

## Verifying the download instead of trusting it

Every release publishes `SHA256SUMS` beside the installers. Check the installer
against it before running it:

```powershell
# 1. Download alphashape-<version>-windows-x86_64.exe and SHA256SUMS from the
#    same GitHub release.
# 2. Compare hashes. Get-FileHash prints uppercase; SHA256SUMS is lowercase.
$actual = (Get-FileHash .\alphashape-*-windows-x86_64.exe -Algorithm SHA256).Hash.ToLower()
$expected = (Select-String -Path .\SHA256SUMS -Pattern 'windows-x86_64\.exe').Line.Split()[0]
if ($actual -eq $expected) { "OK: $actual" } else { "MISMATCH: $actual != $expected" }
```

On Linux or macOS:

```bash
sha256sum --check --ignore-missing SHA256SUMS      # Linux
shasum -a 256 -c SHA256SUMS 2>/dev/null            # macOS, ignoring the other files
```

The same check catches a corrupted download, which is the more common failure.

## What the installer does and does not do

It writes only under the directory you choose (`%LOCALAPPDATA%\Programs\alphashape`
by default, or `%ProgramFiles%\alphashape` if you choose the per-machine
install), one Start menu folder, and the Add/Remove Programs entry for that
mode. It copies the Visual C++ runtime beside the app instead of installing it
system-wide, and it edits `PATH` only if you tick the option on the components
page. The uninstaller removes exactly what it staged.

What it cannot do without a first run is prove any of that to SmartScreen. If
you would rather not click through the warning, build from source: README.md
("Building from source") documents the CMake build, and
`.github/workflows/build.yml` is the exact recipe CI uses.
