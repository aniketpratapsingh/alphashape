# Install, run and uninstall check for the alphashape Windows installer.
#
# Runs on a fresh windows-latest runner that has none of the build dependencies:
# only the .exe this repository's packaging job produced, plus the Python
# interpreter version the module was built for (installed by the caller with
# actions/setup-python).
#
# Usage: smoke.ps1 -Installer <alphashape-*-windows-x86_64.exe> [-PythonVersion 3.12]
#
# Everything here asserts against the installed tree, not the build tree. Any
# failure throws, so the job fails with the message that explains what is wrong.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Installer,
    [string] $PythonVersion = ""
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Step([string] $message) { Write-Host "`n== $message" -ForegroundColor Cyan }
function Fail([string] $message) { throw "alphashape smoke: $message" }

function Assert([bool] $condition, [string] $message) {
    if (-not $condition) { Fail $message }
}

function Assert-Present([string] $path, [string] $what) {
    Assert (Test-Path -LiteralPath $path) "$what is missing: $path"
}

# A silent NSIS uninstaller copies itself to %TEMP%, relaunches from there and
# lets the original exit, so waiting on the process is not enough: poll until the
# thing under test is gone, or time out and say what is still there.
function Wait-Gone([scriptblock] $condition, [string] $what, [int] $timeoutSeconds = 120) {
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (& $condition) { return }
        Start-Sleep -Milliseconds 500
    }
    # Say what is actually still there: "the key still exists" alone does not
    # tell the next reader which files the install and uninstall rules disagree on.
    if (-not (& $condition)) {
        Write-Host "  still present after ${timeoutSeconds}s:" -ForegroundColor Yellow
        foreach ($probe in @($perUserDir, $machineDir)) {
            if ($probe -and (Test-Path -LiteralPath $probe)) {
                Get-ChildItem -LiteralPath $probe -Recurse -Force |
                    Select-Object -First 25 | ForEach-Object { Write-Host "    $($_.FullName)" }
            }
        }
    }
    Fail "$what still exists $timeoutSeconds s after the uninstaller finished"
}

$installerPath = (Resolve-Path -LiteralPath $Installer).Path
$installerName = Split-Path $installerPath -Leaf
Step "installer"
Write-Host "  $installerPath"
$installerSizeMb = [math]::Round((Get-Item -LiteralPath $installerPath).Length / 1MB, 1)
Write-Host "  $installerSizeMb MB"

# Size guard: the budget is 350 MB, checked at the source (the packaging job
# prints the same breakdown). A silent regression that starts shipping debug Qt
# would blow through it here rather than on a user's disk.
$maxInstallerMb = 350
Assert ($installerSizeMb -le $maxInstallerMb) `
    "installer is $installerSizeMb MB, over the ${maxInstallerMb} MB budget"

# ---- locations -------------------------------------------------------------------------------
# Declared before any function uses them: Set-StrictMode makes a reference to an
# unassigned variable an error, and Wait-Gone reports on both trees.
$perUserDir = Join-Path $env:RUNNER_TEMP 'alphashape-per-user'
$machineDir = Join-Path $env:RUNNER_TEMP 'alphashape-per-machine'
$uninstallKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\alphashape'
$appKey = 'HKCU:\Software\alphashape\alphashape'
$startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\alphashape'

# ---- per-user, silent, into a temporary directory -------------------------------------------
if (Test-Path -LiteralPath $perUserDir) { Remove-Item -LiteralPath $perUserDir -Recurse -Force }
$uninstallExe = Join-Path $perUserDir 'Uninstall.exe'

function Invoke-Installer([string[]] $arguments) {
    Write-Host "  > $installerName $($arguments -join ' ')"
    $process = Start-Process -FilePath $installerPath -ArgumentList $arguments -Wait -PassThru
    # NSIS exit codes: 0 success; 1 user abort; 2 the installer's own Abort.
    if ($process.ExitCode -ne 0) {
        Fail ("installer exited $($process.ExitCode) for: " + ($arguments -join ' '))
    }
}

Step "per-user install (/S /CurrentUser)"
# /D= must be the last argument and unquoted; that is an NSIS rule, not a choice.
Invoke-Installer @('/S', '/CurrentUser', "/D=$perUserDir")
Assert-Present $uninstallExe 'the uninstaller'
Assert-Present (Join-Path $perUserDir 'bin\alphashape.exe') 'the GUI'
Assert-Present (Join-Path $perUserDir 'bin\alphashape-cli.exe') 'the CLI'
Write-Host "  installed $perUserDir"

Step "install layout"
$expected = @(
    'bin\alphashape.exe',
    'bin\alphashape-cli.exe',
    'bin\platforms\qwindows.dll',
    'bin\platforms\qoffscreen.dll',
    'python\python-usage.txt',
    'share\proj\proj.db',
    'LICENSES\LICENSE',
    'LICENSES\THIRD-PARTY-NOTICES.txt'
)
foreach ($relative in $expected) {
    Assert-Present (Join-Path $perUserDir $relative) "installed file"
}
# The app-local runtime: without these the GUI would not start on a machine
# that has no Visual Studio installed, which is what a fresh runner is.
foreach ($dll in @('vcruntime140.dll', 'msvcp140.dll')) {
    Assert-Present (Join-Path $perUserDir "bin\$dll") "app-local VC++ runtime"
}
Assert-Present (Join-Path $perUserDir 'bin\Qt6Core.dll') 'Qt (windeployqt output)'
Assert-Present (Join-Path $perUserDir 'bin\Qt6Widgets.dll') 'Qt Widgets (windeployqt output)'
Assert-Present (Join-Path $perUserDir 'python\vcruntime140.dll') 'VC++ runtime beside the module'

Step "Add/Remove Programs entry"
Assert (Test-Path $uninstallKey) "no uninstall registry key $uninstallKey"
$entry = Get-ItemProperty -Path $uninstallKey
Write-Host "  DisplayName    : $($entry.DisplayName)"
Write-Host "  DisplayVersion : $($entry.DisplayVersion)"
Write-Host "  InstallLocation: $($entry.InstallLocation)"
Assert ($entry.DisplayName -like 'alphashape*') "unexpected DisplayName '$($entry.DisplayName)'"
Assert ($entry.InstallLocation -eq $perUserDir) `
    "InstallLocation is '$($entry.InstallLocation)', expected '$perUserDir'"
Assert ($entry.UninstallString -like "*$uninstallExe*") "UninstallString does not name $uninstallExe"

Step "start menu shortcut"
Assert-Present (Join-Path $startMenu 'alphashape.lnk') "Start menu shortcut"

Step "CLI"
$cli = Join-Path $perUserDir 'bin\alphashape-cli.exe'
$ErrorActionPreference = 'Continue'
$tasks = & $cli --list-tasks
$cliExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($cliExit -ne 0) { Fail "alphashape-cli --list-tasks exited $cliExit" }
$tasks | Select-Object -First 5 | ForEach-Object { Write-Host "  $_" }
Assert ($tasks -contains 'AddPhotos') "AddPhotos is not in --list-tasks"
Write-Host "  $($tasks.Count) tasks"

Step "CLI resolves PROJ data from the install"
$ErrorActionPreference = 'Continue'
$projInfo = & $cli --proj-info 2>&1 | ForEach-Object { "$_" }
$projExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($projExit -ne 0) {
    $projInfo | ForEach-Object { Write-Host "  $_" }
    Fail "alphashape-cli --proj-info exited $projExit; the install could not find proj.db"
}
$projInfo | ForEach-Object { Write-Host "  $_" }
$expectedDb = Join-Path $perUserDir 'share\proj\proj.db'
$reportedDb = ($projInfo | Where-Object { $_ -match '^proj database: ' }) -replace '^proj database: ', ''
Assert ((Resolve-Path -LiteralPath $reportedDb).Path -eq (Resolve-Path -LiteralPath $expectedDb).Path) `
    "PROJ database is '$reportedDb', expected the installed copy at '$expectedDb'"

Step "GUI (offscreen)"
$gui = Join-Path $perUserDir 'bin\alphashape.exe'
$previousPlatform = $env:QT_QPA_PLATFORM
$env:QT_QPA_PLATFORM = 'offscreen'
try {
    # `2>&1` on a native program turns stderr lines into ErrorRecords in
    # PowerShell 7, and $ErrorActionPreference = 'Stop' would then abort on a
    # warning the program printed while exiting 0. $ErrorActionPreference is
    # relaxed for the call instead, and the exit code decides.
    $ErrorActionPreference = 'Continue'
    $guiOutput = & $gui --offscreen-smoke 2>&1
    $guiExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = 'Stop'
    $env:QT_QPA_PLATFORM = $previousPlatform
}
$guiOutput | ForEach-Object { Write-Host "  $_" }
if ($guiExit -ne 0) { Fail "alphashape.exe --offscreen-smoke exited $guiExit" }
Assert (($guiOutput -join "`n") -match 'gui ready') "the GUI did not report 'gui ready'"

$previousQt = $env:QT_QPA_PLATFORM
$env:QT_QPA_PLATFORM = 'offscreen'
try {
    $version = & $gui --version 2>$null
    $versionExit = $LASTEXITCODE
} finally {
    $env:QT_QPA_PLATFORM = $previousQt
}
if ($versionExit -ne 0) { Fail "alphashape.exe --version exited $versionExit" }
Write-Host "  $version"

Step "Python module"
# The module ships under the ABI-tagged name CMake gives it
# (alphashape.cp314-win_amd64.pyd), which Python still imports as `alphashape`.
$pyd = Get-ChildItem (Join-Path $perUserDir 'python') -Filter 'alphashape*.pyd' |
    Select-Object -First 1 -ExpandProperty FullName
if (-not $pyd) { Fail "no alphashape*.pyd under $perUserDir\python" }
if (-not $PythonVersion) {
    # The .pyd's file name carries no ABI tag, so the only way to know which
    # interpreter it was built for is to be told. CI passes the version it
    # installed; a standalone run uses whatever python is on PATH and lets the
    # import below be the test.
    $PythonVersion = & python -c 'import sys; print(str(sys.version_info[0]) + "." + str(sys.version_info[1]))'
    if ($LASTEXITCODE -ne 0) { Fail "python is not on PATH; pass -PythonVersion" }
}
Write-Host "  module      : $pyd"
Write-Host "  interpreter : Python $PythonVersion"
$env:PYTHONPATH = Join-Path $perUserDir 'python'
$env:PYTHONHOME = $null
$ErrorActionPreference = 'Continue'
try {
    $pythonCheck = & python -c @"
import sys, alphashape
print("imported from", alphashape.__file__)
tasks = alphashape.list_tasks()
assert "AddPhotos" in tasks, "AddPhotos missing from the task list"
print("python", sys.version.split()[0], "sees", len(tasks), "tasks")
"@
    $pythonExit = $LASTEXITCODE
} finally {
    $ErrorActionPreference = 'Stop'
    Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
}
$pythonCheck | ForEach-Object { Write-Host "  $_" }
if ($pythonExit -ne 0) { Fail "importing alphashape from $pyd failed" }
Assert (($pythonCheck -join "`n") -match '\bpython\s+3\.') "the module did not report a Python version"
# The import must have come from the install, not from a stray copy elsewhere.
$importPath = ($pythonCheck | Where-Object { $_ -match '^imported from ' }) -replace '^imported from ', ''
Assert ($importPath -like "$perUserDir\python\*") "alphashape was imported from '$importPath', not the install"

Step "uninstall (per-user)"
$uninstaller = Start-Process -FilePath $uninstallExe -ArgumentList '/S' -Wait -PassThru
if ($uninstaller.ExitCode -ne 0) { Fail "uninstaller exited $($uninstaller.ExitCode)" }

# The uninstaller removes the directory only when it is empty; a leftover file
# means the install rule and the uninstall rule disagree.
Wait-Gone { -not (Test-Path -LiteralPath $perUserDir) } "the install directory $perUserDir"
Wait-Gone { -not (Test-Path $uninstallKey) } "the uninstall registry key $uninstallKey"
Wait-Gone { -not (Test-Path $appKey) } "the application registry key $appKey"
Wait-Gone { -not (Test-Path $startMenu) } "the Start menu folder $startMenu"
Write-Host "  removed $perUserDir, both registry keys and the Start menu folder"

# ---- per-machine, silent, into a temporary directory ------------------------------------------
# The runner is elevated, so /AllUsers is available. The install goes to the
# same temporary directory rather than %ProgramFiles% so nothing is left on the
# image and the removal check stays exact; the mode (HKLM registration, the
# per-machine default directory) is what is under test here.
Step "per-machine install (/S /AllUsers)"
# The installer is a 32-bit program and sets SetRegView 64, so its HKLM writes
# land in the native (64-bit) view. This check runs from 64-bit PowerShell, which
# reads that same view; without the SetRegView the key would be under
# WOW6432Node and this lookup would (correctly) fail.
$machineKey = 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\alphashape'
$machineAppKey = 'HKLM:\Software\alphashape\alphashape'
if (Test-Path -LiteralPath $machineDir) { Remove-Item -LiteralPath $machineDir -Recurse -Force }

Invoke-Installer @('/S', '/AllUsers', "/D=$machineDir")
Assert-Present (Join-Path $machineDir 'bin\alphashape-cli.exe') 'the CLI (per-machine install)'
if (-not (Test-Path $machineKey)) {
    $wow = 'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\alphashape'
    if (Test-Path $wow) {
        Fail ('no machine-wide uninstall key at ' + $machineKey + ', but one is at ' +
              ${wow} + ': the installer wrote through the 32-bit registry view ' +
              '(missing SetRegView 64)')
    }
    Fail "no machine-wide uninstall registry key $machineKey"
}
$machineEntry = Get-ItemProperty -Path $machineKey
Write-Host "  DisplayName    : $($machineEntry.DisplayName)"
Write-Host "  InstallLocation: $($machineEntry.InstallLocation)"
Assert ($machineEntry.InstallLocation -eq $machineDir) `
    "per-machine InstallLocation is '$($machineEntry.InstallLocation)', expected '$machineDir'"

$ErrorActionPreference = 'Continue'
$machineTasks = & (Join-Path $machineDir 'bin\alphashape-cli.exe') --list-tasks
$machineExit = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($machineExit -ne 0) { Fail "the per-machine CLI exited $machineExit" }
Assert ($machineTasks -contains 'AddPhotos') "the per-machine CLI has no AddPhotos task"

Step "uninstall (per-machine)"
$machineUninstaller = Start-Process -FilePath (Join-Path $machineDir 'Uninstall.exe') `
    -ArgumentList '/S' -Wait -PassThru
if ($machineUninstaller.ExitCode -ne 0) {
    Fail "the per-machine uninstaller exited $($machineUninstaller.ExitCode)"
}
Wait-Gone { -not (Test-Path -LiteralPath $machineDir) } "the per-machine install directory"
Wait-Gone { -not (Test-Path $machineKey) } "the machine-wide uninstall key $machineKey"
Wait-Gone { -not (Test-Path $machineAppKey) } "the machine-wide application key $machineAppKey"

Write-Host "`nWindows installer smoke test passed" -ForegroundColor Green
