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
#
# Both install modes are installed side by side, because the failure they can
# hide from each other is the one that matters most: an uninstaller that works
# out its mode from the registry reads HKCU first, so a per-machine copy's
# Uninstall.exe used to remove the per-user tree. That is only visible with both
# installed, so the machine copy is removed first and the per-user copy is
# re-exercised afterwards.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string] $Installer,
    [string] $PythonVersion = ""
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Native programs are read back as bytes. The engine writes UTF-8, so the
# decoder must be UTF-8 as well or every non-ASCII path this script checks would
# arrive as mojibake and the assertion would fail for the wrong reason.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

function Step([string] $message) { Write-Host "`n== $message" -ForegroundColor Cyan }
function Fail([string] $message) { throw "alphashape smoke: $message" }

function Assert([bool] $condition, [string] $message) {
    if (-not $condition) { Fail $message }
}

# For conditions whose value is a string rather than a comparison: an empty or
# missing value fails, anything else passes. Calling Assert() with a bare string
# is a type error under a [bool] parameter, which is a confusing way to report
# "the program printed nothing".
function Assert-NotEmpty([string] $value, [string] $message) {
    if ([string]::IsNullOrWhiteSpace($value)) { Fail $message }
}

function Assert-Present([string] $path, [string] $what) {
    Assert (Test-Path -LiteralPath $path) "$what is missing: $path"
}

function Read-RegistryPath([string] $key, [string] $name) {
    # Under Set-StrictMode a missing property is an error, so the name is checked
    # rather than indexed: an unset Path is a valid state (a fresh profile) and
    # must read as empty, not abort the run.
    #
    # The value is read through .NET with DoNotExpandEnvironmentNames. PATH is a
    # REG_EXPAND_SZ, and Get-ItemProperty expands "%SystemRoot%\..." while the
    # installer writes the unexpanded text back: comparing one with the other
    # would fail on a perfectly correct edit. Raw bytes on both sides is the only
    # comparison that answers "did the installer change anything else".
    $parts = $key -split ':', 2
    if ($parts.Count -ne 2) { Fail "not a registry path: $key" }
    $hive = switch ($parts[0].ToUpperInvariant()) {
        'HKLM' { [Microsoft.Win32.Registry]::LocalMachine }
        'HKCU' { [Microsoft.Win32.Registry]::CurrentUser }
        default { Fail "unsupported registry hive in '$key'" }
    }
    $subkey = $parts[1].TrimStart('\')
    $opened = $hive.OpenSubKey($subkey, $false)
    if ($null -eq $opened) { return $null }
    try {
        $value = $opened.GetValue($name, $null, `
            [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        if ($null -eq $value) { return $null }
        return [string] $value
    } finally {
        $opened.Dispose()
    }
}

function Path-Elements([string] $value) {
    if ($null -eq $value -or $value -eq '') { return @() }
    return @($value -split ';' | Where-Object { $_ -ne '' })
}

# The value after "prefix" on the first matching line of a program's output, as
# a single string. The input may be an array of lines or one multi-line string
# (PowerShell gives the latter when the command is invoked with a here-string),
# so both are split. Where-Object/-replace would give an array when a line is
# printed twice, and a method call on an array fails.
function Output-Value([string[]] $lines, [string] $prefix) {
    foreach ($line in ($lines -join "`n" -split "`r?`n")) {
        if ($line.StartsWith($prefix)) { return $line.Substring($prefix.Length).Trim() }
    }
    return ''
}

function Path-Contains([string] $value, [string] $entry) {
    if ($null -eq $value) { return $false }
    $wanted = $entry.TrimEnd('\')
    foreach ($element in (Path-Elements $value)) {
        if ($element.TrimEnd('\') -eq $wanted) { return $true }
    }
    return $false
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

function Invoke-Installer([string[]] $arguments) {
    Write-Host "  > $installerName $($arguments -join ' ')"
    $process = Start-Process -FilePath $installerPath -ArgumentList $arguments -Wait -PassThru
    # NSIS exit codes: 0 success; 1 user abort; 2 the installer's own Abort.
    if ($process.ExitCode -ne 0) {
        Fail ("installer exited $($process.ExitCode) for: " + ($arguments -join ' '))
    }
}

# The installed GUI is a Windows-subsystem binary, so it has no console of its
# own: Windows gives such a process no standard handles unless they are passed
# in, and a bare `& $gui` therefore captures nothing. Start-Process with an
# explicit redirection hands it real handles (which the app must use instead of
# adopting CONOUT$), and the exit code is read from the process object.
function Invoke-Captured([string] $exe, [string[]] $arguments, [hashtable] $environment = @{}) {
    # Redirect files that do not exist yet: Start-Process writes them itself,
    # and reusing a file left over from a previous call would append confusion
    # rather than capture this run.
    $tag = [guid]::NewGuid().ToString('N')
    $stdout = Join-Path $env:RUNNER_TEMP "alphashape-smoke-$tag.out"
    $stderr = Join-Path $env:RUNNER_TEMP "alphashape-smoke-$tag.err"
    $previous = @{}
    foreach ($name in $environment.Keys) {
        $previous[$name] = [Environment]::GetEnvironmentVariable($name)
        [Environment]::SetEnvironmentVariable($name, $environment[$name])
    }
    try {
        $process = Start-Process -FilePath $exe -ArgumentList $arguments -Wait -PassThru `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow
        $out = @(Get-Content -LiteralPath $stdout -Encoding utf8 -ErrorAction SilentlyContinue)
        $err = @(Get-Content -LiteralPath $stderr -Encoding utf8 -ErrorAction SilentlyContinue)
    } finally {
        foreach ($name in $environment.Keys) {
            [Environment]::SetEnvironmentVariable($name, $previous[$name])
        }
        Remove-Item -LiteralPath $stdout, $stderr -Force -ErrorAction SilentlyContinue
    }
    return [pscustomobject]@{
        ExitCode = $process.ExitCode
        Output   = $out
        Errors   = $err
    }
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
#
# The per-user directory deliberately contains characters outside the ANSI code
# page. Everything from the installer's registry writes to the engine's PROJ data
# search goes through those bytes, and an implementation that decoded them as
# ANSI would either fail to find proj.db or report a mangled path -- neither of
# which a plain "C:\Temp" install would ever show.
$perUserDir = Join-Path $env:RUNNER_TEMP 'alphashape-Jörg-ü'
$machineDir = Join-Path $env:RUNNER_TEMP 'alphashape-per-machine'
$uninstallKey = 'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\alphashape'
$appKey = 'HKCU:\Software\alphashape\alphashape'
$machineKey = 'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\alphashape'
$machineAppKey = 'HKLM:\Software\alphashape\alphashape'
$startMenu = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\alphashape'
$machineStartMenu = Join-Path $env:ProgramData 'Microsoft\Windows\Start Menu\Programs\alphashape'

$userPathKey = 'HKCU:\Environment'
$machinePathKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment'

# The sentinel proves the PATH edit is surgical: it must survive both the
# installer's edit and the uninstaller's, and every other element must keep its
# order.
$sentinelEntry = 'C:\alphashape-smoke-sentinel'
$originalUserPath = Read-RegistryPath $userPathKey 'Path'
if ($null -eq $originalUserPath) { $originalUserPath = '' }
$baselineUserPath = "$sentinelEntry;$originalUserPath"
if ($baselineUserPath.EndsWith(';')) { $baselineUserPath = $baselineUserPath.Substring(0, $baselineUserPath.Length - 1) }
Set-ItemProperty -LiteralPath $userPathKey -Name 'Path' -Value $baselineUserPath -Type ExpandString
$originalMachinePath = Read-RegistryPath $machinePathKey 'Path'

try {
    # ---- per-user, silent, with PATH, into a non-ASCII directory -------------------------------
    if (Test-Path -LiteralPath $perUserDir) { Remove-Item -LiteralPath $perUserDir -Recurse -Force }
    $uninstallExe = Join-Path $perUserDir 'Uninstall.exe'

    Step "per-user install (/S /CurrentUser /AddToPath)"
    # /D= must be the last argument and unquoted; that is an NSIS rule, not a choice.
    Invoke-Installer @('/S', '/CurrentUser', '/AddToPath', "/D=$perUserDir")
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
        'LICENSES\THIRD-PARTY-NOTICES.txt',
        'LICENSES\Qt\LGPL-3.0-only.txt'
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

    Step "licence texts"
    # Qt is LGPL-3: the text ships because the DLLs do (LICENSES\Qt\LGPL-3.0-only.txt
    # is checked with the layout above). Every vcpkg port whose DLL is in bin\
    # contributes its own share/<port>/copyright, and libomp comes from LLVM.
    foreach ($port in @('libjpeg-turbo', 'tiff', 'libpng', 'proj', 'ceres', 'libomp')) {
        Assert-Present (Join-Path $perUserDir "LICENSES\third-party\$port.txt") `
            "third-party licence text for $port"
    }
    $thirdParty = @(Get-ChildItem -LiteralPath (Join-Path $perUserDir 'LICENSES\third-party') -Filter '*.txt')
    Assert ($thirdParty.Count -ge 8) "only $($thirdParty.Count) third-party licence texts shipped"
    # Non-empty only: a length floor would be wrong. Several ports are public
    # domain or carry a single-sentence dedication (sqlite3's is one line), and
    # the point here is that the copy happened at all -- the per-port existence
    # checks above and the build-time glob are what guarantee the content.
    $empty = @($thirdParty | Where-Object { $_.Length -eq 0 })
    Assert ($empty.Count -eq 0) ("empty licence texts: " +
        (($empty | ForEach-Object { $_.Name }) -join ', '))
    Write-Host "  $($thirdParty.Count) third-party texts: $(($thirdParty | ForEach-Object { $_.BaseName }) -join ', ')"

    # The notices must name the exact Qt in the payload. The template used to
    # spell @QT_VERSION@, which CPack never defines, so the shipped file said
    # "Qt " with no version at all; a version here means the substitution ran.
    $notices = Get-Content -LiteralPath (Join-Path $perUserDir 'LICENSES\THIRD-PARTY-NOTICES.txt') -Raw
    Assert ($notices -notmatch '@[A-Za-z_][A-Za-z0-9_]*@') 'the notices still contain an unsubstituted @placeholder@'
    $noticeMatch = [regex]::Match($notices, '(?m)^Qt (6\.\d+\.\d+) ')
    Assert ($noticeMatch.Success) 'the notices do not name the Qt version'
    $qtFromNotices = $noticeMatch.Groups[1].Value
    $qtFromBinary = (Get-Item -LiteralPath (Join-Path $perUserDir 'bin\Qt6Core.dll')).VersionInfo.FileVersion
    Write-Host "  notices say Qt $qtFromNotices; Qt6Core.dll is $qtFromBinary"
    Assert ($qtFromBinary.StartsWith($qtFromNotices)) `
        "the notices say Qt $qtFromNotices but the shipped Qt6Core.dll is $qtFromBinary"

    Step "Add/Remove Programs entry"
    Assert (Test-Path $uninstallKey) "no uninstall registry key $uninstallKey"
    $displayName = Read-RegistryPath $uninstallKey 'DisplayName'
    $displayVersion = Read-RegistryPath $uninstallKey 'DisplayVersion'
    $installLocation = Read-RegistryPath $uninstallKey 'InstallLocation'
    $uninstallString = Read-RegistryPath $uninstallKey 'UninstallString'
    Write-Host "  DisplayName    : $displayName"
    Write-Host "  DisplayVersion : $displayVersion"
    Write-Host "  InstallLocation: $installLocation"
    Assert ($displayName -like 'alphashape*') "unexpected DisplayName '$displayName'"
    Assert ($installLocation -eq $perUserDir) `
        "InstallLocation is '$installLocation', expected '$perUserDir'"
    Assert ($uninstallString -like "*$uninstallExe*") "UninstallString does not name $uninstallExe"
    $recordedMode = Read-RegistryPath $uninstallKey 'InstallMode'
    Assert ($recordedMode -eq 'CurrentUser') `
        "the uninstall key records InstallMode '$recordedMode', expected CurrentUser"

    Step "start menu shortcut"
    Assert-Present (Join-Path $startMenu 'alphashape.lnk') "Start menu shortcut"

    Step "PATH (per-user)"
    $userPath = Read-RegistryPath $userPathKey 'Path'
    Write-Host "  $userPath"
    Assert (Path-Contains $userPath (Join-Path $perUserDir 'bin')) `
        "$perUserDir\bin was not added to the per-user PATH"
    Assert (Path-Contains $userPath $sentinelEntry) 'the PATH edit dropped an existing entry'

    Step "CLI"
    $cli = Join-Path $perUserDir 'bin\alphashape-cli.exe'
    $tasks = Invoke-Captured $cli @('--list-tasks')
    $tasks.Output | Select-Object -First 5 | ForEach-Object { Write-Host "  $_" }
    if ($tasks.ExitCode -ne 0) {
        $tasks.Errors | ForEach-Object { Write-Host "  $_" }
        Fail "alphashape-cli --list-tasks exited $($tasks.ExitCode)"
    }
    Assert ($tasks.Output -contains 'AddPhotos') "AddPhotos is not in --list-tasks"
    Write-Host "  $($tasks.Output.Count) tasks"

    Step "CLI resolves PROJ data from a non-ASCII install path"
    $projInfo = Invoke-Captured $cli @('--proj-info')
    $projInfo.Output | ForEach-Object { Write-Host "  $_" }
    if ($projInfo.ExitCode -ne 0) {
        $projInfo.Errors | ForEach-Object { Write-Host "  $_" }
        Fail ("alphashape-cli --proj-info exited $($projInfo.ExitCode); the install could not " +
              'find proj.db from its own directory')
    }
    $expectedDb = Join-Path $perUserDir 'share\proj\proj.db'
    $reportedDb = Output-Value $projInfo.Output 'proj database: '
    Assert-NotEmpty $reportedDb '--proj-info printed no database line'
    # The bytes are UTF-8, so the non-ASCII directory must come back intact. A
    # path decoded with the ANSI code page would not compare equal here.
    Assert ($reportedDb.Contains('Jörg-ü')) `
        "the reported PROJ database path lost its non-ASCII characters: '$reportedDb'"
    Assert ((Resolve-Path -LiteralPath $reportedDb).Path -eq (Resolve-Path -LiteralPath $expectedDb).Path) `
        "PROJ database is '$reportedDb', expected the installed copy at '$expectedDb'"

    Step "GUI (offscreen)"
    # The GUI is a Windows-subsystem binary now: no console window appears, and a
    # console that started it is adopted only when the handles were not
    # redirected. Start-Process -RedirectStandardOutput is exactly the redirected
    # case, so this also proves the shim does not steal a caller's redirection.
    $gui = Join-Path $perUserDir 'bin\alphashape.exe'
    $smoke = Invoke-Captured $gui @('--offscreen-smoke') @{ QT_QPA_PLATFORM = 'offscreen' }
    $smoke.Output | ForEach-Object { Write-Host "  $_" }
    $smoke.Errors | ForEach-Object { Write-Host "  $_" }
    if ($smoke.ExitCode -ne 0) { Fail "alphashape.exe --offscreen-smoke exited $($smoke.ExitCode)" }
    Assert (($smoke.Output -join "`n") -match 'gui ready') "the GUI did not report 'gui ready'"

    $version = Invoke-Captured $gui @('--version') @{ QT_QPA_PLATFORM = 'offscreen' }
    $version.Output | ForEach-Object { Write-Host "  $_" }
    if ($version.ExitCode -ne 0) { Fail "alphashape.exe --version exited $($version.ExitCode)" }
    Assert (($version.Output -join "`n") -match 'alphashape-gui \d') 'the GUI printed no version'

    Step "Python module"
    # The module ships under the ABI-tagged name CMake gives it
    # (alphashape.cp312-win_amd64.pyd), which Python still imports as `alphashape`.
    $pyd = Get-ChildItem (Join-Path $perUserDir 'python') -Filter 'alphashape*.pyd' |
        Select-Object -First 1 -ExpandProperty FullName
    if (-not $pyd) { Fail "no alphashape*.pyd under $perUserDir\python" }
    if (-not $PythonVersion) {
        # The .pyd's file name carries the ABI tag, but the patch level matters
        # too, so CI passes the version it installed; a standalone run uses
        # whatever python is on PATH and lets the import below be the test.
        $PythonVersion = & python -c 'import sys; print(str(sys.version_info[0]) + "." + str(sys.version_info[1]))'
        if ($LASTEXITCODE -ne 0) { Fail "python is not on PATH; pass -PythonVersion" }
    }
    Write-Host "  module      : $pyd"
    Write-Host "  interpreter : Python $PythonVersion"
    # The ABI tag in the file name must match the interpreter this script was
    # told to use: they come from the same variable in CI, and a mismatch is the
    # one thing the smoke job exists to catch.
    $tagged = [regex]::Match($pyd, 'alphashape\.cp(\d+)').Groups[1].Value
    $expectedTag = ($PythonVersion -split '\.')[0..1] -join ''
    Assert ($tagged -eq $expectedTag) `
        "the module is built for CPython $tagged but the smoke test expected $PythonVersion"

    $env:PYTHONPATH = Join-Path $perUserDir 'python'
    $env:PYTHONHOME = $null
    # A redirected stdout makes Python encode with the ANSI code page, which
    # turns this install's "Jörg-ü" into replacement characters on the way out
    # and would make the path assertion below fail on a correct import. The
    # engine's own output is UTF-8; Python's must be too.
    $env:PYTHONIOENCODING = 'utf-8'
    $env:PYTHONUTF8 = '1'
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
        Remove-Item Env:\PYTHONPATH -ErrorAction SilentlyContinue
        Remove-Item Env:\PYTHONIOENCODING -ErrorAction SilentlyContinue
        Remove-Item Env:\PYTHONUTF8 -ErrorAction SilentlyContinue
    }
    $pythonCheck | ForEach-Object { Write-Host "  $_" }
    if ($pythonExit -ne 0) { Fail "importing alphashape from $pyd failed" }
    Assert (($pythonCheck -join "`n") -match '\bpython\s+3\.') "the module did not report a Python version"
    # The import must have come from the install, not from a stray copy elsewhere.
    $importPath = Output-Value $pythonCheck 'imported from '
    Assert-NotEmpty $importPath 'the module did not report where it was imported from'
    Assert ($importPath -like "$perUserDir\python\*") "alphashape was imported from '$importPath', not the install"

    # ---- per-machine, silent, with PATH, side by side with the per-user copy -------------------
    # The runner is elevated, so /AllUsers is available. The install goes to a
    # temporary directory rather than %ProgramFiles% so nothing is left on the
    # image and the removal check stays exact; the mode (HKLM registration, the
    # all-users PATH) is what is under test.
    Step "per-machine install (/S /AllUsers /AddToPath)"
    if (Test-Path -LiteralPath $machineDir) { Remove-Item -LiteralPath $machineDir -Recurse -Force }

    Invoke-Installer @('/S', '/AllUsers', '/AddToPath', "/D=$machineDir")
    Assert-Present (Join-Path $machineDir 'bin\alphashape-cli.exe') 'the CLI (per-machine install)'
    Assert-Present (Join-Path $machineDir 'Uninstall.exe') 'the uninstaller (per-machine install)'
    if (-not (Test-Path $machineKey)) {
        $wow = 'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\alphashape'
        if (Test-Path $wow) {
            Fail ('no machine-wide uninstall key at ' + $machineKey + ', but one is at ' +
                  ${wow} + ': the installer wrote through the 32-bit registry view ' +
                  '(missing SetRegView 64)')
        }
        Fail "no machine-wide uninstall registry key $machineKey"
    }
    $machineEntryLocation = Read-RegistryPath $machineKey 'InstallLocation'
    Write-Host "  DisplayName    : $(Read-RegistryPath $machineKey 'DisplayName')"
    Write-Host "  InstallLocation: $machineEntryLocation"
    Assert ($machineEntryLocation -eq $machineDir) `
        "per-machine InstallLocation is '$machineEntryLocation', expected '$machineDir'"
    $machineRecordedMode = Read-RegistryPath $machineKey 'InstallMode'
    Assert ($machineRecordedMode -eq 'AllUsers') `
        "the machine uninstall key records InstallMode '$machineRecordedMode', expected AllUsers"

    Step "PATH (per-machine)"
    $machinePath = Read-RegistryPath $machinePathKey 'Path'
    Assert (Path-Contains $machinePath (Join-Path $machineDir 'bin')) `
        "$machineDir\bin was not added to the machine PATH"
    # The per-user entry must still be in HKCU: the two hives are edited
    # independently, and an installer that wrote both modes to one hive shows it here.
    Assert (Path-Contains (Read-RegistryPath $userPathKey 'Path') (Join-Path $perUserDir 'bin')) `
        'the per-machine install removed the per-user PATH entry from HKCU'

    Step "CLI (per-machine)"
    $machineTasks = Invoke-Captured (Join-Path $machineDir 'bin\alphashape-cli.exe') @('--list-tasks')
    if ($machineTasks.ExitCode -ne 0) {
        $machineTasks.Errors | ForEach-Object { Write-Host "  $_" }
        Fail "the per-machine CLI exited $($machineTasks.ExitCode)"
    }
    Assert ($machineTasks.Output -contains 'AddPhotos') "the per-machine CLI has no AddPhotos task"
    $machineProj = Invoke-Captured (Join-Path $machineDir 'bin\alphashape-cli.exe') @('--proj-info')
    if ($machineProj.ExitCode -ne 0) {
        $machineProj.Output | ForEach-Object { Write-Host "  $_" }
        Fail 'the per-machine CLI could not resolve its own proj.db'
    }
    $machineDb = Output-Value $machineProj.Output 'proj database: '
    Assert-NotEmpty $machineDb 'the per-machine --proj-info printed no database line'
    Assert ((Resolve-Path -LiteralPath $machineDb).Path -eq (Resolve-Path -LiteralPath (Join-Path $machineDir 'share\proj\proj.db')).Path) `
        "the per-machine PROJ database is '$machineDb', not its own copy"

    Step "uninstall (per-machine); the per-user copy must survive"
    $machineUninstaller = Start-Process -FilePath (Join-Path $machineDir 'Uninstall.exe') `
        -ArgumentList '/S' -Wait -PassThru
    if ($machineUninstaller.ExitCode -ne 0) {
        Fail "the per-machine uninstaller exited $($machineUninstaller.ExitCode)"
    }
    Wait-Gone { -not (Test-Path -LiteralPath $machineDir) } "the per-machine install directory"
    Wait-Gone { -not (Test-Path $machineKey) } "the machine-wide uninstall key $machineKey"
    Wait-Gone { -not (Test-Path $machineAppKey) } "the machine-wide application key $machineAppKey"
    Wait-Gone { -not (Test-Path $machineStartMenu) } "the all-users Start menu folder $machineStartMenu"

    # This is the multi-mode case the uninstaller used to get wrong: with both
    # copies installed it read HKCU first, decided it was the per-user copy, and
    # deleted the per-user tree while leaving its own behind. Every one of these
    # assertions failed then.
    Step "the per-user copy is intact after the machine uninstall"
    Assert-Present (Join-Path $perUserDir 'bin\alphashape-cli.exe') 'the per-user CLI after the machine uninstall'
    Assert-Present (Join-Path $perUserDir 'bin\alphashape.exe') 'the per-user GUI after the machine uninstall'
    Assert-Present (Join-Path $perUserDir 'share\proj\proj.db') 'the per-user PROJ data after the machine uninstall'
    Assert-Present (Join-Path $startMenu 'alphashape.lnk') 'the per-user Start menu shortcut after the machine uninstall'
    Assert (Test-Path $uninstallKey) 'the per-user uninstall key after the machine uninstall'
    $survivingLocation = Read-RegistryPath $uninstallKey 'InstallLocation'
    Assert ($survivingLocation -eq $perUserDir) `
        "the per-user InstallLocation became '$survivingLocation'"
    Assert (Path-Contains (Read-RegistryPath $userPathKey 'Path') (Join-Path $perUserDir 'bin')) `
        'the machine uninstall removed the per-user PATH entry'
    Assert (-not (Path-Contains (Read-RegistryPath $machinePathKey 'Path') (Join-Path $machineDir 'bin'))) `
        'the machine uninstall left its own PATH entry behind'

    # The surviving copy must still work, not merely still exist.
    $afterMachine = Invoke-Captured (Join-Path $perUserDir 'bin\alphashape-cli.exe') @('--proj-info')
    if ($afterMachine.ExitCode -ne 0) {
        $afterMachine.Output | ForEach-Object { Write-Host "  $_" }
        Fail 'the per-user CLI stopped working after the machine copy was uninstalled'
    }
    Write-Host "  the per-user copy still runs"

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

    Step "PATH restored"
    $finalUserPath = Read-RegistryPath $userPathKey 'Path'
    Write-Host "  $finalUserPath"
    Assert (-not (Path-Contains $finalUserPath (Join-Path $perUserDir 'bin'))) `
        'the per-user PATH entry survived the uninstall'
    # Element by element, in order: "removed without altering other entries".
    Assert (((Path-Elements $finalUserPath) -join ';') -eq ((Path-Elements $baselineUserPath) -join ';')) `
        "PATH is '$finalUserPath', expected '$baselineUserPath' intact"

    # The machine PATH was edited by the machine install only; it must be back to
    # exactly what it was, with no residue of the temporary directory.
    $finalMachinePath = Read-RegistryPath $machinePathKey 'Path'
    $machineBefore = @(Path-Elements $originalMachinePath)
    $machineAfter = @(Path-Elements $finalMachinePath)
    if (($machineAfter -join ';') -ne ($machineBefore -join ';')) {
        # Name the elements that differ rather than printing two long strings:
        # an installer that rewrites unrelated entries usually changes one
        # character on each of several of them.
        Write-Host "  before: $($machineBefore.Count) elements, after: $($machineAfter.Count)" -ForegroundColor Yellow
        for ($i = 0; $i -lt [Math]::Max($machineBefore.Count, $machineAfter.Count); $i++) {
            $b = if ($i -lt $machineBefore.Count) { $machineBefore[$i] } else { '<none>' }
            $a = if ($i -lt $machineAfter.Count) { $machineAfter[$i] } else { '<none>' }
            if ($a -ne $b) { Write-Host "    [$i] '$b' -> '$a'" -ForegroundColor Yellow }
        }
        Fail 'the machine PATH did not return to its original value'
    }

    Write-Host "`nWindows installer smoke test passed" -ForegroundColor Green
} finally {
    # Leave the runner as it was found, whatever the outcome: a failed run must
    # not poison a later one on a reused image.
    if ($originalUserPath -ne '') {
        Set-ItemProperty -LiteralPath $userPathKey -Name 'Path' -Value $originalUserPath -Type ExpandString
    } else {
        Remove-ItemProperty -LiteralPath $userPathKey -Name 'Path' -ErrorAction SilentlyContinue
    }
    if ($null -ne $originalMachinePath) {
        Set-ItemProperty -LiteralPath $machinePathKey -Name 'Path' -Value $originalMachinePath -Type ExpandString
    }
}
