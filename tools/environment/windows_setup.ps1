<#
  windows_setup.ps1 - Windows counterpart of ubuntu_setup.sh.

  Builds the .venv-loop virtual environment expected by the repo's .cmd
  launchers, then records an L1 software baseline for THIS machine:
  environment inventory, unit tests, and the reference-simulation loop.

  Safety: this script never touches hardware. It opens no serial port,
  passes no --enable flag, and issues no motion command. Every step below
  is evidence level L1 (software only).

  Messages are ASCII on purpose: PowerShell 5.1 mangles non-ASCII in
  scripts saved without a BOM.

  Usage (from Explorer): double-click windows_setup.cmd
  Usage (from a shell):  powershell -ExecutionPolicy Bypass -File windows_setup.ps1
                         optional: -SkipInstall   reuse an existing .venv-loop
#>
param(
    [switch]$SkipInstall,
    [string]$OutDir
)

$ErrorActionPreference = 'Continue'

$Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$Venv = Join-Path $Root '.venv-loop'
$VenvPy = Join-Path $Venv 'Scripts\python.exe'

if (-not $OutDir) {
    $stamp = Get-Date -Format 'yyyy-MM-dd'
    $OutDir = Join-Path $Root "experiments\${stamp}_windows_env\raw"
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$report = [ordered]@{
    generated_at_local = (Get-Date -Format 'yyyy-MM-ddTHH:mm:sszzz')
    repo_root          = $Root
    hardware_motion    = $false
    evidence_level     = 'L1'
    steps              = [ordered]@{}
}

function Write-Step($name, $text) {
    Write-Host ''
    Write-Host ('=' * 68)
    Write-Host ("  {0}  {1}" -f $name, $text)
    Write-Host ('=' * 68)
}

# ---------------------------------------------------------------- 1. python
Write-Step '1/5' 'Locating a Python interpreter'

$interpreter = $null
$candidates = @(
    @{ exe = 'py';     args = @('-3.12') },
    @{ exe = 'py';     args = @('-3') },
    @{ exe = 'python';  args = @() }
)
foreach ($c in $candidates) {
    if (-not (Get-Command $c.exe -ErrorAction SilentlyContinue)) { continue }
    $probe = & $c.exe @($c.args + '-c' + 'import sys;print(sys.version.split()[0]);print(sys.executable)') 2>$null
    if ($LASTEXITCODE -eq 0 -and $probe) {
        $interpreter = @{ exe = $c.exe; args = $c.args; version = $probe[0]; path = $probe[1] }
        break
    }
}

if (-not $interpreter) {
    Write-Host 'FAIL: no Python interpreter found on PATH.'
    Write-Host 'Install Python 3.12 (python.org) and re-run this script.'
    $report.steps['python'] = @{ ok = $false; error = 'interpreter_not_found' }
    Save-Utf8 (Join-Path $OutDir 'summary.json') ($report | ConvertTo-Json -Depth 6)
    exit 1
}

Write-Host ("Interpreter : {0}" -f $interpreter.path)
Write-Host ("Version     : {0}" -f $interpreter.version)

$pinnedOk = $interpreter.version.StartsWith('3.12')
if (-not $pinnedOk) {
    Write-Host ''
    Write-Host ("WARNING: requirements/core.txt pins numpy==2.5.3, which publishes no")
    Write-Host ("         wheel for Python {0}. Install may fall back or fail." -f $interpreter.version)
    Write-Host ("         The repo targets Python 3.12.")
}
$report.steps['python'] = [ordered]@{
    ok = $true; version = $interpreter.version; path = $interpreter.path
    matches_pinned_target = $pinnedOk
}

# ------------------------------------------------------------------ 2. venv
Write-Step '2/5' 'Creating .venv-loop and installing pinned requirements'

if ($SkipInstall -and (Test-Path $VenvPy)) {
    Write-Host 'Reusing existing .venv-loop (-SkipInstall).'
    $report.steps['venv'] = @{ ok = $true; created = $false; path = $Venv }
} else {
    if (-not (Test-Path $VenvPy)) {
        & $interpreter.exe @($interpreter.args + '-m' + 'venv' + $Venv)
    }
    if (-not (Test-Path $VenvPy)) {
        Write-Host 'FAIL: could not create the virtual environment.'
        $report.steps['venv'] = @{ ok = $false; error = 'venv_creation_failed'; path = $Venv }
        Save-Utf8 (Join-Path $OutDir 'summary.json') ($report | ConvertTo-Json -Depth 6)
        exit 1
    }
    & $VenvPy -m pip install --upgrade pip --quiet
    $req = Join-Path $Root 'requirements\dev.txt'
    Write-Host ("Installing from {0} (this can take a few minutes)" -f $req)
    $pipOut = & $VenvPy -m pip install -r $req 2>&1
    $pipOk = ($LASTEXITCODE -eq 0)
    Save-Utf8 (Join-Path $OutDir 'pip_install.log') $pipOut
    $pipOut | Select-Object -Last 3 | ForEach-Object { Write-Host $_ }
    $report.steps['venv'] = [ordered]@{
        ok = $pipOk; created = $true; path = $Venv
        requirements = 'requirements/dev.txt'; log = 'pip_install.log'
    }
    if (-not $pipOk) {
        Write-Host 'FAIL: dependency install did not complete. See pip_install.log.'
        Save-Utf8 (Join-Path $OutDir 'summary.json') ($report | ConvertTo-Json -Depth 6)
        exit 1
    }
}

Save-Utf8 (Join-Path $OutDir 'pip_freeze.txt') (& $VenvPy -m pip freeze)

# ---------------------------------------------------------------- 3. doctor
Write-Step '3/5' 'Environment inventory (tools/environment/doctor.py)'

$doctorOut = & $VenvPy (Join-Path $Root 'tools\environment\doctor.py') 2>&1
Save-Utf8 (Join-Path $OutDir 'doctor.json') $doctorOut
$doctorOut | ForEach-Object { Write-Host $_ }
$report.steps['doctor'] = @{ ok = ($LASTEXITCODE -eq 0); artifact = 'doctor.json' }

# ----------------------------------------------------------------- 4. tests
Write-Step '4/5' 'Unit tests (no hardware; hardware-dependent cases skip)'

Push-Location $Root
$testsOut = & $VenvPy -m unittest discover -s tests -v 2>&1
$testsOk = ($LASTEXITCODE -eq 0)
Save-Utf8 (Join-Path $OutDir 'tests.log') $testsOut
$testsOut | Select-Object -Last 4 | ForEach-Object { Write-Host $_ }
Pop-Location
$report.steps['unit_tests'] = @{ ok = $testsOk; artifact = 'tests.log' }
Write-Host ("Unit tests: {0}" -f $(if ($testsOk) { 'PASS' } else { 'FAIL - see tests.log' }))

# ------------------------------------------------------- 5. reference loop
Write-Step '5/5' 'Reference simulation: collect-sim -> train -> run-sim'

Push-Location $Root
$teacher = Join-Path $OutDir 'teacher.npz'
$policy  = Join-Path $OutDir 'policy.npz'
$rollout = Join-Path $OutDir 'rollout.jsonl'

$collectOut = & $VenvPy -m dummy_loop collect-sim --episodes 40 --seed 7 --output $teacher 2>&1
$collectOk = ($LASTEXITCODE -eq 0)
Save-Utf8 (Join-Path $OutDir 'collect.summary.json') $collectOut

$trainOut = & $VenvPy -m dummy_loop train --dataset $teacher --output $policy 2>&1
$trainOk = ($LASTEXITCODE -eq 0)
Save-Utf8 (Join-Path $OutDir 'train.summary.json') $trainOut

$rolloutOut = & $VenvPy -m dummy_loop run-sim --policy $policy --steps 240 --log $rollout 2>&1
Save-Utf8 (Join-Path $OutDir 'rollout.summary.json') $rolloutOut
$rolloutOk = ($LASTEXITCODE -eq 0)
Pop-Location

$rolloutOut | ForEach-Object { Write-Host $_ }

$initial = $null; $final = $null
try {
    $summary = ($rolloutOut -join "`n") | ConvertFrom-Json
    $initial = $summary.initial_error_rad
    $final   = $summary.final_error_rad
} catch { }

$report.steps['reference_loop'] = [ordered]@{
    ok = ($collectOk -and $trainOk -and $rolloutOk)
    seed = 7; episodes = 40; steps = 240
    initial_error_rad = $initial
    final_error_rad   = $final
    expected_initial_error_rad = 0.3082207001484489
    expected_final_error_rad   = 0.0022264043008919554
    units = 'rad (six-axis joint error norm, not per-axis)'
    scope = 'software smoke test; not a real-robot or grasp validation'
    artifacts = @('collect.summary.json','train.summary.json','rollout.summary.json','rollout.jsonl')
}

# ---------------------------------------------------------------- summary
Save-Utf8 (Join-Path $OutDir 'summary.json') ($report | ConvertTo-Json -Depth 6)

Write-Step 'DONE' 'Summary'
Write-Host ("Python        : {0}" -f $interpreter.version)
Write-Host ("Virtualenv    : {0}" -f $Venv)
Write-Host ("Unit tests    : {0}" -f $(if ($testsOk) { 'PASS' } else { 'FAIL' }))
if ($initial -ne $null) {
    Write-Host ("Reference loop: {0} -> {1} rad" -f $initial, $final)
}
Write-Host ("Evidence      : {0}" -f $OutDir)
Write-Host ''
Write-Host 'No serial port was opened and no motion command was sent.'
