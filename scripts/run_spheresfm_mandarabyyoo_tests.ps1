$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = "C:\Users\Everton-PC\Documents\APLICATIVOS\360toolkit"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$LogDir = "C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\mandarabyyoo\spheresfm_diagnostics\script_runs"

function Invoke-PythonScript {
    param(
        [string]$ScriptPath,
        [string]$BaseLogName
    )

    $stdoutLog = Join-Path $LogDir ($BaseLogName + ".stdout.log")
    $stderrLog = Join-Path $LogDir ($BaseLogName + ".stderr.log")
    Remove-Item $stdoutLog, $stderrLog -ErrorAction SilentlyContinue

    $process = Start-Process -FilePath $PythonExe `
        -ArgumentList $ScriptPath `
        -WorkingDirectory $ProjectRoot `
        -RedirectStandardOutput $stdoutLog `
        -RedirectStandardError $stderrLog `
        -NoNewWindow `
        -PassThru `
        -Wait

    if (Test-Path $stdoutLog) {
        Get-Content $stdoutLog | ForEach-Object { Write-Host $_ }
    }
    if (Test-Path $stderrLog) {
        Get-Content $stderrLog | ForEach-Object { Write-Host $_ }
    }

    return $process.ExitCode
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found at $PythonExe"
}

Push-Location $ProjectRoot
try {
    Write-Output "PYTHON_EXE=$PythonExe"

    $maskExit = Invoke-PythonScript -ScriptPath "scripts/check_mandarabyyoo_masks.py" -BaseLogName "01_mask_check"
    Write-Output "MASK_CHECK_EXIT=$maskExit"

    $diagExit = Invoke-PythonScript -ScriptPath "test_spheresfm_mandarabyyoo.py" -BaseLogName "02_spheresfm_diagnostic"
    Write-Output "SPHERESFM_DIAGNOSTIC_EXIT=$diagExit"

    if ($maskExit -ne 0 -or $diagExit -ne 0) {
        exit 1
    }

    exit 0
}
finally {
    Pop-Location
}