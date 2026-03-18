$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = "C:\Users\Everton-PC\Documents\APLICATIVOS\360toolkit"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ScriptPath = Join-Path $ProjectRoot "scripts\tune_spheresfm_mandarabyyoo_mapper.py"
$LogDir = "C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\mandarabyyoo\spheresfm_alignment_runs\mapper_tuning\logs"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$stdoutLog = Join-Path $LogDir "mapper_tuning_100.stdout.log"
$stderrLog = Join-Path $LogDir "mapper_tuning_100.stderr.log"
Remove-Item $stdoutLog, $stderrLog -ErrorAction SilentlyContinue

$process = Start-Process -FilePath $PythonExe `
    -ArgumentList @($ScriptPath) `
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

exit $process.ExitCode