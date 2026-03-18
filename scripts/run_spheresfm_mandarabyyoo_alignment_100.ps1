$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false

$ProjectRoot = "C:\Users\Everton-PC\Documents\APLICATIVOS\360toolkit"
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ScriptPath = Join-Path $ProjectRoot "scripts\run_spheresfm_mandarabyyoo_alignment.py"
$LogDir = "C:\Users\Everton-PC\Documents\ARQUIVOS_TESTE\mandarabyyoo\spheresfm_alignment_runs\logs"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$stdoutLog = Join-Path $LogDir "subset100_gpu.stdout.log"
$stderrLog = Join-Path $LogDir "subset100_gpu.stderr.log"
Remove-Item $stdoutLog, $stderrLog -ErrorAction SilentlyContinue

$process = Start-Process -FilePath $PythonExe `
    -ArgumentList @($ScriptPath, "--subset-count", "100", "--gpu") `
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