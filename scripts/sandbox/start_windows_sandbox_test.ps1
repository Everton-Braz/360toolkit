$ErrorActionPreference = 'Stop'

$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$wsbPath = Join-Path $repoRoot '360toolkit-sandbox.wsb'
$sandboxExe = Join-Path $env:WINDIR 'System32\WindowsSandbox.exe'

if (-not (Test-Path $sandboxExe)) {
    throw 'Windows Sandbox is not available on this machine. Enable the Windows Sandbox feature first.'
}

if (-not (Test-Path $wsbPath)) {
    throw "Sandbox configuration not found: $wsbPath"
}

Start-Process -FilePath $sandboxExe -ArgumentList ('"{0}"' -f $wsbPath)