$ErrorActionPreference = 'Stop'

$appRoot = 'C:\360toolkit\app'
$reportRoot = 'C:\360toolkit\reports'
$exePath = Join-Path $appRoot '360ToolkitGS.exe'
$timestamp = Get-Date -Format 'yyyyMMdd-HHmmss'
$sessionRoot = Join-Path $reportRoot $timestamp

New-Item -ItemType Directory -Path $sessionRoot -Force | Out-Null

$summary = [ordered]@{
    timestamp = (Get-Date).ToString('s')
    sandboxComputer = $env:COMPUTERNAME
    appExe = $exePath
    appExists = (Test-Path $exePath)
    hostGpuVisible = $false
    nvcudaPresent = $false
    appTorchCudaBundled = $false
    appCudaRuntimeBundled = $false
    launchStatus = 'not-started'
    launchExitCode = $null
    notes = @()
}

try {
    Get-CimInstance Win32_OperatingSystem |
        Select-Object Caption, Version, BuildNumber, OSArchitecture |
        Format-List | Out-File -FilePath (Join-Path $sessionRoot 'os.txt') -Encoding utf8

    $gpuInfo = Get-CimInstance Win32_VideoController |
        Select-Object Name, DriverVersion, AdapterCompatibility, Status
    $gpuInfo | Format-List | Out-File -FilePath (Join-Path $sessionRoot 'gpu.txt') -Encoding utf8
    if ($gpuInfo) {
        $summary.hostGpuVisible = $true
    }

    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if ($nvidiaSmi) {
        & $nvidiaSmi.Source | Out-File -FilePath (Join-Path $sessionRoot 'nvidia-smi.txt') -Encoding utf8
    }
    else {
        'nvidia-smi.exe not available inside sandbox.' | Out-File -FilePath (Join-Path $sessionRoot 'nvidia-smi.txt') -Encoding utf8
        $summary.notes += 'nvidia-smi.exe not available inside sandbox.'
    }

    $nvcudaPath = Join-Path $env:WINDIR 'System32\nvcuda.dll'
    $summary.nvcudaPresent = Test-Path $nvcudaPath
    @(
        "Path: $nvcudaPath"
        "Exists: $($summary.nvcudaPresent)"
    ) | Out-File -FilePath (Join-Path $sessionRoot 'cuda-driver.txt') -Encoding utf8

    $torchCudaPath = Join-Path $appRoot '_internal\torch\lib\torch_cuda.dll'
    $cudaRuntimePath = Join-Path $appRoot '_internal\torch\lib\cudart64_12.dll'
    $summary.appTorchCudaBundled = Test-Path $torchCudaPath
    $summary.appCudaRuntimeBundled = Test-Path $cudaRuntimePath
    @(
        "torch_cuda.dll: $($summary.appTorchCudaBundled) ($torchCudaPath)"
        "cudart64_12.dll: $($summary.appCudaRuntimeBundled) ($cudaRuntimePath)"
    ) | Out-File -FilePath (Join-Path $sessionRoot 'app-cuda-binaries.txt') -Encoding utf8

    if (-not $summary.appExists) {
        throw "Packaged executable not found: $exePath"
    }

    $process = Start-Process -FilePath $exePath -WorkingDirectory $appRoot -PassThru
    $summary.launchStatus = 'started'

    $exited = $process.WaitForExit(20000)
    if ($exited) {
        $summary.launchStatus = 'exited-early'
        $summary.launchExitCode = $process.ExitCode
        $summary.notes += "App exited within smoke-test window. Exit code: $($process.ExitCode)"
    }
    else {
        $summary.launchStatus = 'running-after-20s'
        $summary.notes += 'App stayed alive for 20 seconds in Windows Sandbox.'
        Stop-Process -Id $process.Id -Force
        $summary.notes += 'Smoke-test stopped the app after verification.'
    }
}
catch {
    $summary.launchStatus = 'failed'
    $summary.notes += $_.Exception.Message
}
finally {
    $summary | ConvertTo-Json -Depth 4 | Out-File -FilePath (Join-Path $sessionRoot 'summary.json') -Encoding utf8
    $summary.GetEnumerator() |
        ForEach-Object { "{0}: {1}" -f $_.Key, $_.Value } |
        Out-File -FilePath (Join-Path $sessionRoot 'summary.txt') -Encoding utf8
}