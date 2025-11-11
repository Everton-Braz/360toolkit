# 360FrameTools - Final Build Verification Script
# Tests all WinError 1114 fixes

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "360FrameTools - FINAL BUILD VERIFICATION" -ForegroundColor Cyan
Write-Host "Testing WinError 1114 Fixes" -ForegroundColor Cyan
Write-Host "============================================================`n" -ForegroundColor Cyan

$distPath = "dist\360ToolkitGS-FULL"
$exePath = "$distPath\360ToolkitGS-FULL.exe"
$internalPath = "$distPath\_internal"

# Check if build exists
if (-not (Test-Path $exePath)) {
    Write-Host "[ERROR] Build not found at: $exePath" -ForegroundColor Red
    Write-Host "Please wait for build to complete" -ForegroundColor Yellow
    exit 1
}

Write-Host "[OK] Executable found" -ForegroundColor Green

# Check build size
$size = (Get-ChildItem $distPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1GB
Write-Host "[OK] Build size: $([math]::Round($size, 2)) GB" -ForegroundColor Green

if ($size -lt 8) {
    Write-Host "[WARNING] Size is smaller than expected (should be ~9 GB with all DLLs)" -ForegroundColor Yellow
}

# CRITICAL: Verify MSVC Runtime DLLs (THE FIX for WinError 1114)
Write-Host "`n=== CRITICAL: MSVC Runtime DLLs (WinError 1114 Fix) ===" -ForegroundColor Yellow

$msvcDlls = @(
    'msvcp140.dll',
    'vcruntime140.dll',
    'vcruntime140_1.dll'
)

$msvcFound = 0
foreach ($dll in $msvcDlls) {
    $path = "$internalPath\torch\lib\$dll"
    if (Test-Path $path) {
        $fileSize = (Get-Item $path).Length / 1KB
        Write-Host "[OK] $dll - $([math]::Round($fileSize, 1)) KB" -ForegroundColor Green
        $msvcFound++
    } else {
        Write-Host "[MISSING] $dll - CRITICAL!" -ForegroundColor Red
    }
}

if ($msvcFound -eq 0) {
    Write-Host "`n[ERROR] NO MSVC Runtime DLLs found!" -ForegroundColor Red
    Write-Host "This will cause WinError 1114 - rebuild required" -ForegroundColor Red
} else {
    Write-Host "`n[OK] Found $msvcFound MSVC Runtime DLLs" -ForegroundColor Green
}

# Verify CUDA DLLs
Write-Host "`n=== CUDA DLLs (GPU Support) ===" -ForegroundColor Yellow

$cudaDlls = Get-ChildItem "$internalPath\torch\lib" -Filter "cu*.dll" -ErrorAction SilentlyContinue
if ($cudaDlls.Count -gt 0) {
    Write-Host "[OK] Found $($cudaDlls.Count) CUDA DLLs" -ForegroundColor Green
    $totalCudaSize = ($cudaDlls | Measure-Object -Property Length -Sum).Sum / 1GB
    Write-Host "    Total CUDA size: $([math]::Round($totalCudaSize, 2)) GB" -ForegroundColor Cyan
} else {
    Write-Host "[WARNING] No CUDA DLLs found - GPU masking will not work" -ForegroundColor Yellow
}

# Verify PyTorch DLLs
Write-Host "`n=== PyTorch Core DLLs ===" -ForegroundColor Yellow

$torchCoreDlls = @('c10.dll', 'torch_cpu.dll', 'torch_python.dll')
foreach ($dll in $torchCoreDlls) {
    $path = "$internalPath\torch\lib\$dll"
    if (Test-Path $path) {
        $fileSize = (Get-Item $path).Length / 1MB
        Write-Host "[OK] $dll - $([math]::Round($fileSize, 1)) MB" -ForegroundColor Green
    } else {
        Write-Host "[MISSING] $dll" -ForegroundColor Red
    }
}

# Verify SDK
Write-Host "`n=== Insta360 SDK ===" -ForegroundColor Yellow

$sdkPath = "$internalPath\sdk\MediaSDK-3.0.5-20250619-win64"
if (Test-Path $sdkPath) {
    Write-Host "[OK] SDK found at: $sdkPath" -ForegroundColor Green
} else {
    Write-Host "[WARNING] SDK not found - Stage 1 extraction may fail" -ForegroundColor Yellow
}

# Verify FFmpeg
Write-Host "`n=== FFmpeg ===" -ForegroundColor Yellow

$ffmpegPath = "$internalPath\ffmpeg\ffmpeg.exe"
if (Test-Path $ffmpegPath) {
    Write-Host "[OK] FFmpeg found" -ForegroundColor Green
} else {
    Write-Host "[WARNING] FFmpeg not found - Stage 2 may have issues" -ForegroundColor Yellow
}

# Test execution with diagnostic output
Write-Host "`n=== LAUNCHING APPLICATION (WinError 1114 Test) ===" -ForegroundColor Cyan
Write-Host "Watch for runtime hook messages:" -ForegroundColor Yellow
Write-Host "  - '[PyTorch Hook] Starting runtime hook execution...'" -ForegroundColor Gray
Write-Host "  - '[PyTorch Hook] Pre-loaded: msvcp140.dll'" -ForegroundColor Gray
Write-Host "  - '[PyTorch Hook] Added X paths to PATH'" -ForegroundColor Gray
Write-Host "`nIf app crashes with DLL error, check output above for missing DLLs`n" -ForegroundColor Yellow

# Launch and capture output
$process = Start-Process -FilePath $exePath -PassThru -RedirectStandardOutput "app_stdout.txt" -RedirectStandardError "app_stderr.txt" -WindowStyle Normal

Write-Host "Application launched (PID: $($process.Id))" -ForegroundColor Cyan
Write-Host "Waiting 5 seconds for initialization..." -ForegroundColor Yellow

Start-Sleep -Seconds 5

# Check if process is still running
if ($process.HasExited) {
    Write-Host "`n[ERROR] Application crashed!" -ForegroundColor Red
    Write-Host "Exit code: $($process.ExitCode)" -ForegroundColor Red
    
    # Check for log file
    if (Test-Path "$distPath\360frametools.log") {
        Write-Host "`n=== Application Log (Last 30 lines) ===" -ForegroundColor Yellow
        Get-Content "$distPath\360frametools.log" -Tail 30
    }
    
    # Check stderr
    if (Test-Path "app_stderr.txt") {
        Write-Host "`n=== Standard Error ===" -ForegroundColor Yellow
        Get-Content "app_stderr.txt"
    }
    
    # Search for specific error
    $log = Get-Content "$distPath\360frametools.log" -Raw
    if ($log -match "WinError 1114") {
        Write-Host "`n[CRITICAL] WinError 1114 still present!" -ForegroundColor Red
        Write-Host "Possible causes:" -ForegroundColor Yellow
        Write-Host "  1. MSVC Runtime DLL missing (check above)" -ForegroundColor Yellow
        Write-Host "  2. MSVC Runtime not pre-loaded correctly" -ForegroundColor Yellow
        Write-Host "  3. Missing other DLL dependencies" -ForegroundColor Yellow
        Write-Host "`nRecommended: Use Dependencies.exe to check c10.dll" -ForegroundColor Cyan
    }
    
} else {
    Write-Host "`n[SUCCESS] Application is running!" -ForegroundColor Green
    Write-Host "Process ID: $($process.Id)" -ForegroundColor Green
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "  1. Load a test video in the app" -ForegroundColor White
    Write-Host "  2. Run Stage 3: AI Masking (GPU test)" -ForegroundColor White
    Write-Host "  3. Check masking speed (<1s per image = GPU working)" -ForegroundColor White
    Write-Host "`nTo close app: Press Ctrl+C in the application window" -ForegroundColor Gray
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "VERIFICATION COMPLETE" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
