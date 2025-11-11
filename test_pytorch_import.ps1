# Quick PyTorch/Ultralytics Import Test
Write-Host "=== Testing PyTorch and Ultralytics imports ===" -ForegroundColor Cyan

# Wait for build to complete
Write-Host "Waiting for build..." -ForegroundColor Yellow
while (!(Test-Path "dist\360ToolkitGS-FULL\360ToolkitGS-FULL.exe")) {
    Start-Sleep -Seconds 2
}

# Check if it's a fresh build
$lastWrite = (Get-Item "dist\360ToolkitGS-FULL\360ToolkitGS-FULL.exe").LastWriteTime
$age = (Get-Date) - $lastWrite

if ($age.TotalMinutes -gt 5) {
    Write-Host "[WARNING] EXE is $([math]::Round($age.TotalMinutes, 1)) minutes old - may not be latest build" -ForegroundColor Yellow
} else {
    Write-Host "[OK] Fresh build ($([math]::Round($age.TotalMinutes, 1)) minutes old)" -ForegroundColor Green
}

# Clean up old log
Remove-Item "dist\360ToolkitGS-FULL\360frametools.log" -ErrorAction SilentlyContinue

# Launch app
Write-Host "`nLaunching application..." -ForegroundColor Cyan
Start-Process "dist\360ToolkitGS-FULL\360ToolkitGS-FULL.exe" -WorkingDirectory "dist\360ToolkitGS-FULL"

# Wait for initialization
Write-Host "Waiting for initialization (10 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

# Check log
if (Test-Path "dist\360ToolkitGS-FULL\360frametools.log") {
    Write-Host "`n=== APPLICATION LOG ===" -ForegroundColor Cyan
    $log = Get-Content "dist\360ToolkitGS-FULL\360frametools.log"
    
    # Check for runtime hook messages
    Write-Host "`n1. Runtime Hook Execution:" -ForegroundColor Yellow
    $log | Select-String "PyTorch Hook|SDK Hook" | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }
    
    # Check for sys.path additions
    Write-Host "`n2. Sys.path modifications:" -ForegroundColor Yellow
    $log | Select-String "Added to sys.path" | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }
    
    # Check for module detection
    Write-Host "`n3. Module Detection:" -ForegroundColor Yellow
    $log | Select-String "Torch module location|__init__.py" | ForEach-Object { Write-Host "   $_" -ForegroundColor Gray }
    
    # Check for import results
    Write-Host "`n4. Import Results:" -ForegroundColor Yellow
    $pytorchSuccess = $log | Select-String "PyTorch loaded successfully"
    $ultralyticsSuccess = $log | Select-String "Ultralytics loaded successfully"
    $pytorchFail = $log | Select-String "PyTorch not available"
    $ultralyticsFail = $log | Select-String "Ultralytics not available"
    
    if ($pytorchSuccess) {
        Write-Host "   [SUCCESS] PyTorch imported!" -ForegroundColor Green
        $pytorchSuccess | ForEach-Object { Write-Host "      $_" -ForegroundColor Gray }
    } elseif ($pytorchFail) {
        Write-Host "   [FAILED] PyTorch import failed" -ForegroundColor Red
        $pytorchFail | ForEach-Object { Write-Host "      $_" -ForegroundColor Red }
    }
    
    if ($ultralyticsSuccess) {
        Write-Host "   [SUCCESS] Ultralytics imported!" -ForegroundColor Green
        $ultralyticsSuccess | ForEach-Object { Write-Host "      $_" -ForegroundColor Gray }
    } elseif ($ultralyticsFail) {
        Write-Host "   [FAILED] Ultralytics import failed" -ForegroundColor Red
        $ultralyticsFail | ForEach-Object { Write-Host "      $_" -ForegroundColor Red }
    }
    
    # Summary
    Write-Host "`n=== SUMMARY ===" -ForegroundColor Cyan
    if ($pytorchSuccess -and $ultralyticsSuccess) {
        Write-Host "[SUCCESS] Both PyTorch and Ultralytics imported successfully!" -ForegroundColor Green
        Write-Host "GPU masking should now be available." -ForegroundColor Green
    } else {
        Write-Host "[ISSUE] One or more imports failed. Check errors above." -ForegroundColor Red
    }
    
} else {
    Write-Host "[ERROR] Log file not created - app may have crashed" -ForegroundColor Red
}

Write-Host "`n=== Test Complete ===" -ForegroundColor Cyan
