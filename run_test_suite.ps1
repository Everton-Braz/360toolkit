$ErrorActionPreference = "Stop"

# Configuration
$baseTestDir = "C:\Users\User\Documents\APLICATIVOS\Arquivos_Teste\TESTE_360TOOLKIT"
$inputFile = Join-Path $baseTestDir "VID_20251110_154106_00_171.insv"
$projectDir = "C:\Users\User\Documents\APLICATIVOS\360ToolKit"
$pythonScript = Join-Path $projectDir "run_pipeline_test.py"

# Check input file
if (-not (Test-Path $inputFile)) {
    Write-Error "Input file not found at: $inputFile"
    exit 1
}

Write-Host "Starting Test Suite..."
Write-Host "Input File: $inputFile"

# Activate Conda Environment (if needed, assuming running from active env or setting it up)
# We assume the user runs this from the correct environment or we can try to activate it.
# But usually it's better to rely on the current shell if active.
# However, for robustness, let's print python version.
$pythonExe = "C:\Users\User\miniconda3\envs\360toolkit-cpu\python.exe"
if (-not (Test-Path $pythonExe)) {
    Write-Warning "Conda environment python not found at default location. Using system python."
    $pythonExe = "python"
}

& $pythonExe --version

# Loop for 3 tests
for ($i = 1; $i -le 3; $i++) {
    $testName = "TESTE_$i"
    $outputDir = Join-Path $baseTestDir $testName
    
    Write-Host "`n----------------------------------------"
    Write-Host "Running $testName"
    Write-Host "Output Directory: $outputDir"
    
    # Clean/Create Directory
    if (Test-Path $outputDir) {
        Write-Host "Cleaning existing directory..."
        Remove-Item -Path $outputDir -Recurse -Force
    }
    New-Item -Path $outputDir -ItemType Directory -Force | Out-Null
    
    # Run Python Test Script
    Write-Host "Executing pipeline..."
    & $pythonExe $pythonScript $outputDir $inputFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[$testName] Completed Successfully."
    } else {
        Write-Host "[$testName] Failed with exit code $LASTEXITCODE."
    }
    
    # Verify Output
    $stage1Count = (Get-ChildItem (Join-Path $outputDir "stage1_frames") -ErrorAction SilentlyContinue).Count
    $stage2Count = (Get-ChildItem (Join-Path $outputDir "stage2_perspectives") -ErrorAction SilentlyContinue).Count
    $stage3Count = (Get-ChildItem (Join-Path $outputDir "stage3_masks") -Filter "*_mask.png" -ErrorAction SilentlyContinue).Count
    
    Write-Host "Results for ${testName}:"
    Write-Host "  Stage 1 Frames: $stage1Count"
    Write-Host "  Stage 2 Images: $stage2Count"
    Write-Host "  Stage 3 Masks:  $stage3Count"
}

Write-Host "`nTest Suite Completed."
