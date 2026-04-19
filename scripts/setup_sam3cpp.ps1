param(
    [string]$RepoDir = "downloads/sam3cpp",
    [string]$ModelName = "sam3-q4_0.ggml",
    [switch]$SkipClone,
    [switch]$SkipBuild,
    [switch]$SkipModelDownload
)

$ErrorActionPreference = 'Stop'

$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$repoPath = Join-Path $workspaceRoot $RepoDir
$modelsDir = Join-Path $repoPath 'models'
$buildDir = Join-Path $repoPath 'build'
$modelPath = Join-Path $modelsDir $ModelName
$repoUrl = 'https://github.com/peters/sam3.cpp'
$repoBranch = 'cuda-backend'
$modelUrl = "https://huggingface.co/PABannier/sam3.cpp/resolve/main/$ModelName"
$vswhereDefault = 'C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe'

Write-Host "SAM3.cpp setup root: $repoPath"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required but was not found in PATH."
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "cmake is required but was not found in PATH."
}

function Get-VsWherePath {
    if (Test-Path $vswhereDefault) {
        return $vswhereDefault
    }

    $vswhereCommand = Get-Command vswhere.exe -ErrorAction SilentlyContinue
    if ($vswhereCommand) {
        return $vswhereCommand.Source
    }

    return $null
}

function Get-VcVars64Path {
    $vswherePath = Get-VsWherePath
    if (-not $vswherePath) {
        throw "vswhere.exe was not found. Install Visual Studio Build Tools 2022 with the C++ workload."
    }

    $installationPath = & $vswherePath -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
    if (-not $installationPath) {
        throw "Visual Studio Build Tools with the C++ workload were not found."
    }

    $vcvarsPath = Join-Path $installationPath 'VC\Auxiliary\Build\vcvars64.bat'
    if (-not (Test-Path $vcvarsPath)) {
        throw "vcvars64.bat was not found at: $vcvarsPath"
    }

    return $vcvarsPath
}

function Invoke-WithMsvc {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WorkingDirectory,

        [Parameter(Mandatory = $true)]
        [string]$CommandLine
    )

    $vcvarsPath = Get-VcVars64Path
    $cmdScript = "call `"$vcvarsPath`" && cd /d `"$WorkingDirectory`" && $CommandLine"
    & cmd.exe /c $cmdScript
    if ($LASTEXITCODE -ne 0) {
        throw "MSVC command failed with exit code $LASTEXITCODE: $CommandLine"
    }
}

function Test-Sam3RepoLooksValid {
    param([string]$Path)

    return (Test-Path (Join-Path $Path '.git')) -and (Test-Path (Join-Path $Path 'CMakeLists.txt'))
}

if (-not $SkipClone) {
    if (-not (Test-Path $repoPath)) {
        git clone --recursive --branch $repoBranch $repoUrl $repoPath
    } elseif (-not (Test-Sam3RepoLooksValid -Path $repoPath)) {
        Write-Host "Existing folder is not a full sam3.cpp checkout. Initializing repository in place..."
        if (-not (Test-Path (Join-Path $repoPath '.git'))) {
            git -C $repoPath init
        }
        git -C $repoPath remote remove origin 2>$null
        git -C $repoPath remote add origin $repoUrl
        git -C $repoPath fetch origin
        git -C $repoPath checkout -B $repoBranch origin/$repoBranch
        git -C $repoPath submodule update --init --recursive
    } else {
        Write-Host "SAM3.cpp repo already exists: $repoPath"
    }
}

if (-not $SkipModelDownload) {
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null
    if (-not (Test-Path $modelPath)) {
        Write-Host "Downloading default full SAM3 model: $ModelName"
        Invoke-WebRequest -Uri $modelUrl -OutFile $modelPath
    } else {
        Write-Host "Model already present: $modelPath"
    }
}

if (-not $SkipBuild) {
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    Invoke-WithMsvc -WorkingDirectory $repoPath -CommandLine 'cmake -S . -B build -G "Visual Studio 17 2022" -A x64'
    Invoke-WithMsvc -WorkingDirectory $repoPath -CommandLine 'cmake --build build --config Release --target segment_persons'

    $sam3ImageProject = Join-Path $buildDir 'examples\sam3_image.vcxproj'
    if (Test-Path $sam3ImageProject) {
        Invoke-WithMsvc -WorkingDirectory $repoPath -CommandLine 'cmake --build build --config Release --target sam3_image'
    } else {
        Write-Warning 'sam3_image target was not generated. SDL2 is probably missing, so GUI preview remains optional.'
    }
}

Write-Host ""
Write-Host "Expected app integration paths:"
Write-Host "  segment_persons.exe: $(Join-Path $repoPath 'build/examples/Release/segment_persons.exe')"
Write-Host "  sam3_image.exe:      $(Join-Path $repoPath 'build/examples/Release/sam3_image.exe')"
Write-Host "  model:               $modelPath"