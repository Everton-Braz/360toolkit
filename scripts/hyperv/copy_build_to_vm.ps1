param(
    [string]$VmName = '360ToolkitGS-GPU-Test',
    [string]$SourcePath = (Join-Path $PSScriptRoot '..\..\dist\360ToolkitGS'),
    [string]$GuestDestinationDir = 'C:\Temp\360ToolkitGS',
    [bool]$ZipBeforeCopy = $true,
    [switch]$KeepArchive,
    [switch]$NoClobber
)

$ErrorActionPreference = 'Stop'

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Resolve-AbsolutePath {
    param([string]$Path)
    return (Resolve-Path -LiteralPath $Path).Path
}

if (-not (Test-IsAdministrator)) {
    throw 'Run this script from an elevated PowerShell session. Copy-VMFile requires Hyper-V administrative access.'
}

Import-Module Hyper-V -ErrorAction Stop

if (-not (Get-VM -Name $VmName -ErrorAction SilentlyContinue)) {
    throw "VM not found: $VmName"
}

$guestService = Get-VMIntegrationService -VMName $VmName -Name 'Guest Service Interface' -ErrorAction SilentlyContinue
if (-not $guestService) {
    throw (
        "Guest Service Interface integration service is not available for VM '$VmName'. " +
        "Use Get-VMIntegrationService -VMName '$VmName' on the host to inspect integration services, " +
        "or use the SMB fallback helper at scripts\\hyperv\\share_build_for_vm.ps1."
    )
}
if (-not $guestService.Enabled) {
    Enable-VMIntegrationService -VMName $VmName -Name 'Guest Service Interface' | Out-Null
}

$resolvedSourcePath = Resolve-AbsolutePath -Path $SourcePath
if (-not (Test-Path -LiteralPath $resolvedSourcePath)) {
    throw "Source path not found: $resolvedSourcePath"
}

$sourceItem = Get-Item -LiteralPath $resolvedSourcePath
$copySourcePath = $resolvedSourcePath
$archivePath = $null

if ($ZipBeforeCopy -and $sourceItem.PSIsContainer) {
    $archiveDir = Join-Path $env:TEMP '360toolkit-vm-transfer'
    New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
    $archivePath = Join-Path $archiveDir ($sourceItem.Name + '.zip')

    if (Test-Path -LiteralPath $archivePath) {
        Remove-Item -LiteralPath $archivePath -Force
    }

    Write-Host "[1/3] Compressing $resolvedSourcePath -> $archivePath"
    Compress-Archive -Path (Join-Path $resolvedSourcePath '*') -DestinationPath $archivePath -CompressionLevel Optimal
    $copySourcePath = $archivePath
}

$copyItem = Get-Item -LiteralPath $copySourcePath
$guestDestinationPath = if ($copyItem.PSIsContainer) {
    Join-Path $GuestDestinationDir $copyItem.Name
} else {
    Join-Path $GuestDestinationDir $copyItem.Name
}

Write-Host "[2/3] Copying $copySourcePath -> ${VmName}:$guestDestinationPath"
Copy-VMFile -Name $VmName `
    -SourcePath $copySourcePath `
    -DestinationPath $guestDestinationPath `
    -FileSource Host `
    -CreateFullPath `
    -Force:(!$NoClobber)

Write-Host "[3/3] Transfer complete"
Write-Host ''
Write-Host 'Inside the guest VM:'

if ($archivePath) {
    $extractDir = Join-Path $GuestDestinationDir $sourceItem.Name
    Write-Host "Expand-Archive -LiteralPath '$guestDestinationPath' -DestinationPath '$extractDir' -Force"
    if (-not $KeepArchive) {
        Remove-Item -LiteralPath $archivePath -Force -ErrorAction SilentlyContinue
    }
}
else {
    Write-Host "Build copied to: $guestDestinationPath"
}
