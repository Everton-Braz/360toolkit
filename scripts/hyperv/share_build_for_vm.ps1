param(
    [string]$SourcePath = (Join-Path $PSScriptRoot '..\..\releases\360ToolkitGS-v1.3.0-windows-x64-full-bundled.zip'),
    [string]$ShareName = '360toolkit_vm_incoming',
    [string]$ShareRoot = "$env:PUBLIC\Documents\360toolkit-vm-share",
    [switch]$KeepExistingFiles
)

$ErrorActionPreference = 'Stop'

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Resolve-WellKnownSidAccount {
    param([string]$Sid)

    try {
        return ([System.Security.Principal.SecurityIdentifier]::new($Sid)).Translate([System.Security.Principal.NTAccount]).Value
    }
    catch {
        return $null
    }
}

if (-not (Test-IsAdministrator)) {
    throw 'Run this script from an elevated PowerShell session. Creating an SMB share requires Administrator rights.'
}

Import-Module SmbShare -ErrorAction Stop

$resolvedSource = (Resolve-Path -LiteralPath $SourcePath).Path
if (-not (Test-Path -LiteralPath $resolvedSource)) {
    throw "Source path not found: $resolvedSource"
}

New-Item -ItemType Directory -Path $ShareRoot -Force | Out-Null

if (-not $KeepExistingFiles) {
    Get-ChildItem -LiteralPath $ShareRoot -Force -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
}

$sourceItem = Get-Item -LiteralPath $resolvedSource
$targetPath = Join-Path $ShareRoot $sourceItem.Name

Write-Host "[1/3] Copying source into share folder"
if ($sourceItem.PSIsContainer) {
    Copy-Item -LiteralPath $resolvedSource -Destination $targetPath -Recurse -Force
}
else {
    Copy-Item -LiteralPath $resolvedSource -Destination $targetPath -Force
}

$existingShare = Get-SmbShare -Name $ShareName -ErrorAction SilentlyContinue
if ($existingShare) {
    if ($existingShare.Path -ne $ShareRoot) {
        throw "SMB share '$ShareName' already exists and points to $($existingShare.Path)."
    }
}
else {
    Write-Host "[2/3] Creating SMB share $ShareName"
    $everyoneAccount = Resolve-WellKnownSidAccount -Sid 'S-1-1-0'
    $currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

    if ($everyoneAccount) {
        New-SmbShare -Name $ShareName -Path $ShareRoot -ReadAccess $everyoneAccount -FullAccess $currentUser | Out-Null
    }
    else {
        Write-Warning 'Could not resolve the localized Everyone account. Creating the share for the current user only.'
        New-SmbShare -Name $ShareName -Path $ShareRoot -FullAccess $currentUser | Out-Null
    }
}

$hostName = $env:COMPUTERNAME
$uncRoot = "\\$hostName\$ShareName"

Write-Host "[3/3] Share ready"
Write-Host ''
Write-Host 'Inside the VM, open these paths:'
Write-Host $uncRoot
Write-Host (Join-Path $uncRoot $sourceItem.Name)
Write-Host ''
Write-Host 'If the source is a ZIP, inside the VM you can run:'
if (-not $sourceItem.PSIsContainer -and $sourceItem.Extension -eq '.zip') {
    Write-Host "Expand-Archive -LiteralPath '$(Join-Path $uncRoot $sourceItem.Name)' -DestinationPath 'C:\Temp\360ToolkitGS' -Force"
}