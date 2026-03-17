$ErrorActionPreference = 'Stop'

param(
    [Parameter(Mandatory = $true)]
    [string]$SourcePath,

    [string]$VmName = '360ToolkitGS-GPU-Test',
    [string]$VmRoot = "$env:PUBLIC\Documents\Hyper-V\360toolkit",
    [string]$SwitchName,
    [int]$StartupMemoryGB = 8,
    [int]$MinimumMemoryGB = 4,
    [int]$MaximumMemoryGB = 16,
    [int]$ProcessorCount = 8,
    [switch]$TryGpuPartition,
    [switch]$StartVm
)

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Resolve-SourceDisk {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "SourcePath not found: $Path"
    }

    $item = Get-Item -LiteralPath $Path
    if ($item.PSIsContainer) {
        $disk = Get-ChildItem -LiteralPath $item.FullName -Recurse -File |
            Where-Object { $_.Extension -in '.vhdx', '.vhd' } |
            Sort-Object FullName |
            Select-Object -First 1

        if (-not $disk) {
            throw "No .vhdx or .vhd file was found under: $Path"
        }

        return $disk.FullName
    }

    if ($item.Extension -notin '.vhdx', '.vhd') {
        throw "SourcePath must be a .vhdx/.vhd file or a directory containing one. Got: $($item.FullName)"
    }

    return $item.FullName
}

function Resolve-VmSwitch {
    param(
        [string]$Name
    )

    if ($Name) {
        $vmSwitch = Get-VMSwitch -Name $Name -ErrorAction SilentlyContinue
        if (-not $vmSwitch) {
            throw "Hyper-V switch not found: $Name"
        }
        return $vmSwitch
    }

    $defaultSwitch = Get-VMSwitch -Name 'Default Switch' -ErrorAction SilentlyContinue
    if ($defaultSwitch) {
        return $defaultSwitch
    }

    $firstSwitch = Get-VMSwitch -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($firstSwitch) {
        return $firstSwitch
    }

    throw 'No Hyper-V virtual switch was found. Create a switch first or pass -SwitchName.'
}

function Get-PartitionableGpuInfo {
    if (-not (Get-Command Get-VMHostPartitionableGpu -ErrorAction SilentlyContinue)) {
        return @()
    }

    try {
        return @(Get-VMHostPartitionableGpu -ErrorAction Stop)
    }
    catch {
        Write-Warning "GPU partition inventory failed: $($_.Exception.Message)"
        return @()
    }
}

if (-not (Test-IsAdministrator)) {
    throw 'Run this script from an elevated PowerShell session. Hyper-V VM creation and GPU-P assignment require Administrator rights.'
}

Import-Module Hyper-V -ErrorAction Stop

$sourceDisk = Resolve-SourceDisk -Path $SourcePath
$vmSwitch = Resolve-VmSwitch -Name $SwitchName
$vmFolder = Join-Path $VmRoot $VmName
$vhdFolder = Join-Path $vmFolder 'Virtual Hard Disks'
$configFolder = Join-Path $vmFolder 'Virtual Machines'
$targetDisk = Join-Path $vhdFolder ([IO.Path]::GetFileName($sourceDisk))

if (Get-VM -Name $VmName -ErrorAction SilentlyContinue) {
    throw "A VM named '$VmName' already exists. Remove or rename it before running this script again."
}

New-Item -ItemType Directory -Path $vhdFolder -Force | Out-Null
New-Item -ItemType Directory -Path $configFolder -Force | Out-Null

Write-Host "[1/5] Copying source disk to $targetDisk"
Copy-Item -LiteralPath $sourceDisk -Destination $targetDisk -Force

$startupMemory = [int64]$StartupMemoryGB * 1GB
$minimumMemory = [int64]$MinimumMemoryGB * 1GB
$maximumMemory = [int64]$MaximumMemoryGB * 1GB

Write-Host "[2/5] Creating VM $VmName"
$vm = New-VM -Name $VmName `
    -Generation 2 `
    -MemoryStartupBytes $startupMemory `
    -VHDPath $targetDisk `
    -Path $vmFolder `
    -SwitchName $vmSwitch.Name

Write-Host "[3/5] Configuring memory, CPU, and integration services"
Set-VM -Name $VmName `
    -DynamicMemory `
    -MemoryMinimumBytes $minimumMemory `
    -MemoryMaximumBytes $maximumMemory `
    -AutomaticCheckpointsEnabled $false `
    -CheckpointType Disabled | Out-Null

Set-VMProcessor -VMName $VmName -Count $ProcessorCount | Out-Null
Enable-VMIntegrationService -VMName $VmName -Name 'Guest Service Interface' -ErrorAction SilentlyContinue | Out-Null
Set-VMFirmware -VMName $VmName -EnableSecureBoot On | Out-Null

$gpuAssigned = $false
$gpuInventory = Get-PartitionableGpuInfo

if ($TryGpuPartition) {
    Write-Host "[4/5] Attempting GPU partition assignment"

    if (-not (Get-Command Add-VMGpuPartitionAdapter -ErrorAction SilentlyContinue)) {
        Write-Warning 'Add-VMGpuPartitionAdapter is not available on this host.'
    }
    else {
        try {
            Add-VMGpuPartitionAdapter -VMName $VmName -ErrorAction Stop | Out-Null
            $gpuAssigned = $true
        }
        catch {
            Write-Warning "GPU partition assignment failed: $($_.Exception.Message)"
        }
    }
}

Write-Host "[5/5] Final VM summary"
$summary = [pscustomobject]@{
    VmName = $VmName
    VmPath = $vmFolder
    VhdPath = $targetDisk
    Switch = $vmSwitch.Name
    StartupMemoryGB = $StartupMemoryGB
    ProcessorCount = $ProcessorCount
    SourceDisk = $sourceDisk
    PartitionableGpuCount = @($gpuInventory).Count
    GpuPartitionAssigned = $gpuAssigned
}

$summary | Format-List

if ($StartVm) {
    Write-Host "Starting VM $VmName"
    Start-VM -Name $VmName | Out-Null
}

Write-Host ''
Write-Host 'Next steps:'
Write-Host '1. Boot the VM and complete Windows first-run if needed.'
Write-Host '2. Inside the guest, verify whether Device Manager shows a GPU-backed display adapter.'
Write-Host '3. Install guest GPU drivers if required by your NVIDIA GPU-P path.'
Write-Host '4. Copy dist\360ToolkitGS into the guest and run the packaged EXE there.'