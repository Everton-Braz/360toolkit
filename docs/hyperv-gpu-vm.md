# Persistent GPU VM For 360ToolkitGS

This is the path that is closer to a real customer machine than Windows Sandbox.

## Why this path

- Windows Sandbox is disposable and shares more host behavior than a normal PC.
- A persistent Hyper-V VM gives you a separate disk, separate user profile, separate installed software state, and repeatable snapshots.
- If GPU partitioning works on the host, you can also test CUDA-adjacent behavior more realistically.

## Current host constraints

- This repository already has a working Windows Sandbox GPU smoke test.
- The current session is not elevated, so Hyper-V VM creation cannot be completed from this session.
- No local Windows guest media or VHDX was found.

## Recommended media source

Use one of these official Microsoft options:

1. A Microsoft Windows developer VM or VHDX package for Hyper-V.
2. A Windows evaluation ISO or VHDX that you are allowed to use for testing.

The most practical route is a prebuilt Windows VHDX or developer VM package because it avoids a full OS install workflow.

## Provision the VM

Run this from an elevated PowerShell session after you download a Windows VHDX or extract a package containing one:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\hyperv\new_gpu_test_vm.ps1 \
  -SourcePath "D:\Downloads\WindowsDevVM" \
  -VmName "360ToolkitGS-GPU-Test" \
  -TryGpuPartition \
  -StartVm
```

You can pass either:

- A `.vhdx` or `.vhd` file directly.
- A directory that contains a `.vhdx` or `.vhd` file.

## What the script does

- Verifies that PowerShell is running as Administrator.
- Finds a VHDX/VHD source disk.
- Copies the source disk into a dedicated Hyper-V VM folder.
- Creates a Generation 2 Hyper-V VM.
- Configures dynamic memory, CPU count, and guest integration service.
- Tries to add a GPU partition adapter when the host supports GPU-P.
- Optionally starts the VM.

## Recommended VM test shape

For a realistic customer-style validation:

1. Use a clean local Windows account inside the VM.
2. Do not install your dev toolchain inside the guest.
3. Copy only the packaged app output into the guest.
4. Keep networking enabled so the app sees a normal online/offline environment.
5. Snapshot the VM before first test so you can return to a clean state.

## Faster copy to the VM

Copying the whole extracted build folder through the VM UI is slow because it contains many small files. The faster path is:

1. Compress the build into one ZIP on the host.
2. Use Hyper-V Guest Service Interface to push the ZIP into the VM.
3. Extract it inside the guest.

Use this helper from an elevated PowerShell session on the host:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\hyperv\copy_build_to_vm.ps1 \
  -VmName "360ToolkitGS-GPU-Test"
```

By default it:

- takes `dist\360ToolkitGS`
- compresses it to a temporary ZIP
- copies the ZIP to `C:\Temp\360ToolkitGS` inside the guest
- prints the `Expand-Archive` command to run inside the VM

If you want a different source or guest destination:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\hyperv\copy_build_to_vm.ps1 \
  -VmName "360ToolkitGS-GPU-Test" \
  -SourcePath ".\releases\360ToolkitGS-v1.3.0-windows-x64-full-bundled.zip" \
  -GuestDestinationDir "C:\Temp\Incoming"
```

## If Guest Service Interface is unavailable

Some imported VMs do not expose the Hyper-V Guest Service Interface needed by `Copy-VMFile`.

In that case, use the SMB fallback helper from an elevated PowerShell session on the host:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\hyperv\share_build_for_vm.ps1 \
  -SourcePath ".\releases\360ToolkitGS-v1.3.0-windows-x64-full-bundled.zip"
```

That script:

- copies the selected file or folder into a local share folder
- creates an SMB share named `360toolkit_vm_incoming`
- prints the UNC path for the VM to open, such as `\\HOSTNAME\360toolkit_vm_incoming`

Inside the VM, open the UNC path in File Explorer or PowerShell and copy or extract the ZIP locally.

## GPU-P caveats

- GPU partitioning requires Administrator rights.
- On Microsoft documentation, the fully documented GPU partition workflow is Windows Server-focused.
- On client Windows hosts, GPU-P behavior varies by GPU, driver, BIOS SR-IOV support, and Windows build.
- Even when the adapter is attached, you may still need the correct guest GPU driver path to expose useful CUDA behavior.

## If you only need app isolation

If GPU-P is not available on the host, still use the same persistent VM flow without `-TryGpuPartition`. That remains more realistic than Sandbox for validating:

- missing runtimes
- file path assumptions
- clean-user-state problems
- packaging regressions
- export and reconstruction path handling