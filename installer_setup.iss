; 360FrameTools Installer Script for Inno Setup
; Download Inno Setup from: https://jrsoftware.org/isinfo.php

[Setup]
AppName=360FrameTools
AppVersion=1.0.0
AppPublisher=360FrameTools Development Team
AppPublisherURL=https://github.com/yourusername/360FrameTools
DefaultDirName={autopf}\360FrameTools
DefaultGroupName=360FrameTools
OutputDir=installer_output
OutputBaseFilename=360FrameTools-GPU-Setup-v1.0.0
Compression=lzma2/ultra64
SolidCompression=yes
SetupIconFile=resources\icon.ico
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\360ToolkitGS-GPU.exe
LicenseFile=LICENSE.txt
; Minimum Windows 10 required
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Main executable and all dependencies
Source: "dist\360ToolkitGS-GPU\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; GPU installation helper scripts
Source: "install_pytorch_gpu.bat"; DestDir: "{app}"; Flags: ignoreversion
Source: "README_GPU_VERSION.md"; DestDir: "{app}"; DestName: "README.txt"; Flags: ignoreversion isreadme

[Icons]
Name: "{group}\360FrameTools"; Filename: "{app}\360ToolkitGS-GPU.exe"
Name: "{group}\Install PyTorch GPU Support"; Filename: "{app}\install_pytorch_gpu.bat"; Comment: "Install PyTorch and Ultralytics for AI masking"
Name: "{group}\README"; Filename: "{app}\README.txt"
Name: "{group}\{cm:UninstallProgram,360FrameTools}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\360FrameTools"; Filename: "{app}\360ToolkitGS-GPU.exe"; Tasks: desktopicon

[Run]
; Optional: Run after installation
Filename: "{app}\README.txt"; Description: "View README (GPU installation instructions)"; Flags: postinstall shellexec skipifsilent
Filename: "{app}\install_pytorch_gpu.bat"; Description: "Install GPU support (PyTorch + Ultralytics)"; Flags: postinstall skipifsilent

[Code]
// Check if Visual C++ Redistributable is installed
function VCRedistNeedsInstall: Boolean;
var
  Version: String;
begin
  // Check for VC++ 2015-2022 Redistributable (required by PyQt6 and OpenCV)
  if RegQueryStringValue(HKLM, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64', 'Version', Version) then
    Result := False // Already installed
  else
    Result := True; // Needs installation
end;

procedure InitializeWizard;
begin
  if VCRedistNeedsInstall then
  begin
    MsgBox('This application requires Visual C++ Redistributable 2015-2022.' + #13#10 + 
           'The installer will download it automatically.', mbInformation, MB_OK);
  end;
end;
