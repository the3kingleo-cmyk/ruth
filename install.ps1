# Install Ruth on Windows (PowerShell).  Usage:  powershell -ExecutionPolicy Bypass -File install.ps1
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = $null
foreach ($c in @("py", "python", "python3")) {
  if (Get-Command $c -ErrorAction SilentlyContinue) { $py = $c; break }
}
if (-not $py) { Write-Host "Ruth needs Python 3.9+ from https://www.python.org/downloads/"; exit 1 }
$dest = if ($env:RUTH_INSTALL_DIR) { $env:RUTH_INSTALL_DIR } else { Join-Path $env:LOCALAPPDATA "Ruth\app" }
Write-Host "Installing Ruth into $dest"
& $py -m venv $dest
& "$dest\Scripts\python.exe" -m pip install --quiet --upgrade pip
& "$dest\Scripts\python.exe" -m pip install --quiet $here
$bin = Join-Path $dest "Scripts"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if (-not ($userPath -split ";" | Where-Object { $_ -eq $bin })) {
  [Environment]::SetEnvironmentVariable("Path", "$userPath;$bin", "User")
  Write-Host "Added $bin to your PATH (open a new terminal)."
}
Write-Host "Done. Start her with:  ruth app     (or: ruth talk)"
