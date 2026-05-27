param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

function Get-UvCommand {
    return Get-Command "uv" -ErrorAction SilentlyContinue
}

function Ensure-Uv {
    $UvCommand = Get-UvCommand
    if ($null -ne $UvCommand) {
        return $UvCommand.Source
    }

    Write-Host "uv is not installed. Installing uv..."
    powershell -ExecutionPolicy Bypass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"

    $UvBin = Join-Path $env:USERPROFILE ".local\bin"
    if (Test-Path $UvBin) {
        $env:Path = "$UvBin;$env:Path"
    }

    $UvCommand = Get-UvCommand
    if ($null -eq $UvCommand) {
        throw "uv installation completed but uv was not found on PATH. Restart PowerShell or add $UvBin to PATH."
    }

    return $UvCommand.Source
}

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$SpecPath = Join-Path $ProjectRoot "gym_vision.spec"
$GuiSpecPath = Join-Path $ProjectRoot "test_ws_server_gui.spec"
$Uv = Ensure-Uv

if (-not (Test-Path $SpecPath)) {
    throw "PyInstaller spec file not found: $SpecPath. Make sure gym_vision.spec exists in the project root."
}
if (-not (Test-Path $GuiSpecPath)) {
    throw "PyInstaller spec file not found: $GuiSpecPath. Make sure test_ws_server_gui.spec exists in the project root."
}

if ($Clean) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "build", "dist"
}

& $Uv run --with pyinstaller pyinstaller --clean --noconfirm $SpecPath
& $Uv run --with pyinstaller pyinstaller --clean --noconfirm $GuiSpecPath

Copy-Item -Force "config.yaml" "dist\config.yaml"

Write-Host ""
Write-Host "Build complete:"
Write-Host "  dist\gym_vision.exe"
Write-Host "  dist\test_ws_server_gui.exe"
Write-Host "  dist\config.yaml"
Write-Host ""
Write-Host "Edit dist\config.yaml after build to change cameras, websocket URL, DB path, FPS, and other runtime settings."
