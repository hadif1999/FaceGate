param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot
$SpecPath = Join-Path $ProjectRoot "gym_vision.spec"

if (-not (Test-Path $SpecPath)) {
    throw "PyInstaller spec file not found: $SpecPath. Make sure gym_vision.spec exists in the project root."
}

if ($Clean) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "build", "dist"
}

uv run --with pyinstaller pyinstaller --clean --noconfirm $SpecPath

Copy-Item -Force "config.yaml" "dist\config.yaml"

Write-Host ""
Write-Host "Build complete:"
Write-Host "  dist\gym_vision.exe"
Write-Host "  dist\config.yaml"
Write-Host ""
Write-Host "Edit dist\config.yaml after build to change cameras, websocket URL, DB path, FPS, and other runtime settings."
