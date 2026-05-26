param(
    [switch]$Clean
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if ($Clean) {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "build", "dist"
}

uv run --with pyinstaller pyinstaller --clean --noconfirm gym_vision.spec

Copy-Item -Force "config.yaml" "dist\config.yaml"

Write-Host ""
Write-Host "Build complete:"
Write-Host "  dist\gym_vision.exe"
Write-Host "  dist\config.yaml"
Write-Host ""
Write-Host "Edit dist\config.yaml after build to change cameras, websocket URL, DB path, FPS, and other runtime settings."
