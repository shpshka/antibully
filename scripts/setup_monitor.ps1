param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
if (-not (Test-Path '.venv')) { & $Python -m venv .venv }
& '.\.venv\Scripts\python.exe' -m pip install --upgrade pip wheel
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-runtime.txt
if (-not (Test-Path '.env.runtime')) { Copy-Item '.env.runtime.example' '.env.runtime' }
Write-Host 'Ready. Start with .\scripts\start_monitor.ps1' -ForegroundColor Green
