param([int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot
$python = Join-Path $repoRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw 'Run setup first: .\scripts\setup_monitor.ps1' }
$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) { throw "Port $Port is already in use by PID $($listener.OwningProcess)" }
$arguments = @('-m', 'uvicorn', 'src.runtime.app:app', '--host', '127.0.0.1', '--port', "$Port")
if (Test-Path '.env.runtime') { $arguments += @('--env-file', '.env.runtime') }
& $python @arguments
