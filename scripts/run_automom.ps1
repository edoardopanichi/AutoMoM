$ErrorActionPreference = 'Stop'

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if (Test-Path .env) {
  Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    $parts = $_ -split '=', 2
    if ($parts.Count -eq 2) {
      [Environment]::SetEnvironmentVariable($parts[0].Trim(), $parts[1].Trim(), 'Process')
    }
  }
}

if (-not (Test-Path .venv)) {
  python -m venv .venv
}

$PythonExe = Join-Path $RootDir '.venv\Scripts\python.exe'
if (-not (Test-Path $PythonExe)) {
  throw "Virtual environment python not found at $PythonExe"
}

& $PythonExe -m pip install --upgrade pip | Out-Null
& $PythonExe -m pip install -r requirements.txt -r requirements-dev.txt | Out-Null

$OllamaHost = if ($env:AUTOMOM_OLLAMA_HOST) { $env:AUTOMOM_OLLAMA_HOST } else { 'http://127.0.0.1:11434' }
$OllamaAutostart = if ($env:AUTOMOM_OLLAMA_AUTOSTART) { $env:AUTOMOM_OLLAMA_AUTOSTART } else { '1' }
$FormatterBackend = if ($env:AUTOMOM_FORMATTER_BACKEND) { $env:AUTOMOM_FORMATTER_BACKEND } else { 'ollama' }
$OllamaLogPath = Join-Path $RootDir 'data\ollama.log'

function Test-OllamaReady {
  try {
    Invoke-WebRequest -Uri "$OllamaHost/api/tags" -Method Get -TimeoutSec 2 | Out-Null
    return $true
  }
  catch {
    return $false
  }
}

function Start-OllamaIfNeeded {
  if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    throw "'ollama' is not installed. Install it first: https://docs.ollama.com/windows"
  }

  if (Test-OllamaReady) { return }

  if ($OllamaAutostart -ne '1') {
    throw "Ollama API is not reachable at $OllamaHost and autostart is disabled. Run 'ollama serve' manually or set AUTOMOM_OLLAMA_AUTOSTART=1."
  }

  New-Item -ItemType Directory -Force -Path (Join-Path $RootDir 'data') | Out-Null
  $null = Start-Process -FilePath 'ollama' -ArgumentList 'serve' -RedirectStandardOutput $OllamaLogPath -RedirectStandardError $OllamaLogPath -PassThru

  for ($i = 0; $i -lt 30; $i++) {
    if (Test-OllamaReady) {
      Write-Host "Ollama ready at $OllamaHost"
      return
    }
    Start-Sleep -Seconds 1
  }

  throw "Ollama failed to start. Check logs at $OllamaLogPath"
}

if ($FormatterBackend.ToLowerInvariant() -eq 'ollama') {
  Start-OllamaIfNeeded
}

$HostBind = if ($env:AUTOMOM_HOST) { $env:AUTOMOM_HOST } else { '127.0.0.1' }
$PortBind = if ($env:AUTOMOM_PORT) { $env:AUTOMOM_PORT } else { '8000' }

& $PythonExe -m uvicorn backend.app.main:app --host $HostBind --port $PortBind
