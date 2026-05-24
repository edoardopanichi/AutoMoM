$ErrorActionPreference = 'Stop'

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

$DiarizationDir = Join-Path $RootDir 'data\models\diarization\pyannote-speaker-diarization-community-1'
$TranscriptionDir = Join-Path $RootDir 'data\models\transcription'
$FormatterDir = Join-Path $RootDir 'data\models\formatter'

New-Item -ItemType Directory -Force -Path $DiarizationDir, $TranscriptionDir, $FormatterDir | Out-Null

$ConfigPath = Join-Path $DiarizationDir 'config.yaml'
if (-not (Test-Path $ConfigPath)) {
@"
version: 3.1
pipeline:
  name: mock
"@ | Set-Content -Path $ConfigPath -Encoding UTF8
}

$ModelPath = Join-Path $TranscriptionDir 'model.gguf'
if (-not (Test-Path $ModelPath)) {
  Set-Content -Path $ModelPath -Value 'mock-transcription' -NoNewline -Encoding UTF8
}

$SelectedModelPath = Join-Path $FormatterDir 'selected_model.txt'
if (-not (Test-Path $SelectedModelPath)) {
  Set-Content -Path $SelectedModelPath -Value 'llama3.1:8b-instruct-q4_K_M' -NoNewline -Encoding UTF8
}

$ModelsDir = Join-Path $RootDir 'data\models'
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
$ConsentPath = Join-Path $ModelsDir 'consent.json'
@"
{
  "diarization": true,
  "transcription": true,
  "formatter": true
}
"@ | Set-Content -Path $ConsentPath -Encoding UTF8

Write-Host 'Mock models prepared in data/models'
