# Prefer the existing project environment, which owns training dependencies.
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location -LiteralPath $projectRoot
if (Test-Path -LiteralPath "$projectRoot\.venv\Scripts\python.exe") {
    & "$projectRoot\.venv\Scripts\python.exe" -m uvicorn main:app --host 127.0.0.1 --port 8000 @args
    exit $LASTEXITCODE
}
if (Test-Path -LiteralPath "$projectRoot\.packages") {
    $env:PYTHONPATH = "$projectRoot\.packages"
}
python -m uvicorn main:app --host 127.0.0.1 --port 8000 @args
