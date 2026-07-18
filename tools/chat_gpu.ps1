param(
    [Parameter(Position = 0)]
    [string]$Prompt = "Hello, introduce yourself in one sentence.",
    [string]$SshHost = "gpu",
    [int]$LocalPort = 8000,
    [int]$RemotePort = 8000,
    [string]$Model = "Qwen3-0.6B"
)

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$clientPath = Join-Path $repoRoot "tools\chat_stream.py"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$forward = "${LocalPort}:127.0.0.1:${RemotePort}"
$tunnel = Start-Process -FilePath "ssh" `
    -ArgumentList @("-N", "-L", $forward, $SshHost) `
    -PassThru `
    -WindowStyle Hidden

try {
    Start-Sleep -Seconds 1
    $clientArgs = @(
        $clientPath,
        "--base-url", "http://127.0.0.1:$LocalPort",
        "--model", $Model,
        $Prompt
    )
    if (Test-Path -LiteralPath $venvPython) {
        & $venvPython @clientArgs
    }
    elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python @clientArgs
    }
    else {
        $uvCache = Join-Path $repoRoot ".uv-chat-cache"
        & uv --cache-dir $uvCache run --no-project python @clientArgs
    }
    exit $LASTEXITCODE
}
finally {
    if (-not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id
    }
}
