param(
    [string]$SshHost = "gpu",
    [int]$LocalPort = 8000,
    [int]$RemotePort = 8000,
    [string]$Model = "Qwen3-0.6B",
    [int]$MaxTokens = 1024,
    [double]$Temperature = 1.0,
    [string]$SystemPrompt = "",
    [switch]$Thinking
)

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$clientPath = Join-Path $repoRoot "tools\chat_interactive.py"
$venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
$forward = "${LocalPort}:127.0.0.1:${RemotePort}"
$tunnelLog = [System.IO.Path]::GetTempFileName()
$tunnel = Start-Process -FilePath "ssh" `
    -ArgumentList @(
        "-N",
        "-o", "ExitOnForwardFailure=yes",
        "-o", "ServerAliveInterval=15",
        "-o", "ServerAliveCountMax=3",
        "-L", $forward,
        $SshHost
    ) `
    -PassThru `
    -RedirectStandardError $tunnelLog `
    -WindowStyle Hidden

try {
    $tunnelReady = $false
    $deadline = (Get-Date).AddSeconds(10)
    while ((Get-Date) -lt $deadline) {
        if ($tunnel.HasExited) {
            $sshError = Get-Content -LiteralPath $tunnelLog -Raw -ErrorAction SilentlyContinue
            throw "SSH tunnel exited before becoming ready. $sshError"
        }
        $connection = [System.Net.Sockets.TcpClient]::new()
        try {
            $connectTask = $connection.ConnectAsync("127.0.0.1", $LocalPort)
            if ($connectTask.Wait(500) -and $connection.Connected) {
                $tunnelReady = $true
                break
            }
        }
        catch {
            # The SSH process is still negotiating; retry until the deadline.
        }
        finally {
            $connection.Dispose()
        }
        Start-Sleep -Milliseconds 200
    }
    if (-not $tunnelReady) {
        throw "SSH tunnel did not become ready on local port $LocalPort within 10 seconds."
    }

    $clientArgs = @(
        $clientPath,
        "--base-url", "http://127.0.0.1:$LocalPort",
        "--model", $Model,
        "--max-tokens", $MaxTokens,
        "--temperature", $Temperature
    )
    if ($SystemPrompt) {
        $clientArgs += @("--system-prompt", $SystemPrompt)
    }
    if ($Thinking) {
        $clientArgs += "--thinking"
    }

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
    Remove-Item -LiteralPath $tunnelLog -Force -ErrorAction SilentlyContinue
}
