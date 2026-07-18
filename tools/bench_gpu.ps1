param(
    [string]$SshHost = "gpu",
    [int]$LocalPort = 8001,
    [int]$RemotePort = 8000,
    [string]$Model = "Qwen3-0.6B",
    [string]$Profile = "smoke",
    [int]$NumRequests = 0,
    [int]$MaxConcurrency = 0,
    [string]$RequestRate = "",
    [int]$InputLen = 0,
    [int]$OutputLen = 0,
    [string[]]$Metadata = @(),
    [string]$OutputJson = "",
    [switch]$RequestDetails,
    [switch]$AllowErrors
)

$repoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$clientPath = Join-Path $repoRoot "bench_online.py"
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
        "--profile", $Profile
    )
    if ($NumRequests -gt 0) { $clientArgs += @("--num-requests", $NumRequests) }
    if ($MaxConcurrency -gt 0) { $clientArgs += @("--max-concurrency", $MaxConcurrency) }
    if ($RequestRate) { $clientArgs += @("--request-rate", $RequestRate) }
    if ($InputLen -gt 0) { $clientArgs += @("--input-len", $InputLen) }
    if ($OutputLen -gt 0) { $clientArgs += @("--output-len", $OutputLen) }
    foreach ($item in $Metadata) { $clientArgs += @("--metadata", $item) }
    if ($OutputJson) { $clientArgs += @("--output-json", $OutputJson) }
    if ($RequestDetails) { $clientArgs += "--request-details" }
    if ($AllowErrors) { $clientArgs += "--allow-errors" }

    $clientRan = $false
    $clientExitCode = 1
    if (Test-Path -LiteralPath $venvPython) {
        & $venvPython -c "import httpx" *> $null
        if ($LASTEXITCODE -eq 0) {
            & $venvPython @clientArgs
            $clientExitCode = $LASTEXITCODE
            $clientRan = $true
        }
    }
    if (-not $clientRan -and (Get-Command python -ErrorAction SilentlyContinue)) {
        & python -c "import httpx" *> $null
        if ($LASTEXITCODE -eq 0) {
            & python @clientArgs
            $clientExitCode = $LASTEXITCODE
            $clientRan = $true
        }
    }
    if (-not $clientRan) {
        $uvCache = Join-Path $repoRoot ".uv-benchmark-cache"
        & uv --cache-dir $uvCache run --isolated --no-project --with "httpx>=0.27.0" python @clientArgs
        $clientExitCode = $LASTEXITCODE
    }
    exit $clientExitCode
}
finally {
    if (-not $tunnel.HasExited) {
        Stop-Process -Id $tunnel.Id
    }
    Remove-Item -LiteralPath $tunnelLog -Force -ErrorAction SilentlyContinue
}
