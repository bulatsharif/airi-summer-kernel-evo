param(
    [string]$RemoteHost = "176.109.107.137",
    [string]$RemoteUser = "User17",
    [int]$LocalPort = 18001,
    [int]$RemotePort = 8001
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Test-TunnelPort {
    param([int]$Port)

    return [bool](Get-NetTCPConnection `
        -LocalAddress 127.0.0.1 `
        -LocalPort $Port `
        -State Listen `
        -ErrorAction SilentlyContinue)
}

$projectRoot = Split-Path -Parent $PSScriptRoot
$secretsPath = Join-Path $projectRoot ".opencode.local.ps1"
if (-not (Test-Path -LiteralPath $secretsPath)) {
    throw "Local secrets file was not found: $secretsPath"
}
. $secretsPath

$runningDesktop = Get-Process -Name OpenCode -ErrorAction SilentlyContinue
if ($runningDesktop) {
    throw "OpenCode is already running. Fully quit it, then run this script again."
}

foreach ($requiredName in @(
    "QWEN_API_KEY",
    "CUTE_HARNESS_API_KEY",
    "SCHOOL17_PASSWORD"
)) {
    $requiredValue = [Environment]::GetEnvironmentVariable(
        $requiredName,
        "Process"
    )
    if ([string]::IsNullOrWhiteSpace($requiredValue)) {
        throw "$requiredName is missing from $secretsPath"
    }
}

$env:QWEN_BASE_URL = "http://127.0.0.1:$LocalPort/v1"
$env:CUTE_HARNESS_URL = "http://109.236.57.62:18080"

# Work around the packaged Desktop 1.18.x dependency installer requesting the
# nonexistent @opencode-ai/plugin@local package.
$env:OPENCODE_DISABLE_DEFAULT_PLUGINS = "1"
$env:OPENCODE_PURE = "1"

if (-not (Test-TunnelPort -Port $LocalPort)) {
    $pythonw = Get-Command pythonw -ErrorAction Stop
    $tunnelHelper = Join-Path $PSScriptRoot "qwen_tunnel.py"
    $tunnelArguments = @(
        "`"$tunnelHelper`"",
        "--host", $RemoteHost,
        "--user", $RemoteUser,
        "--local-port", $LocalPort,
        "--remote-port", $RemotePort
    )
    $tunnelProcess = Start-Process `
        -FilePath $pythonw.Source `
        -ArgumentList $tunnelArguments `
        -WindowStyle Hidden `
        -PassThru

    $ready = $false
    foreach ($attempt in 1..20) {
        if (Test-TunnelPort -Port $LocalPort) {
            $ready = $true
            break
        }
        Start-Sleep -Milliseconds 250
    }
    if (-not $ready) {
        if (-not $tunnelProcess.HasExited) {
            Stop-Process -Id $tunnelProcess.Id
        }
        throw "The background SSH tunnel failed to listen on 127.0.0.1:$LocalPort"
    }
} else {
    Write-Host "Reusing the existing tunnel on 127.0.0.1:$LocalPort."
}

$desktopPath = Join-Path `
    $env:LOCALAPPDATA `
    "Programs\@opencode-aidesktop\OpenCode.exe"

if (-not (Test-Path -LiteralPath $desktopPath)) {
    throw "OpenCode Desktop was not found at $desktopPath"
}

Write-Host "Starting OpenCode Desktop."
Write-Host "Open this project in the UI: $projectRoot"
Start-Process -FilePath $desktopPath -WorkingDirectory $projectRoot
