param(
    [string]$RemoteHost = "176.109.107.137",
    [string]$RemoteUser = "User17",
    [int]$LocalPort = 18001,
    [int]$RemotePort = 8001
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ssh = Get-Command ssh -ErrorAction Stop
$existing = Get-NetTCPConnection `
    -LocalAddress 127.0.0.1 `
    -LocalPort $LocalPort `
    -State Listen `
    -ErrorAction SilentlyContinue

if ($existing) {
    throw "127.0.0.1:$LocalPort is already listening. Reuse or stop the existing tunnel."
}

$forward = "{0}:127.0.0.1:{1}" -f $LocalPort, $RemotePort
Write-Host "Opening Qwen tunnel:"
Write-Host "  127.0.0.1:$LocalPort -> ${RemoteHost}:127.0.0.1:$RemotePort"
Write-Host "Keep this terminal open. Press Ctrl+C to stop the tunnel."

& $ssh.Source `
    -N `
    -o ExitOnForwardFailure=yes `
    -o ServerAliveInterval=30 `
    -o ServerAliveCountMax=3 `
    -L $forward `
    "${RemoteUser}@${RemoteHost}"

if ($LASTEXITCODE -ne 0) {
    throw "ssh tunnel exited with code $LASTEXITCODE"
}
