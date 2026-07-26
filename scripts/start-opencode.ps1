param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$OpenCodeArgs
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$projectRoot = Split-Path -Parent $PSScriptRoot
if (-not $env:QWEN_BASE_URL) {
    $env:QWEN_BASE_URL = "http://127.0.0.1:18001/v1"
}
if (-not $env:QWEN_API_KEY) {
    throw "QWEN_API_KEY is not set"
}

& (Join-Path $PSScriptRoot "test-qwen-endpoint.ps1")

Push-Location $projectRoot
try {
    $installed = Get-Command opencode -ErrorAction SilentlyContinue
    if ($installed) {
        & $installed.Source @OpenCodeArgs
    } else {
        $npx = Get-Command npx -ErrorAction Stop
        Write-Host "OpenCode is not globally installed; starting opencode-ai via npx."
        & $npx.Source -y opencode-ai @OpenCodeArgs
    }
    if ($LASTEXITCODE -ne 0) {
        throw "OpenCode exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
