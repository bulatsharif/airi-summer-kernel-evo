$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $env:QWEN_BASE_URL) {
    throw "QWEN_BASE_URL is not set"
}
if (-not $env:QWEN_API_KEY) {
    throw "QWEN_API_KEY is not set"
}

$baseUrl = $env:QWEN_BASE_URL.TrimEnd("/")
$headers = @{
    Authorization = "Bearer $env:QWEN_API_KEY"
}

$models = Invoke-RestMethod `
    -Method Get `
    -Uri "$baseUrl/models" `
    -Headers $headers `
    -TimeoutSec 20

$modelIds = @($models.data | ForEach-Object { $_.id })
if ($modelIds -notcontains "qwen3.6-35b-a3b") {
    throw "qwen3.6-35b-a3b is missing from /models: $($modelIds -join ', ')"
}

$body = @{
    model = "qwen3.6-35b-a3b"
    messages = @(
        @{
            role = "user"
            content = "Reply with exactly OPENCODE_QWEN_OK"
        }
    )
    temperature = 0
    max_tokens = 16
} | ConvertTo-Json -Depth 5

$chat = Invoke-RestMethod `
    -Method Post `
    -Uri "$baseUrl/chat/completions" `
    -Headers $headers `
    -ContentType "application/json" `
    -Body $body `
    -TimeoutSec 60

$content = $chat.choices[0].message.content
if ($content.Trim() -ne "OPENCODE_QWEN_OK") {
    throw "unexpected model response: $content"
}

Write-Output (
    "qwen_endpoint=PASS model=qwen3.6-35b-a3b " +
    "response=$($content.Trim())"
)
