param(
    [int]$Workers = 10
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv\Scripts\python.exe"
$sourceRoot = Join-Path $repoRoot "src"
$env:PYTHONPATH = $sourceRoot

$lowConfig = Join-Path $repoRoot "examples\m2_dynamic_round1_proxy_baseline.json"
$mediumConfig = Join-Path $repoRoot "examples\m2_dynamic_round1_proxy_medium_friction.json"
$highConfig = Join-Path $repoRoot "examples\m2_dynamic_round1_proxy_high_friction.json"

$lowOutput = Join-Path $repoRoot "results\m2_dynamic_round1_proxy_m220"
$frictionOutput = Join-Path $repoRoot "results\m2_dynamic_round1_proxy_friction_m220"

$lowCampaign = Join-Path $lowOutput "campaign_a0425310a96263ebf26c"
$mediumCampaign = Join-Path $frictionOutput "campaign_2e801627888086dcffd3"
$highCampaign = Join-Path $frictionOutput "campaign_a84f0573e4551236dce2"

function Get-CompletedCount {
    param([string]$CampaignPath)

    $paths = Join-Path $CampaignPath "paths"
    if (-not (Test-Path -LiteralPath $paths)) {
        return 0
    }
    return @(
        Get-ChildItem -LiteralPath $paths -Recurse -Filter "summary.json" `
            -ErrorAction SilentlyContinue
    ).Count
}

function Test-CampaignProcess {
    param([string]$ConfigName)

    return $null -ne (
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -match "^python(.exe)?$" -and
                $_.CommandLine -like "*spine_sim.cli*" -and
                $_.CommandLine -like "*$ConfigName*"
            } |
            Select-Object -First 1
    )
}

function Complete-Campaign {
    param(
        [string]$ConfigPath,
        [string]$OutputPath,
        [string]$CampaignPath
    )

    while ((Get-CompletedCount $CampaignPath) -lt 900) {
        if (Test-Path -LiteralPath $CampaignPath) {
            & $python -m spine_sim.cli resume $ConfigPath `
                --output $OutputPath --workers $Workers
        }
        else {
            & $python -m spine_sim.cli run-campaign $ConfigPath `
                --output $OutputPath --workers $Workers
        }
        if ($LASTEXITCODE -ne 0) {
            Write-Warning (
                "Campaign exited with code $LASTEXITCODE; retrying from saved cases."
            )
            Start-Sleep -Seconds 5
        }
    }
}

# The low-friction campaign was already running before this queue was created.
# Do not compete for CPU; wait for that process and resume it only if it exits
# before all 900 case summaries exist.
while ((Get-CompletedCount $lowCampaign) -lt 900) {
    if (Test-CampaignProcess "m2_dynamic_round1_proxy_baseline.json") {
        Start-Sleep -Seconds 30
    }
    else {
        Complete-Campaign $lowConfig $lowOutput $lowCampaign
    }
}

Complete-Campaign $mediumConfig $frictionOutput $mediumCampaign
Complete-Campaign $highConfig $frictionOutput $highCampaign

[pscustomobject]@{
    low_completed = Get-CompletedCount $lowCampaign
    medium_completed = Get-CompletedCount $mediumCampaign
    high_completed = Get-CompletedCount $highCampaign
} | ConvertTo-Json
