# Finish Requirement 7.5 after labeling results/judge/owner_labels.csv.
# Run from the repository root:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\finalize_owner_labels.ps1
# Commits the labels, computes Cohen's kappa, regenerates results/summary.md and the README,
# checks the README's numbers, commits, and pushes.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$wsl = @("-d", "Ubuntu-24.04", "--", "bash", "/mnt/c/dev/Grounded-QA-Service/scripts/wsl_run.sh")

function Invoke-Step([string]$what, [string[]]$command) {
    Write-Host "== $what"
    & wsl @wsl @command
    if ($LASTEXITCODE -ne 0) { throw "$what failed (exit $LASTEXITCODE)" }
}

git add results/judge/owner_labels.csv
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
    git commit -q -m "owner faithfulness labels for the judge-agreement check (Req 7.5)"
    if ($LASTEXITCODE -ne 0) { throw "git commit of the labels failed" }
}
Invoke-Step "judge agreement (Cohen's kappa)" @("python", "scripts/finalize_owner_labels.py")
Invoke-Step "results table" @("python", "scripts/write_results_table.py")
Invoke-Step "README" @("python", "scripts/render_readme.py")
Invoke-Step "README number check" @("python", "scripts/check_readme.py")
git add results README.md
git commit -q -m "judge agreement from owner labels; regenerate results table and README"
if ($LASTEXITCODE -ne 0) { throw "git commit failed" }
git push -q origin main
if ($LASTEXITCODE -ne 0) { throw "git push failed" }
Write-Host "Done: results/summary.md and README.md now report Cohen's kappa."
