$ErrorActionPreference = 'Stop'

$paperDir = Split-Path -Parent $PSScriptRoot
$validator = Join-Path $PSScriptRoot 'validate_submission.py'
$claimScanner = Join-Path $PSScriptRoot 'claim_scan.py'
$anonymityScanner = Join-Path $PSScriptRoot 'anonymity_scan.py'
$gitPerlDir = 'C:\Program Files\Git\usr\bin'
if (-not (Get-Command perl -ErrorAction SilentlyContinue) -and
    (Test-Path -LiteralPath (Join-Path $gitPerlDir 'perl.exe'))) {
    $env:Path = "$gitPerlDir;$env:Path"
}
$previousTexInputs = $env:TEXINPUTS
$templateDir = Join-Path $paperDir 'vendor\cvpr2026'
$env:TEXINPUTS = "$templateDir;$previousTexInputs"

python $validator --paper-dir $paperDir
if ($LASTEXITCODE -ne 0) {
    throw 'Submission source validation failed. This is expected while results or implementation synchronization are pending.'
}
python $claimScanner --paper-dir $paperDir --output (Join-Path $paperDir '.aris\claim-scan-submission.json')
if ($LASTEXITCODE -ne 0) {
    throw 'Submission-wide source claim scan failed.'
}

Push-Location $paperDir
try {
    latexmk -C -jobname=main_submission submission_wrapper.tex | Out-Host
    latexmk -pdf -interaction=nonstopmode -halt-on-error -jobname=main_submission submission_wrapper.tex | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Submission compilation failed with exit code $LASTEXITCODE"
    }
    $pdfText = Join-Path $paperDir '.aris\main_submission.txt'
    pdftotext -layout main_submission.pdf $pdfText
    if ($LASTEXITCODE -ne 0) {
        throw 'Submission PDF text extraction failed.'
    }
    python $claimScanner --paper-dir $paperDir --pdf-text $pdfText --output (Join-Path $paperDir '.aris\claim-scan-submission-pdf.json')
    if ($LASTEXITCODE -ne 0) {
        throw 'Submission PDF claim scan failed.'
    }
    $secretFile = $env:RAC_ANONYMITY_SECRET_FILE
    if (-not $secretFile -or -not (Test-Path -LiteralPath $secretFile)) {
        throw 'RAC_ANONYMITY_SECRET_FILE must name the private out-of-repository secret list.'
    }
    python $anonymityScanner --root (Split-Path -Parent $paperDir) --secret-file $secretFile --pdf-text $pdfText --output (Join-Path $paperDir '.aris\anonymity-scan-submission.json')
    if ($LASTEXITCODE -ne 0) {
        throw 'Submission anonymity scan failed.'
    }
}
finally {
    Pop-Location
    $env:TEXINPUTS = $previousTexInputs
}
