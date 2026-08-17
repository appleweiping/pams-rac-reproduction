$ErrorActionPreference = 'Stop'

$paperDir = (Split-Path -Parent $PSScriptRoot)
$python = 'C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$gitPerlDir = 'C:\Program Files\Git\usr\bin'
$env:PYTHONUTF8 = '1'
if (-not (Get-Command perl -ErrorAction SilentlyContinue) -and
    (Test-Path -LiteralPath (Join-Path $gitPerlDir 'perl.exe'))) {
    $env:Path = "$gitPerlDir;$env:Path"
}

$readiness = Join-Path $PSScriptRoot 'validate_readiness.py'
$claims = Join-Path $PSScriptRoot 'claim_scan.py'
$pageGate = Join-Path $PSScriptRoot 'validate_pdf_4plus1.py'
$authors = Join-Path $PSScriptRoot 'validate_author_metadata.py'
$draftPdf = Join-Path $paperDir 'main.pdf'
$arisDir = Join-Path $paperDir '.aris'

# This preflight intentionally fails while any result, method, author, venue,
# audit, cross-family-review, or placeholder gate remains unresolved.
& $python $readiness --paper-dir $paperDir --mode submission --pdf $draftPdf `
  --output (Join-Path $arisDir 'readiness-submission-preflight.json')
if ($LASTEXITCODE -ne 0) {
    throw 'Submission preflight failed closed. Resolve the recorded blockers before compiling a submission artifact.'
}

Push-Location $paperDir
try {
    latexmk -pdf -interaction=nonstopmode -halt-on-error `
      -jobname=main_submission submission_wrapper.tex | Out-Host
    if ($LASTEXITCODE -ne 0) {
        throw "Submission compilation failed with exit code $LASTEXITCODE"
    }

    $submissionPdf = Join-Path $paperDir 'main_submission.pdf'
    $pdfText = Join-Path $arisDir 'main_submission.txt'
    pdftotext -layout $submissionPdf $pdfText
    if ($LASTEXITCODE -ne 0) {
        throw 'Submission PDF text extraction failed.'
    }

    & $python $claims --paper-dir $paperDir --mode submission --pdf-text $pdfText `
      --output (Join-Path $arisDir 'claim-scan-submission-pdf.json')
    if ($LASTEXITCODE -ne 0) { throw 'Submission source/PDF claim scan failed.' }

    & $python $pageGate $submissionPdf --paper-dir $paperDir --mode submission `
      --output (Join-Path $arisDir 'pdf-4plus1-submission.json')
    if ($LASTEXITCODE -ne 0) { throw 'ICASSP 4+1 page gate failed.' }

    & $python $authors --paper-dir $paperDir --mode submission `
      --output (Join-Path $arisDir 'author-metadata-submission.json')
    if ($LASTEXITCODE -ne 0) { throw 'Author or tracked-tree privacy gate failed.' }

    & $python $readiness --paper-dir $paperDir --mode submission --pdf $submissionPdf `
      --output (Join-Path $arisDir 'readiness-submission-final.json')
    if ($LASTEXITCODE -ne 0) { throw 'Final submission readiness gate failed.' }
}
finally {
    Pop-Location
}
