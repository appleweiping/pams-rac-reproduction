$ErrorActionPreference = 'Stop'

$paperDir = Split-Path -Parent $PSScriptRoot
$gitPerlDir = 'C:\Program Files\Git\usr\bin'
if (-not (Get-Command perl -ErrorAction SilentlyContinue) -and
    (Test-Path -LiteralPath (Join-Path $gitPerlDir 'perl.exe'))) {
    $env:Path = "$gitPerlDir;$env:Path"
}
$previousTexInputs = $env:TEXINPUTS
$templateDir = Join-Path $paperDir 'vendor\icassp2026'
$env:TEXINPUTS = "$templateDir;$previousTexInputs"
Push-Location $paperDir
try {
    latexmk -C main.tex | Out-Host
    $latexCommand = 'latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex 2>&1'
    & cmd.exe /d /s /c $latexCommand |
        Tee-Object -FilePath compile.log | Out-Host
    $compileExitCode = $LASTEXITCODE
    if ($compileExitCode -ne 0) {
        throw "Draft compilation failed with exit code $compileExitCode"
    }
}
finally {
    Pop-Location
    $env:TEXINPUTS = $previousTexInputs
}
