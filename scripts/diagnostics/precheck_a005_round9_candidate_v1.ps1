param(
    [Parameter(Mandatory = $true)][string]$Repository
)

$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath $Repository).Path
$fixedMd = Join-Path $root 'refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE.md'
$fixedJson = Join-Path $root 'refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE.json'
$oldMd = Join-Path $root 'refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260817_134853.md'
$oldJson = Join-Path $root 'refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260817_134853.json'
$newMd = Join-Path $root 'refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260819_195424.md'
$newJson = Join-Path $root 'refine-logs/temporac/TEMPORAC_CONTRACT_AMENDMENT_005_CERTIFICATE_PROVENANCE_20260819_195424.json'

$expected = @{
    $fixedMd = '6e0ce987d88c1d3f7abbaec47d28630efdc9a3012c7192da6b692feb9f55d981'
    $fixedJson = 'b135fc10c01aa6611af131d44b5ca762d802df514184ba6c76dfe5b6c7a80ef0'
    $oldMd = '7da6647f7c37f981dd516d7b152123a156b206732a77a1ee82c7a3e492ab77bb'
    $oldJson = '4cff6d4ac91c0114093ee6aeed0247eb2fcda2bc6f4192c266ccfe67977ab898'
    $newMd = '38e607a36d4de8f70daf5ea4ae612377a74004f860aa102819dac29b1d1585c1'
    $newJson = 'dc8e205ad9b1ba842d692756a7a2fddc8dc1f1229533c027e522920d61242ba6'
}
foreach ($path in $expected.Keys) {
    $actual = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -cne $expected[$path]) {
        throw "SHA-256 mismatch: $path"
    }
}

$jsonText = [IO.File]::ReadAllText($newJson, [Text.UTF8Encoding]::new($false))
$null = $jsonText | ConvertFrom-Json
$markdownText = [IO.File]::ReadAllText($newMd, [Text.UTF8Encoding]::new($false))
$startMarker = "~~~json`n"
$endMarker = "`n~~~`n"
$start = $markdownText.IndexOf($startMarker, [StringComparison]::Ordinal)
if ($start -lt 0) { throw 'Markdown JSON start marker missing' }
$start += $startMarker.Length
$end = $markdownText.IndexOf($endMarker, $start, [StringComparison]::Ordinal)
if ($end -lt 0) { throw 'Markdown JSON end marker missing' }
$appendix = $markdownText.Substring($start, $end - $start) + "`n"
if ($appendix -cne $jsonText) { throw 'Markdown appendix differs from standalone JSON' }

$forbidden = @(
    'FD32/original result failure',
    'Parent resumes only by PTRACE_SYSCALL',
    'Every PTRACE_SYSCALL resume selects exactly one closed gate',
    'selects exactly one expanded outcome by (K,current stage,action,mapped token)',
    'unique expanded outcomes by K,stage,action,token',
    'fresh bound root identity'
)
foreach ($text in $forbidden) {
    if ($markdownText.Contains($text, [StringComparison]::Ordinal) -or
        $jsonText.Contains($text, [StringComparison]::Ordinal)) {
        throw "obsolete semantic string remains: $text"
    }
}

$required = @(
    'ATTACH_BOOTSTRAP',
    'AUTHORITY_TRANSITION',
    'FAILURE_TERMINATION',
    'type0x7f failure frame',
    'outputs of that selected row and are never selector inputs',
    'repairs the same observable root in place'
)
foreach ($text in $required) {
    if (-not $markdownText.Contains($text, [StringComparison]::Ordinal)) {
        throw "required successor semantic string missing: $text"
    }
}

[ordered]@{
    schema = 'temporac.a005-round9-author-precheck.powershell.v1'
    result = 'PASS'
    fixed_aliases_unchanged = $true
    old_candidate_unchanged = $true
    appendix_matches_json = $true
    new_markdown_sha256 = $expected[$newMd]
    new_json_sha256 = $expected[$newJson]
} | ConvertTo-Json -Compress
