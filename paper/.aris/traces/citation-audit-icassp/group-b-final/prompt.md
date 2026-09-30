# Fresh ICASSP Citation Final Audit — Group B

## Reviewer assignment

- Agent: `/root/icassp_citation_reaudit_b`
- Model: `gpt-5.6-sol`
- Reasoning: `xhigh`
- Fork/context: `none` (fresh zero-context review)
- Review independence: `same-family`
- Acceptance status: `provisional`
- Audit date: 2026-08-14

## Exact task

Independently audit the current `paper/references.bib` and the active TeX citation contexts for exactly these eight keys:

1. `ma2026detrc`
2. `wang2026d2stx`
3. `welch1967fft`
4. `lomb1976unequal`
5. `griffin1984stft`
6. `shazeer2017moe`
7. `luiten2021hota`
8. `ristani2016idmetrics`

Do not inspect or reuse any previous citation audit, history, or log. Re-browse primary publisher/official records and authoritative CVF, Springer, IEEE, DBLP, and Crossref sources. For each key, decide:

- existence: `YES`, `NO`, or `UNCERTAIN`;
- metadata: title, authors, year, venue, volume/issue/pages or article number, DOI/URL;
- context: `SUPPORTS`, `WEAK`, or `WRONG` for every active occurrence;
- final entry verdict: `KEEP`, `FIX`, `REPLACE`, or `REMOVE`.

Do not modify manuscript TeX or `references.bib`. Write only the requested trace artifacts.

## Frozen input identity

The PDF gate is mandatory:

`paper/main.pdf` SHA-256 must equal `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45`.

The audited input snapshot is:

| File | SHA-256 |
|---|---|
| `paper/main.pdf` | `873c83da0dbf4b24a6e1199f3dcd980e17d0b032580e0591f88a7c545ec67a45` |
| `paper/references.bib` | `322a5d16b0078e8b3436608d4c6158b67c5c8922b0dcae94537ea456c9f6f646` |
| `paper/main.tex` | `a4fe29c47f200946c7d6f3522bd273f5895aabda2d7a959fdbf16397fe1cbe70` |
| `paper/preamble.tex` | `763b456cb2e22585ca23160c2b12d641e764c70da94641247f6268af1a7306a2` |
| `paper/math_commands.tex` | `e6dabb7ce789e0d78026e6bda287fe17faedb6d20c0df9b4f603cb3b6c123bee` |
| `paper/sections/0_abstract.tex` | `fd69c7936952bffc8e70d7d0ccd35140ab7dbaac82d6ff953e60f4f6461eebef` |
| `paper/sections/1_introduction.tex` | `6823651162f59e5f5263949f2d308f594b667cb3cba0044433c39031ae04db9d` |
| `paper/sections/3_method.tex` | `7e92c00eb4a7956b059536fd47240cefbea960aee61d4f3fdef1a7e24930c50d` |
| `paper/sections/4_experiments.tex` | `6052c816e94f7d7f50b560d397495a37863b05d240ed5e418e4b6ec4d3aa131b` |
| `paper/sections/5_conclusion.tex` | `e5c656a98343f10b367c175273d572f3a42d5436539429f28ba87764452e7b3c` |

## Active occurrences to review

| Key | Active TeX occurrence(s) |
|---|---|
| `ma2026detrc` | `sections/1_introduction.tex:30`; `sections/4_experiments.tex:94` |
| `wang2026d2stx` | `sections/1_introduction.tex:30`; `sections/4_experiments.tex:95` |
| `welch1967fft` | `sections/1_introduction.tex:34` |
| `lomb1976unequal` | `sections/1_introduction.tex:34`; `sections/3_method.tex:88` |
| `griffin1984stft` | `sections/1_introduction.tex:34` |
| `shazeer2017moe` | `sections/1_introduction.tex:34` |
| `luiten2021hota` | `sections/4_experiments.tex:17` |
| `ristani2016idmetrics` | `sections/4_experiments.tex:18` |

