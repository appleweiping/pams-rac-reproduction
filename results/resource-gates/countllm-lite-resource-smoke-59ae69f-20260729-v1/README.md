# CountLLM-Lite resource gate smoke

**Status:** `smoke_only`

This designated-server artifact records a resource decision and one small
bridge autograd smoke. It is not a CountLLM or CountLLM-Lite baseline result,
does not use Vicuna-7B, and is not eligible for a paper-comparison table.

The isolated container exposed one NVIDIA RTX A6000 with
`47.3182373046875 GiB` (`50,807,570,432` bytes). That is below the frozen
`48 GiB` training gate, so the full reduced-resource recipe was not started.
The only CUDA computation was a `70.7 MB` decimal (`67.4 MiB`) peak-allocation
forward/backward through the small query bridge:

- input `[1, 32, 64]`;
- two 128-dimensional Transformer layers and 16 learned queries;
- output `[1, 16, 256]`;
- finite input/query gradients;
- `2.2722` seconds for the smoke.

No Vicuna model, 4-bit weights, LoRA training, video encoder, UCFRep sample,
development label, or test input was loaded. WebVid stage 1 and UCFRep stages
2/3 were not run. Consequently this evidence says only that the bridge
forward/backward works and the designated GPU fails the preregistered memory
gate.

The server source revision was
`59ae69f24fa75b8d608633c75b8cda5aab4700ed`. The original retained server
path is:

`/media/lenovo/data2/pams-rac/runs/baselines/countllm-lite-resource-smoke-59ae69f-20260729-v1`

Integrity:

| File | SHA-256 |
|---|---|
| [`smoke.json`](smoke.json) | `06b21d74f234721bfe6b93da36e834095eb562a3c87b341d6851105591219d98` |
| [`smoke.receipt.json`](smoke.receipt.json) | `38da680e9aaeea1f5297a86393706dc69f43b81dab4bf387af564cd72ee42628` |

The receipt binds `smoke.json` to the same artifact SHA, source revision, and
`sealed_test_touched=false` declaration.
