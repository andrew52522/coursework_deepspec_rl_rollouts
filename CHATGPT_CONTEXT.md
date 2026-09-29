# ChatGPT context for this coursework

Use this file as the **source of truth for operational context** before giving setup or code instructions.

## Project

Repository: `andrew52522/coursework_deepspec_rl_rollouts`

Goal: coursework on live GRPO policy updates and speculative decoding behavior. The final experiment is intended for 2× RTX 5090 or 2× H100 with Qwen3-4B, verl/FSDP2, vLLM, GSM8K and frozen draft heads.

The current laptop is only a **preflight/development machine**, not the source of final performance results.

## Current machine

- Windows 10 Pro host.
- Ubuntu 24.04 under **WSL2**.
- NVIDIA RTX 4070 Laptop GPU, 8 GB VRAM.
- 64 GB physical RAM on the Windows host.
- User works remotely from a Mac over the already-configured local-network SSH path into the WSL Ubuntu terminal.
- Project and Linux build files must live in WSL Linux filesystem under `/home/...`, not under `/mnt/c`.

## GPU/WSL rule

The NVIDIA display driver belongs to **Windows**.

Do **not** install a Linux NVIDIA display driver inside WSL.

Do not recommend:
- `sudo apt install nvidia-driver-*`
- `sudo ubuntu-drivers autoinstall`
- `cuda-drivers`
- the generic `cuda` meta-package

unless there is a separately verified WSL-specific reason.

If a CUDA toolkit is ever needed for compilation, install a WSL-compatible toolkit-only configuration without replacing the Windows-provided WSL driver interface.

## Dependency policy

Pinned verl commit for the coursework:

`6093e007cc341973c9d9a6fb3867a85976c7c458`

verl must be cloned separately from the coursework repo.

Use the pinned commit's:
- `pyproject.toml`
- `uv.lock`
- example scripts
- source code

as the authority for exact dependency versions/config keys.

Do not blindly apply the current/latest verl documentation to this pinned commit.

Do **not** fix environment problems with random:
- `pip install -U torch`
- `pip install -U vllm`
- `pip install flash-attn`
- `pip install -U transformers`

The intended environment is materialized with `uv --frozen`.

## One-GPU GRPO preflight target

Use:
- model: `Qwen/Qwen3-0.6B`
- dataset: GSM8K
- speculative decoding: OFF
- `SPEC_METHOD=none`
- one visible GPU
- vLLM tensor parallel size 1
- FSDP2 actor
- CPU offload only when justified by measured 8-GB VRAM pressure
- very small micro-batches/sequence lengths initially

The goal is correctness:
1. rollout;
2. reward;
3. group-relative advantage;
4. policy loss;
5. backward;
6. non-zero gradient on at least one useful batch;
7. optimizer step;
8. changed actor weights;
9. weight sync into rollout engine;
10. next rollout with the updated policy;
11. checkpoint/resume.

Laptop throughput is not a coursework result.

## Two-laptop phase

Only after one laptop works.

Two 4070 laptops are for distributed infrastructure testing:
- Ray multi-node;
- NCCL;
- FSDP2 world_size=2;
- distributed backward;
- checkpoint/resume.

Do not use ordinary LAN performance to infer H100/5090 throughput.

Avoid cross-node vLLM TP=2 for the tiny preflight model; use the network to test distributed training infrastructure.

## Final phase

After laptop preflight:
- switch model to Qwen3-4B;
- use 2×5090 or 2×H100;
- first run SPEC_METHOD=none;
- prove 2-step autoregressive GRPO live loop;
- only then add EAGLE-3/DFlash/DSpark and acceptance metrics.

## How the assistant should work with the user

The user executes commands manually. The assistant proposes commands/code.

For setup/debugging:
1. give one logical step at a time;
2. say whether the command runs in Windows PowerShell or WSL Bash;
3. explain what it changes or verifies;
4. state expected output;
5. wait for the real output before moving to a dependent step;
6. on failure, diagnose from evidence before changing dependencies;
7. change one major variable at a time;
8. preserve working states with Git commits.

Do not hide uncertainty. If exact behavior depends on the pinned verl source, inspect that source instead of guessing.

## Git policy

Commit:
- scripts;
- configs;
- docs;
- tests;
- metrics code;
- small reproducibility manifests;
- curated small evidence.

Do not commit:
- model weights;
- `.venv`;
- package/model caches;
- raw datasets/parquet;
- checkpoints/optimizer states;
- large rollout dumps;
- secrets;
- tokens;
- SSH private keys.

## Networking note

The working Mac → laptop → WSL SSH path already exists. Do not redesign it during ML setup unless there is a concrete networking problem.

Windows 10 WSL2 normally uses NAT networking. Do not assume Windows 11 mirrored-networking features are available.

## Documentation order

1. `docs/00_wsl4070_environment_preflight.md`
2. GRPO 1×4070 preflight plan
3. optional 2×4070 distributed preflight
4. final 2×5090 / 2×H100 experiment
