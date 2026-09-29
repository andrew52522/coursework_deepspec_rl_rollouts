# coursework_deepspec_rl_rollouts

Coursework repository for experiments on **GRPO policy updates and speculative decoding during RL rollouts**.

## Current development path

The project is being built in stages so that the expensive final GPU run is not the first time the software stack is tested.

1. **Environment preflight — 1× RTX 4070 Laptop / WSL2**
   - Windows 10 Pro host
   - Ubuntu 24.04 in WSL2
   - RTX 4070 Laptop 8 GB
   - 64 GB host RAM
   - development over SSH from a Mac
2. **Functional GRPO preflight — Qwen3-0.6B**
   - GSM8K
   - verl
   - FSDP2
   - vLLM
   - full backward / optimizer update / weight sync
3. **Optional distributed preflight — 2× RTX 4070 laptops**
   - Ray multi-node
   - NCCL
   - FSDP2 world size 2
4. **Final experiment — 2× RTX 5090 or 2× H100**
   - Qwen3-4B
   - GRPO
   - vLLM
   - EAGLE-3 / DFlash / DSpark
   - speculative decoding acceptance and runtime measurements

## Read in this order

1. [00 — Prepare a clean WSL2 + RTX 4070 environment](docs/00_wsl4070_environment_preflight.md)
2. [01 — Full GRPO preflight on 1×4070, then optional 2×4070](docs/01_grpo_1x4070_then_2x4070.md)
3. [Operational context for ChatGPT](CHATGPT_CONTEXT.md)

## Reproducibility policy

The coursework implementation pins the external verl checkout rather than copying verl into this repository.

Current pinned verl commit:

`6093e007cc341973c9d9a6fb3867a85976c7c458`

Large model weights, datasets, caches and training checkpoints are intentionally kept out of Git.

## Project status

Current stage: **prepare and validate the clean 1×4070 WSL2 environment before the first GRPO run**.
