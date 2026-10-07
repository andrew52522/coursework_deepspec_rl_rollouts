# DeepSpec RL Rollouts

Coursework project on **speculative decoding in the rollout phase of RL post-training**.

The project studies whether speculative decoding methods that are effective for inference remain useful under RL conditions, where the verifier/policy changes over time and rollout batches can be much larger than typical serving batches.

Main stack:

- **verl** — RL orchestration;
- **GRPO** — policy optimization;
- **FSDP2** — training backend;
- **vLLM** — rollout generation;
- **Qwen** — target/verifier model family;
- **GSM8K** — initial rule-based task;
- **prefix / ngram, EAGLE-3, DFlash, DSpark** — speculative decoding methods.

---

# Implementation roadmap

## Stage 0 — software and GPU stack

- [x] Native Ubuntu GPU environment
- [x] NVIDIA driver + Docker + NVIDIA Container Toolkit
- [x] CUDA 13 container
- [x] PyTorch CUDA/BF16
- [x] vLLM V2
- [x] pinned `verl` environment

The initial WSL2 implementation exposed WSL-specific CUDA/UVA/IPC problems, therefore the validated reference environment was moved to native Linux.

---

## Stage 1 — minimal GRPO correctness

Reference setup:

```text
1 × NVIDIA RTX 4070 Laptop GPU
8 GB VRAM

Qwen/Qwen3-0.6B
GSM8K
GRPO
FSDP2
vLLM V2
BF16
```

Validated path:

- [x] policy → vLLM rollout generation
- [x] GSM8K rule reward
- [x] old-policy log probabilities
- [x] reference-policy log probabilities
- [x] GRPO advantages
- [x] actor forward / backward
- [x] Adam optimizer update
- [x] policy weight update
- [x] actor → vLLM weight synchronization

The largest validated baseline contains:

```text
64 trajectories
1 full RL iteration
```

---

## Stage 2 — repository reproducibility

- [x] pin `verl` commit and Python stack
- [x] containerize CUDA userspace
- [x] automatically prepare `Qwen3-0.6B`
- [x] automatically prepare GSM8K
- [x] CUDA smoke test
- [x] vLLM smoke test
- [x] one-command baseline launcher
- [x] automatic GRPO log summary
- [x] reproduce the complete 64-trajectory run from the repository

Current entry points:

```bash
./scripts/setup.sh
./scripts/run_baseline.sh
```

The latest clean reproduction finished with process exit code `0` and completed the full rollout → reward → GRPO update → weight synchronization pipeline.

---

## Stage 3 — get access to experiment hardware

- [ ] get access to a server with **2 × RTX 5090 32 GB** or **2 × A100 80 GB**
- [ ] reproduce the current Qwen3-0.6B baseline on the new server
- [ ] verify both GPUs are visible inside the same job/node
- [ ] inspect GPU topology
- [ ] run CUDA P2P smoke
- [ ] run NCCL collective smoke
- [ ] validate FSDP2 with world size 2

Only after the two-GPU environment is validated should the experiment move to Qwen3-4B.

---

## Stage 4 — Qwen3-4B speculative-decoding experiment

Qwen3-4B is the main controlled experiment because compatible pretrained draft heads are available for the same target model:

```text
Qwen3-4B
├── EAGLE-3
├── DFlash
└── DSpark
```

Planned experiment:

- [ ] run vanilla Qwen3-4B inference and GRPO baseline
- [ ] validate `prefix / ngram` baseline
- [ ] validate EAGLE-3
- [ ] validate DFlash
- [ ] validate DSpark
- [ ] compare acceptance, generation time, throughput and VRAM
- [ ] sweep several rollout batch sizes
- [ ] save several policy versions `θ0 → θ1 → θ2`
- [ ] measure frozen-drafter acceptance as the verifier changes
- [ ] integrate speculative decoding into the `verl` rollout path
- [ ] verify speculative generation after actor → vLLM weight synchronization

The first experiment keeps the drafter frozen:

```text
θk      current GRPO policy / verifier
θref    frozen reference policy
φ       speculative drafter

θk changes during RL
φ remains frozen
```

The first question is therefore not how to refresh the drafter, but whether measurable drafter staleness appears at all.

---

## Stage 5 — medium-size Qwen3.5+ experiment

Qwen3-4B is a controlled DeepSpec stage, not the final model-family endpoint.

After the Qwen3 experiment:

- [ ] move to a medium-size **Qwen3.5+** target
- [ ] target roughly the **9B+ class**, depending on available server memory and software support
- [ ] reproduce the baseline measurements
- [ ] test speculative mechanisms actually supported by that architecture
- [ ] compare the conclusions with the controlled Qwen3-4B experiment

The same three pretrained Qwen3-4B draft heads are **not assumed to transfer to Qwen3.5+**.

This stage is intended to separate:

```text
speculative-decoding algorithm behaviour
from
availability of a compatible trained drafter
```

---

## Stage 6 — optional extensions

- [ ] Qwen3-8B controlled experiment
- [ ] longer GRPO training
- [ ] more saved policy versions
- [ ] MATH / Math500 evaluation
- [ ] domain adaptation of existing draft heads
- [ ] periodic drafter refresh
- [ ] online drafter training

Training draft heads from scratch is not required for the initial coursework result.

---

## Stage 7 — final analysis

Primary measurements:

- [ ] acceptance rate
- [ ] mean accepted draft length
- [ ] rollout wall-clock time
- [ ] output throughput
- [ ] GPU memory usage
- [ ] batch-size dependence
- [ ] full RL iteration time
- [ ] weight synchronization time
- [ ] reward / output quality
- [ ] drafter acceptance vs policy version

Main comparisons:

```text
throughput(method, batch_size)

latency(method, batch_size)

acceptance(method, batch_size)

acceptance(method, policy_version)

VRAM(method, batch_size)

rollout_time / full_RL_iteration_time
```

A useful interpretation separates draft quality from speculation overhead:

```text
α = reduction in verifier decoding steps
μ = additional speculative-decoding cost
```

The purpose is to avoid reporting a single serving-style speedup number that may not represent RL rollout workloads.

---

# Coursework completion target

Minimum defensible result:

```text
[x] reproducible vanilla GRPO stack
        ↓
[ ] Qwen3-4B
        ↓
[ ] vanilla + prefix/ngram + EAGLE-3 + DFlash + DSpark
        ↓
[ ] controlled batch sweep
        ↓
[ ] acceptance + latency + throughput + memory measurements
        ↓
COURSEWORK RESULT
```

RL-specific extension:

```text
CONTROLLED SPECULATIVE RESULT
        ↓
[ ] θ0 / θ1 / θ2 policy versions
        ↓
[ ] drafter-staleness measurements
        ↓
[ ] speculative rollout inside verl
        ↓
EXTENDED RL RESULT
```

Generalization stage:

```text
Qwen3-4B CONTROLLED RESULT
        ↓
[ ] medium-size Qwen3.5+ target
        ↓
[ ] supported speculative mechanisms
        ↓
GENERALIZATION RESULT
```

---

# Reproducing the current baseline

The repository contains a small validated experiment that can be reproduced before using the final two-GPU research server.

## Host requirements

The host should already provide:

```text
Linux
NVIDIA driver
Docker
NVIDIA Container Toolkit
Git
Python 3
Internet access during setup
```

Clone the repository and run:

```bash
git clone https://github.com/andrew52522/coursework_deepspec_rl_rollouts.git
cd coursework_deepspec_rl_rollouts

./scripts/setup.sh
./scripts/run_baseline.sh
```

`scripts/setup.sh`:

```text
→ clones the pinned verl source
→ builds the CUDA container
→ materializes the pinned environment
→ downloads Qwen3-0.6B
→ prepares GSM8K
→ runs CUDA/vLLM smoke tests
```

`scripts/run_baseline.sh`:

```text
→ runs the validated 1×4070 GRPO configuration
→ records the raw log
→ records the process exit code
→ summarizes the RL run
```

Pinned `verl` commit:

```text
6093e007cc341973c9d9a6fb3867a85976c7c458
```

Exact environment versions are stored in:

```text
environment/versions.env
```

The reference validated run is documented in:

```text
reports/2026-10-07_1x4070_grpo_baseline.md
```

---

# Current validated baseline

Latest clean reproduction:

```text
model                       Qwen3-0.6B
dataset                     GSM8K
algorithm                   GRPO
rollout                     vLLM V2
training                    FSDP2 / BF16

RL iterations               1
trajectories                64
successful trajectories     4 / 64

reward mean                 0.0625
policy-gradient loss        0.0171233
gradient norm               2.18767

actor max allocated         5.906 GiB
actor max reserved          6.674 GiB

generation                  22.09 s
actor update                9.05 s
weight synchronization      0.29 s
full RL iteration           47.39 s

total tokens                18,685
throughput                  394.26 tokens/s

process exit code           0
```

Exact rewards and timings are stochastic and are not expected to be identical between reproductions.

The acceptance criterion is completion of the entire pipeline with a real non-zero learning signal and successful actor-to-rollout weight synchronization.

---

# Repository layout

```text
.
├── docker/
│   └── Dockerfile.cu130
│
├── environment/
│   └── versions.env
│
├── scripts/
│   ├── setup.sh
│   ├── run_baseline.sh
│   │
│   ├── bootstrap/
│   │   ├── cuda_container_smoke.sh
│   │   └── vllm_smoke.py
│   │
│   ├── grpo/
│   │   ├── run_1x4070_qwen3_0.6b.sh
│   │   └── run_1x4070_qwen3_0.6b_smoke.sh
│   │
│   └── analysis/
│       └── summarize_grpo_log.py
│
├── reports/
│   └── 2026-10-07_1x4070_grpo_baseline.md
│
└── papers/
    └── DeepSpec.pdf
```

Models, datasets, caches, checkpoints and raw logs are intentionally kept outside Git.

---

# Research question

The final question is not simply whether speculative decoding makes inference faster.

The coursework asks:

> **How do speculative-decoding methods behave in RL rollout workloads as batch size grows and the verifier policy changes during training?**

In particular:

```text
Does acceptance degrade after policy updates?

Does higher acceptance actually improve wall-clock rollout time?

At what rollout batch size does speculative decoding stop helping?

Do different draft architectures degrade differently as the verifier changes?

How much rollout speedup survives at the full RL-iteration level?

Does the conclusion change when moving from the controlled Qwen3-4B setup
to a medium-size Qwen3.5+ model?
```
