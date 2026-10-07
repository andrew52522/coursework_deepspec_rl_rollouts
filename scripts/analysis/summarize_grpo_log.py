#!/usr/bin/env python3

import argparse
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def clean(text: str) -> str:
    return ANSI_RE.sub("", text)


def number(value: str):
    value = value.strip()

    m = re.fullmatch(r"np\.\w+\(([-+0-9.eE]+)\)", value)
    if m:
        value = m.group(1)

    try:
        return float(value)
    except ValueError:
        return None


def parse_last_metrics(text: str):
    matches = list(
        re.finditer(
            r"\bstep:(\d+)\s+-\s+(.*)",
            text,
        )
    )

    if not matches:
        raise RuntimeError("No verl step metrics line found")

    match = matches[-1]
    step_number = int(match.group(1))
    payload = match.group(2)

    metrics = {}

    for chunk in payload.split(" - "):
        if ":" not in chunk:
            continue

        key, value = chunk.split(":", 1)
        parsed = number(value)

        if parsed is not None:
            metrics[key.strip()] = parsed

    return step_number, metrics


def parse_run_config(text: str):
    matches = re.findall(r"GRPO_RUN_CONFIG\s+([^\n]+)", text)

    if not matches:
        return {}

    result = {}

    for token in matches[-1].split():
        if "=" not in token:
            continue

        key, value = token.split("=", 1)

        try:
            result[key] = int(value)
            continue
        except ValueError:
            pass

        try:
            result[key] = float(value)
            continue
        except ValueError:
            pass

        result[key] = value

    return result


def parse_exit_code(text: str):
    m = re.search(
        r"=== DOCKER/GRPO EXIT CODE ===\s*\n\s*(-?\d+)",
        text,
    )

    return int(m.group(1)) if m else None


def get(metrics, key):
    return metrics.get(key)


def fmt(value, digits=4):
    if value is None:
        return "n/a"

    if abs(value) >= 1000:
        return f"{value:,.1f}"

    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def state(condition):
    return "PASS" if condition else "FAIL"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)

    parser.add_argument(
        "--trajectories",
        type=int,
        default=None,
        help="Override trajectory count for old logs without GRPO_RUN_CONFIG.",
    )

    parser.add_argument(
        "--gpu-capacity-gib",
        type=float,
        default=None,
        help="Optional usable GPU capacity for headroom calculation.",
    )

    args = parser.parse_args()

    text = clean(args.log.read_text(errors="replace"))

    logged_step, metrics = parse_last_metrics(text)
    config = parse_run_config(text)
    exit_code = parse_exit_code(text)

    rl_iteration = int(
        get(metrics, "training/global_step")
        or logged_step
    )

    train_batch_size = config.get("train_batch_size")
    rollout_n = config.get("rollout_n")

    trajectories = args.trajectories

    if (
        trajectories is None
        and train_batch_size is not None
        and rollout_n is not None
    ):
        trajectories = train_batch_size * rollout_n

    ppo_epochs = config.get("ppo_epochs")
    ppo_mini_batch_size = config.get("ppo_mini_batch_size")

    expected_optimizer_updates = None

    if (
        train_batch_size is not None
        and ppo_mini_batch_size
        and ppo_epochs is not None
        and train_batch_size % ppo_mini_batch_size == 0
    ):
        expected_optimizer_updates = (
            train_batch_size // ppo_mini_batch_size
        ) * ppo_epochs

    gen_time = get(metrics, "timing_s/gen")
    old_logprob_time = get(metrics, "timing_s/old_log_prob")
    ref_time = get(metrics, "timing_s/ref")
    adv_time = get(metrics, "timing_s/adv")
    actor_time = get(metrics, "timing_s/update_actor")
    sync_time = get(metrics, "timing_s/update_weights")
    step_time = get(metrics, "timing_s/step")

    fatal_cuda_error = bool(
        re.search(
            r"torch\.OutOfMemoryError|CUDA out of memory|"
            r"invalid resource handle|device not ready",
            text,
            re.IGNORECASE,
        )
    )

    pipeline_complete = all(
        value is not None and value >= 0
        for value in (
            gen_time,
            old_logprob_time,
            ref_time,
            adv_time,
            actor_time,
            sync_time,
        )
    )

    run_complete = (
        exit_code == 0
        and pipeline_complete
        and not fatal_cuda_error
    )

    score_min = get(metrics, "critic/score/min")
    score_mean = get(metrics, "critic/score/mean")
    score_max = get(metrics, "critic/score/max")

    adv_min = get(metrics, "critic/advantages/min")
    adv_mean = get(metrics, "critic/advantages/mean")
    adv_max = get(metrics, "critic/advantages/max")

    pg_loss = get(metrics, "actor/pg_loss")
    kl_loss = get(metrics, "actor/kl_loss")
    total_loss = get(metrics, "actor/loss")
    grad_norm = get(metrics, "actor/grad_norm")
    lr = get(metrics, "actor/lr")

    successful_trajectories = None

    # Valid for the GSM8K binary rule reward used in these experiments.
    if (
        trajectories is not None
        and score_mean is not None
        and score_min is not None
        and score_max is not None
        and 0 <= score_min <= score_max <= 1
    ):
        estimated = score_mean * trajectories

        if abs(estimated - round(estimated)) < 1e-6:
            successful_trajectories = int(round(estimated))

    max_alloc = get(metrics, "actor/perf/max_memory_allocated_gb")
    max_reserved = get(metrics, "actor/perf/max_memory_reserved_gb")
    cpu_memory = get(metrics, "actor/perf/cpu_memory_used_gb")

    response_mean = get(metrics, "response_length/mean")
    response_min = get(metrics, "response_length/min")
    response_max = get(metrics, "response_length/max")
    response_clip = get(metrics, "response_length/clip_ratio")

    prompt_mean = get(metrics, "prompt_length/mean")
    prompt_min = get(metrics, "prompt_length/min")
    prompt_max = get(metrics, "prompt_length/max")
    prompt_clip = get(metrics, "prompt_length/clip_ratio")

    total_tokens = get(metrics, "perf/total_num_tokens")
    throughput = get(metrics, "perf/throughput")

    reward_signal = (
        score_min is not None
        and score_max is not None
        and score_max > score_min
    )

    nonzero_advantage = (
        adv_min is not None
        and adv_max is not None
        and (abs(adv_min) > 1e-12 or abs(adv_max) > 1e-12)
    )

    nonzero_pg = (
        pg_loss is not None
        and abs(pg_loss) > 1e-12
    )

    print("=" * 72)
    print("GRPO RUN SUMMARY")
    print("=" * 72)

    print(f"log                          {args.log}")
    print(f"run_status                   {state(run_complete)}")
    print(f"process_exit_code            {exit_code if exit_code is not None else 'n/a'}")
    print(f"rl_iteration                 {rl_iteration}")

    if trajectories is not None:
        print(f"trajectories                 {trajectories}")

    if expected_optimizer_updates is not None:
        print(
            f"configured_optimizer_updates  {expected_optimizer_updates}"
        )

    print()
    print("PIPELINE")
    print("-" * 72)
    print(f"rollout_generation           {state(gen_time is not None)}")
    print(f"old_log_prob                 {state(old_logprob_time is not None)}")
    print(f"reference_log_prob           {state(ref_time is not None)}")
    print(f"advantage_computation        {state(adv_time is not None)}")
    print(f"actor_update                 {state(actor_time is not None)}")
    print(f"rollout_weight_sync          {state(sync_time is not None)}")
    print(f"fatal_cuda_error             {'YES' if fatal_cuda_error else 'NO'}")

    print()
    print("LEARNING SIGNAL")
    print("-" * 72)

    if successful_trajectories is not None:
        success_rate = successful_trajectories / trajectories
        print(
            f"successful_trajectories      "
            f"{successful_trajectories}/{trajectories} "
            f"({100 * success_rate:.2f}%)"
        )

    print(
        f"reward_min/mean/max           "
        f"{fmt(score_min)} / {fmt(score_mean)} / {fmt(score_max)}"
    )
    print(f"reward_variation              {'YES' if reward_signal else 'NO'}")

    print(
        f"advantage_min/mean/max        "
        f"{fmt(adv_min)} / {fmt(adv_mean)} / {fmt(adv_max)}"
    )
    print(
        f"nonzero_advantage             "
        f"{'YES' if nonzero_advantage else 'NO'}"
    )

    print(f"policy_gradient_loss          {fmt(pg_loss, 8)}")
    print(f"kl_loss                       {fmt(kl_loss, 8)}")
    print(f"actor_loss                    {fmt(total_loss, 8)}")
    print(f"gradient_norm                 {fmt(grad_norm, 6)}")
    print(f"learning_rate                 {fmt(lr, 10)}")
    print(f"nonzero_policy_gradient       {'YES' if nonzero_pg else 'NO'}")

    print()
    print("SEQUENCE LENGTHS")
    print("-" * 72)
    print(
        f"prompt_length_min/mean/max    "
        f"{fmt(prompt_min, 1)} / {fmt(prompt_mean, 1)} / {fmt(prompt_max, 1)}"
    )
    print(f"prompt_clip_ratio             {fmt(prompt_clip, 4)}")

    print(
        f"response_length_min/mean/max  "
        f"{fmt(response_min, 1)} / {fmt(response_mean, 1)} / {fmt(response_max, 1)}"
    )
    print(f"response_clip_ratio           {fmt(response_clip, 4)}")

    print()
    print("MEMORY")
    print("-" * 72)
    print(f"actor_max_allocated_gib       {fmt(max_alloc, 3)}")
    print(f"actor_max_reserved_gib        {fmt(max_reserved, 3)}")
    print(f"cpu_memory_used_gib           {fmt(cpu_memory, 3)}")

    if (
        args.gpu_capacity_gib is not None
        and max_reserved is not None
    ):
        headroom = args.gpu_capacity_gib - max_reserved
        utilization = max_reserved / args.gpu_capacity_gib

        print(f"gpu_capacity_gib              {args.gpu_capacity_gib:.3f}")
        print(f"reserved_headroom_gib         {headroom:.3f}")
        print(f"reserved_fraction             {utilization:.3f}")

    print()
    print("TIMING")
    print("-" * 72)

    timing_rows = (
        ("generation_s", gen_time),
        ("old_log_prob_s", old_logprob_time),
        ("reference_s", ref_time),
        ("advantage_s", adv_time),
        ("actor_update_s", actor_time),
        ("weight_sync_s", sync_time),
        ("rl_iteration_s", step_time),
    )

    for name, value in timing_rows:
        if value is not None:
            print(f"{name:29s}{value:.4f}")

    print(f"total_tokens                  {fmt(total_tokens, 0)}")
    print(f"throughput_tokens_per_s       {fmt(throughput, 2)}")

    print("=" * 72)


if __name__ == "__main__":
    main()
