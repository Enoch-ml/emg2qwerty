#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import optuna

VAL_CER_RE = re.compile(r"'val/CER':\s*([0-9]+(?:\.[0-9]+)?)")
TEST_CER_RE = re.compile(r"'test/CER':\s*([0-9]+(?:\.[0-9]+)?)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bayesian optimization for the conformer CTC model."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.home() / "emg2qwerty")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--study-name", type=str, default="conformer_ctc_bayes_v3")
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URI, e.g. sqlite:///optuna_conformer.db",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=80,
        help="Use fewer epochs for search; retrain best config later.",
    )
    parser.add_argument("--timeout-hours", type=float, default=None)
    parser.add_argument("--seed", type=int, default=1501)
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--hydra-model-arg",
        type=str,
        default="model=rotation_invariant_conformer_ctc",
        help="Hydra arg that selects the conformer model config.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override(s), repeatable. Example: --extra-override batch_size=16",
    )
    parser.add_argument(
        "--trial-log-dir",
        type=Path,
        default=None,
        help="Directory for per-trial stdout logs. Default: <repo-root>/run_logs/optuna_trials",
    )
    return parser.parse_args()


def extract_metric(stdout: str, regex: re.Pattern[str], name: str) -> float:
    matches = regex.findall(stdout)
    if not matches:
        raise RuntimeError(f"Could not parse {name} from training output.")
    return float(matches[-1])


def suggest_overrides(trial: optuna.Trial, max_epochs: int, seed: int) -> list[str]:
    # Highest-impact knobs first: optimization + core encoder capacity + effective sequence length.
    lr = trial.suggest_float("optimizer.lr", 2e-4, 2e-3, log=True)
    scheduler = trial.suggest_categorical(
        "lr_scheduler",
        [
            "linear_warmup_cosine_annealing",
            "cosine_annealing",
            "reduce_on_plateau",
        ],
    )
    d_model = trial.suggest_categorical("module.d_model", [192, 256, 320])
    num_layers = trial.suggest_categorical("module.num_layers", [6, 8, 10])
    dropout = trial.suggest_float("module.dropout", 0.05, 0.25)
    window_length = trial.suggest_categorical("datamodule.window_length", [4000, 6000, 8000])
    conv_kernel = trial.suggest_categorical("module.conv_kernel_size", [15, 31])

    # IMPORTANT: fixed categorical space across all trials.
    # All current d_model choices are divisible by 8, so both 4 and 8 heads are valid.
    num_heads = trial.suggest_categorical("module.num_heads", [4, 8])

    # Mild regularization / frontend capacity levers.
    mlp_width = trial.suggest_categorical("module.mlp_width", [96, 128])
    time_mask = trial.suggest_categorical("specaug.time_mask_param", [20, 25, 30])
    n_time_masks = trial.suggest_categorical("specaug.n_time_masks", [2, 3, 4])

    overrides = [
        f"seed={seed}",
        f"trainer.max_epochs={max_epochs}",
        f"optimizer.lr={lr}",
        f"lr_scheduler={scheduler}",
        f"module.d_model={d_model}",
        f"module.num_layers={num_layers}",
        f"module.num_heads={num_heads}",
        f"module.dropout={dropout}",
        f"datamodule.window_length={window_length}",
        f"module.conv_kernel_size={conv_kernel}",
        f"module.mlp_features=[{mlp_width},{mlp_width}]",
        f"specaug.time_mask_param={time_mask}",
        f"specaug.n_time_masks={n_time_masks}",
        # Keep decoder fixed during model search. Tune beam search after selecting encoder.
        "decoder=ctc_greedy",
        # Safer checkpoint behavior during sweeps.
        "callbacks.1.save_last=false",
    ]

    if scheduler == "linear_warmup_cosine_annealing":
        warmup_epochs = trial.suggest_categorical(
            "lr_scheduler.scheduler.warmup_epochs", [5, 10, 15]
        )
        overrides.append(f"lr_scheduler.scheduler.warmup_epochs={warmup_epochs}")

    if scheduler == "reduce_on_plateau":
        patience = trial.suggest_categorical(
            "lr_scheduler.scheduler.patience", [5, 10, 15]
        )
        factor = trial.suggest_categorical(
            "lr_scheduler.scheduler.factor", [0.1, 0.3, 0.5]
        )
        overrides.extend(
            [
                f"lr_scheduler.scheduler.patience={patience}",
                f"lr_scheduler.scheduler.factor={factor}",
            ]
        )

    return overrides


def build_command(args: argparse.Namespace, trial: optuna.Trial) -> tuple[list[str], Path]:
    trial_log_dir = args.trial_log_dir or (args.repo_root / "run_logs" / "optuna_trials")
    trial_log_dir.mkdir(parents=True, exist_ok=True)

    log_path = trial_log_dir / (
        f"trial_{trial.number:03d}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
    )

    cmd = [args.python, "-m", "emg2qwerty.train"]
    if args.hydra_model_arg:
        cmd.append(args.hydra_model_arg)
    cmd.extend(suggest_overrides(trial, args.max_epochs, args.seed))
    cmd.extend(args.extra_override)

    return cmd, log_path


def objective_factory(args: argparse.Namespace):
    def objective(trial: optuna.Trial) -> float:
        cmd, log_path = build_command(args, trial)
        trial.set_user_attr("command", " ".join(cmd))
        trial.set_user_attr("log_path", str(log_path))

        collected: list[str] = []

        with log_path.open("w") as f:
            proc = subprocess.Popen(
                cmd,
                cwd=args.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None

            for line in proc.stdout:
                f.write(line)
                f.flush()
                collected.append(line)

            proc.wait()

        stdout = "".join(collected)

        if proc.returncode != 0:
            trial.set_user_attr("status", "failed")
            trial.set_user_attr("returncode", proc.returncode)

            if "OutOfMemoryError" in stdout or "CUDA out of memory" in stdout:
                trial.set_user_attr("failure_reason", "oom")
                raise optuna.TrialPruned("OOM")

            raise RuntimeError(f"Training failed for trial {trial.number}; see {log_path}")

        val_cer = extract_metric(stdout, VAL_CER_RE, "val/CER")

        try:
            test_cer = extract_metric(stdout, TEST_CER_RE, "test/CER")
        except Exception:
            test_cer = math.nan

        trial.set_user_attr("status", "ok")
        trial.set_user_attr("val_CER", val_cer)
        trial.set_user_attr("test_CER", test_cer)

        return val_cer

    return objective


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.expanduser().resolve()

    if args.trial_log_dir is not None:
        args.trial_log_dir = args.trial_log_dir.expanduser().resolve()

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    timeout = None if args.timeout_hours is None else int(args.timeout_hours * 3600)

    study.optimize(
        objective_factory(args),
        n_trials=args.n_trials,
        timeout=timeout,
        gc_after_trial=True,
        catch=(RuntimeError, ValueError),
    )

    best = study.best_trial
    summary = {
        "study_name": study.study_name,
        "best_value": best.value,
        "best_params": best.params,
        "best_log_path": best.user_attrs.get("log_path"),
        "best_command": best.user_attrs.get("command"),
        "n_trials": len(study.trials),
    }

    print(json.dumps(summary, indent=2))

    out_dir = args.repo_root / "run_logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.study_name}_best.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()