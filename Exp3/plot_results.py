import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

SCENARIOS = ["Anxiety", "Despair", "Irritation", "Rage"]


def _find_latest_mode_dir(prefix: str) -> str:
    if not os.path.isdir(RESULTS_DIR):
        return ""
    candidates = [
        d for d in os.listdir(RESULTS_DIR)
        if d.startswith(prefix + "_") and os.path.isdir(os.path.join(RESULTS_DIR, d))
    ]
    if not candidates:
        return ""
    candidates.sort(key=lambda d: os.path.getmtime(os.path.join(RESULTS_DIR, d)))
    return os.path.join(RESULTS_DIR, candidates[-1])


def _load_model_result(path: str) -> Dict[str, Dict[str, float]]:
    """Return mapping: scenario -> {feature -> value} from model_result.csv."""
    out: Dict[str, Dict[str, float]] = {}
    if not path or not os.path.isfile(path):
        return out
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emotion = (row.get("Emotion") or "").strip()
            if not emotion:
                continue
            try:
                out[emotion] = {
                    "Suddenness": float(row.get("Suddenness", 0.0) or 0.0),
                    "Goal_relevance": float(row.get("Goal_relevance", 0.0) or 0.0),
                    "Conduciveness": float(row.get("Conduciveness", 0.0) or 0.0),
                    "Power": float(row.get("Power", 0.0) or 0.0),
                }
            except ValueError:
                continue
    return out


def _infer_scenario_from_log_name(fname: str) -> str:
    name = fname.lower()
    if "anx" in name:
        return "Anxiety"
    if "despair" in name:
        return "Despair"
    if "irrit" in name:
        return "Irritation"
    if "rage" in name:
        return "Rage"
    return ""


def _compute_average_rewards_from_logs(mode_dir: str) -> Dict[str, float]:
    """Compute average per-step reward per scenario from step logs.

    Looks under <mode_dir>/logs/*.csv and averages the 'reward' column per
    file, mapping each file to a scenario based on its filename. This avoids
    exploding scales when many training steps are logged.
    """
    rewards: Dict[str, float] = {}
    if not mode_dir:
        return rewards
    logs_dir = os.path.join(mode_dir, "logs")
    if not os.path.isdir(logs_dir):
        return rewards

    for fname in os.listdir(logs_dir):
        if not fname.lower().endswith(".csv"):
            continue
        scenario = _infer_scenario_from_log_name(fname)
        if not scenario:
            continue
        path = os.path.join(logs_dir, fname)
        total = 0.0
        count = 0
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        total += float(row.get("reward", 0.0) or 0.0)
                        count += 1
                    except ValueError:
                        continue
        except OSError:
            continue
        if count == 0:
            avg = 0.0
        else:
            avg = total / float(count)
        # If multiple logs exist for the same scenario, keep the last one seen.
        rewards[scenario] = avg
    return rewards


def plot_reward_comparison(baseline_dir: str, shaped_dir: str) -> Dict[str, Tuple[float, float]]:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    baseline_rewards = _compute_average_rewards_from_logs(baseline_dir)
    shaped_rewards = _compute_average_rewards_from_logs(shaped_dir)

    if not baseline_rewards or not shaped_rewards:
        print("[Reward] Skipping reward_comparison.png (no step logs found for one or both modes).")
        return {}

    x = list(range(len(SCENARIOS)))
    width = 0.35
    base_vals: List[float] = []
    shaped_vals: List[float] = []
    summary: Dict[str, Tuple[float, float]] = {}

    for s in SCENARIOS:
        b = baseline_rewards.get(s)
        e = shaped_rewards.get(s)
        if b is None or e is None:
            base_vals.append(0.0)
            shaped_vals.append(0.0)
            summary[s] = (float("nan"), float("nan"))
        else:
            base_vals.append(b)
            shaped_vals.append(e)
            summary[s] = (b, e)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([i - width / 2 for i in x], base_vals, width, label="Baseline")
    ax.bar([i + width / 2 for i in x], shaped_vals, width, label="Emotion-shaped")
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS)
    ax.set_ylabel("Average per-step reward")
    ax.set_title("Average per-step reward per scenario")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "reward_comparison.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Reward] Saved {out_path}")

    return summary


def plot_appraisal_comparison(baseline_app: Dict[str, Dict[str, float]],
                              shaped_app: Dict[str, Dict[str, float]]) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if not baseline_app or not shaped_app:
        print("[Appraisal] Skipping appraisal_comparison.png (missing model_result.csv).")
        return

    features = ["Suddenness", "Goal_relevance", "Conduciveness", "Power"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
    axes = axes.flatten()

    x = list(range(len(SCENARIOS)))
    width = 0.35

    for idx, feat in enumerate(features):
        ax = axes[idx]
        base_vals = []
        shaped_vals = []
        for s in SCENARIOS:
            b = baseline_app.get(s, {}).get(feat, 0.0)
            e = shaped_app.get(s, {}).get(feat, 0.0)
            # Ensure missing or NaN values are rendered as 0 for plotting.
            try:
                b_val = float(b)
            except (TypeError, ValueError):
                b_val = 0.0
            try:
                e_val = float(e)
            except (TypeError, ValueError):
                e_val = 0.0
            if b_val != b_val:  # NaN check
                b_val = 0.0
            if e_val != e_val:
                e_val = 0.0
            base_vals.append(b_val)
            shaped_vals.append(e_val)
        ax.bar([i - width / 2 for i in x], base_vals, width, label="Baseline")
        ax.bar([i + width / 2 for i in x], shaped_vals, width, label="Emotion-shaped")
        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIOS, rotation=15)
        ax.set_title(feat)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Appraisal comparison: baseline vs emotion-shaped")
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = os.path.join(PLOTS_DIR, "appraisal_comparison.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Appraisal] Saved {out_path}")


def plot_td_error_comparison(baseline_dir: str, shaped_dir: str) -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    def load_tde_for_scenario(mode_dir: str, scenario: str) -> List[float]:
        if not mode_dir:
            return []
        logs_dir = os.path.join(mode_dir, "logs")
        if not os.path.isdir(logs_dir):
            return []
        target = None
        for fname in os.listdir(logs_dir):
            if not fname.lower().endswith(".csv"):
                continue
            if _infer_scenario_from_log_name(fname) == scenario:
                target = os.path.join(logs_dir, fname)
        if target is None:
            return []
        tde: List[float] = []
        try:
            with open(target, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        tde.append(float(row.get("td_error", 0.0) or 0.0))
                    except ValueError:
                        continue
        except OSError:
            return []
        return tde

    # Use Anxiety as the representative scenario for TD-error comparison.
    scenario = "Anxiety"
    base_tde = load_tde_for_scenario(baseline_dir, scenario)
    shaped_tde = load_tde_for_scenario(shaped_dir, scenario)

    if not base_tde or not shaped_tde:
        print("[TD-Error] Skipping td_error_comparison.png (missing logs for Anxiety).")
        return

    # Option 1: smooth TD-error with a moving average window.
    def smooth(series: List[float], window: int = 100) -> np.ndarray:
        arr = np.asarray(series, dtype=float)
        if arr.size < window:
            return arr
        kernel = np.ones(window, dtype=float) / float(window)
        return np.convolve(arr, kernel, mode="valid")

    # Optionally restrict to first N steps to keep the plot readable.
    max_steps = 1000
    base_trimmed = base_tde[:max_steps]
    shaped_trimmed = shaped_tde[:max_steps]

    base_smooth = smooth(base_trimmed)
    shaped_smooth = smooth(shaped_trimmed)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(base_smooth) + 1), base_smooth, label="Baseline")
    ax.plot(range(1, len(shaped_smooth) + 1), shaped_smooth, label="Emotion-shaped")
    ax.set_xlabel("Step (smoothed)")
    ax.set_ylabel("TD error (moving average)")
    ax.set_title("Smoothed TD-error over steps (Anxiety scenario)")
    ax.legend()
    fig.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "td_error_comparison.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[TD-Error] Saved {out_path}")


def main() -> None:
    os.makedirs(PLOTS_DIR, exist_ok=True)

    baseline_dir = _find_latest_mode_dir("baseline")
    shaped_dir = _find_latest_mode_dir("emotion_shaped")

    if not baseline_dir or not shaped_dir:
        print("Could not find both baseline_* and emotion_shaped_* folders in results/.")
        return

    print(f"Using baseline dir: {baseline_dir}")
    print(f"Using emotion-shaped dir: {shaped_dir}")

    # Load appraisal data
    base_model = _load_model_result(os.path.join(baseline_dir, "model_result.csv"))
    shaped_model = _load_model_result(os.path.join(shaped_dir, "model_result.csv"))

    # 1) Reward comparison
    reward_summary = plot_reward_comparison(baseline_dir, shaped_dir)

    # 2) Appraisal comparison
    plot_appraisal_comparison(base_model, shaped_model)

    # 3) TD-error comparison (if logs exist)
    plot_td_error_comparison(baseline_dir, shaped_dir)

    # Print a brief textual summary
    print("\n=== Summary of reward changes (emotion-shaped - baseline) ===")
    for s in SCENARIOS:
        vals = reward_summary.get(s)
        if not vals or any(not isinstance(v, (int, float)) or v != v for v in vals):  # NaN check
            print(f"{s}: N/A (missing logs)")
            continue
        b, e = vals
        diff = e - b
        print(f"{s}: baseline={b:.3f}, shaped={e:.3f}, delta={diff:.3f}")


if __name__ == "__main__":
    main()
