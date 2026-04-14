import csv
import os
import shutil
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

SCENARIOS = [
    "anxiety",
    "despair",
    "irritation",
    "rage",
]

LAMBDA_VALUES = [0.0, 0.2, 0.5, 1.0]


def _clear_logs_dir() -> None:
    if not os.path.isdir(LOGS_DIR):
        return
    for fname in os.listdir(LOGS_DIR):
        path = os.path.join(LOGS_DIR, fname)
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError:
                pass


def _run_scenarios_with_env(env: Dict[str, str]) -> None:
    for scenario in SCENARIOS:
        script = os.path.join("02_mdp_model", f"{scenario}.py")
        print(f"Running {script} with env: USE_EMOTION_REWARD={env.get('USE_EMOTION_REWARD')} EMOTION_REWARD_LAMBDA={env.get('EMOTION_REWARD_LAMBDA')}")
        subprocess.run(["python", script], cwd=BASE_DIR, check=True, env=env)

    # Run SVM inference
    infer_script = os.path.join("03_model_infer", "01_svm_infer.py")
    print(f"Running {infer_script}...")
    subprocess.run(["python", infer_script], cwd=BASE_DIR, check=True, env=env)


def _snapshot_results(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    # Copy model_result.csv
    model_csv = os.path.join(DATA_DIR, "model_result.csv")
    if os.path.exists(model_csv):
        shutil.copy(model_csv, os.path.join(out_dir, "model_result.csv"))

    # Copy SVM outputs
    if os.path.isdir(DATA_DIR):
        for fname in os.listdir(DATA_DIR):
            if fname.startswith("svm_") and fname.endswith(".csv"):
                src = os.path.join(DATA_DIR, fname)
                dst = os.path.join(out_dir, fname)
                shutil.copy(src, dst)

    # Copy logs into subfolder
    if os.path.isdir(LOGS_DIR):
        logs_out = os.path.join(out_dir, "logs")
        os.makedirs(logs_out, exist_ok=True)
        for fname in os.listdir(LOGS_DIR):
            src = os.path.join(LOGS_DIR, fname)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(logs_out, fname))


def _compute_avg_reward_from_logs(result_dir: str) -> float:
    """Compute average cumulative reward across scenarios using logs in result_dir.

    Expects logs in <result_dir>/logs/*.csv with a 'reward' column.
    """
    logs_dir = os.path.join(result_dir, "logs")
    if not os.path.isdir(logs_dir):
        return float("nan")

    scenario_totals: Dict[str, float] = {}
    scenario_counts: Dict[str, int] = {}

    def infer_scenario(fname: str) -> str:
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

    for fname in os.listdir(logs_dir):
        if not fname.lower().endswith(".csv"):
            continue
        scenario = infer_scenario(fname)
        if not scenario:
            continue
        path = os.path.join(logs_dir, fname)
        total = 0.0
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        total += float(row.get("reward", 0.0) or 0.0)
                    except ValueError:
                        continue
        except OSError:
            continue
        scenario_totals[scenario] = total
        scenario_counts[scenario] = 1

    if not scenario_totals:
        return float("nan")

    avg = sum(scenario_totals.values()) / len(scenario_totals)
    return avg


def run_lambda_experiments() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    lambda_dirs: List[Tuple[float, str]] = []

    for lam in LAMBDA_VALUES:
        _clear_logs_dir()
        env = os.environ.copy()
        env["USE_NEURAL_APPRAISAL"] = "0"
        if lam == 0.0:
            env["USE_EMOTION_REWARD"] = "0"
            env["EMOTION_REWARD_LAMBDA"] = "0.0"
        else:
            env["USE_EMOTION_REWARD"] = "1"
            env["EMOTION_REWARD_LAMBDA"] = str(lam)
        env["LOG_STEPS"] = "1"  # ensure logs for reward computation

        print(f"\n=== Running lambda={lam} ===")
        _run_scenarios_with_env(env)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(RESULTS_DIR, f"lambda_{lam}_{timestamp}")
        _snapshot_results(out_dir)
        lambda_dirs.append((lam, out_dir))

    # Compute average reward per lambda and plot
    lambda_vals: List[float] = []
    avg_rewards: List[float] = []

    for lam, d in lambda_dirs:
        avg = _compute_avg_reward_from_logs(d)
        lambda_vals.append(lam)
        avg_rewards.append(avg)
        if avg == avg:  # not NaN
            print(f"lambda={lam}: average cumulative reward across scenarios = {avg:.3f}")
        else:
            print(f"lambda={lam}: average cumulative reward = N/A (no logs)")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(lambda_vals, avg_rewards, marker="o")
    ax.set_xlabel("Lambda (reward shaping strength)")
    ax.set_ylabel("Average cumulative reward")
    ax.set_title("Lambda vs average cumulative reward")
    fig.tight_layout()

    out_path = os.path.join(PLOTS_DIR, "lambda_vs_reward.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[Lambda] Saved {out_path}")


def run_neural_appraisal_test() -> None:
    """Optional test: compare rule-based vs neural appraisal under shaping.

    Runs all four scenarios twice (rule-based vs neural appraisal) with
    reward shaping enabled, collects average cumulative rewards, and
    produces a small comparison plot.
    """

    def load_model_result(path: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        if not os.path.isfile(path):
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

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # 1) Rule-based appraisal with shaping
    _clear_logs_dir()
    env_rule = os.environ.copy()
    env_rule["USE_NEURAL_APPRAISAL"] = "0"
    env_rule["USE_EMOTION_REWARD"] = "1"
    env_rule["EMOTION_REWARD_LAMBDA"] = "0.5"
    env_rule["LOG_STEPS"] = "1"
    print("\n=== Neural test: rule-based appraisal ===")
    _run_scenarios_with_env(env_rule)
    ts_rule = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_rule = os.path.join(RESULTS_DIR, f"neural_rulebased_{ts_rule}")
    _snapshot_results(out_rule)

    # 2) Neural appraisal with shaping
    _clear_logs_dir()
    env_neural = os.environ.copy()
    env_neural["USE_NEURAL_APPRAISAL"] = "1"
    env_neural["USE_EMOTION_REWARD"] = "1"
    env_neural["EMOTION_REWARD_LAMBDA"] = "0.5"
    env_neural["LOG_STEPS"] = "1"
    print("\n=== Neural test: neural appraisal ===")
    _run_scenarios_with_env(env_neural)
    ts_neural = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_neural = os.path.join(RESULTS_DIR, f"neural_neural_{ts_neural}")
    _snapshot_results(out_neural)

    # Compute average rewards
    avg_rule = _compute_avg_reward_from_logs(out_rule)
    avg_neural = _compute_avg_reward_from_logs(out_neural)

    print("\n=== Neural appraisal test: average cumulative reward ===")
    if avg_rule == avg_rule:
        print(f"Rule-based appraisal: {avg_rule:.3f}")
    else:
        print("Rule-based appraisal: N/A")
    if avg_neural == avg_neural:
        print(f"Neural appraisal:    {avg_neural:.3f}")
    else:
        print("Neural appraisal: N/A")

    # Load appraisal vectors for Anxiety (if present) and print differences
    rule_app = load_model_result(os.path.join(out_rule, "model_result.csv"))
    neural_app = load_model_result(os.path.join(out_neural, "model_result.csv"))
    anx_rule = rule_app.get("Anxiety") or {}
    anx_neural = neural_app.get("Anxiety") or {}

    if anx_rule and anx_neural:
        print("\nAnxiety appraisal (rule-based vs neural):")
        for feat in ["Suddenness", "Goal_relevance", "Conduciveness", "Power"]:
            r_val = anx_rule.get(feat, float("nan"))
            n_val = anx_neural.get(feat, float("nan"))
            print(f"  {feat}: rule={r_val:.3f}, neural={n_val:.3f}")

        # Simple comparison plot for Anxiety
        labels = ["Suddenness", "Goal", "Conduciveness", "Power"]
        x = list(range(len(labels)))
        width = 0.35
        r_vals = [anx_rule.get("Suddenness", 0.0),
                  anx_rule.get("Goal_relevance", 0.0),
                  anx_rule.get("Conduciveness", 0.0),
                  anx_rule.get("Power", 0.0)]
        n_vals = [anx_neural.get("Suddenness", 0.0),
                  anx_neural.get("Goal_relevance", 0.0),
                  anx_neural.get("Conduciveness", 0.0),
                  anx_neural.get("Power", 0.0)]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar([i - width / 2 for i in x], r_vals, width, label="Rule-based")
        ax.bar([i + width / 2 for i in x], n_vals, width, label="Neural")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Appraisal value")
        ax.set_title("Anxiety appraisal: rule-based vs neural")
        ax.legend()
        fig.tight_layout()

        out_path = os.path.join(PLOTS_DIR, "neural_appraisal_comparison.png")
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"[Neural] Saved {out_path}")
    else:
        print("Anxiety appraisal vectors not found in model_result.csv for neural test.")


def main() -> None:
    run_lambda_experiments()
    # Optional neural appraisal experiment
    run_neural_appraisal_test()


if __name__ == "__main__":
    main()
