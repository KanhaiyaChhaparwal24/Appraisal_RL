import os
import shutil
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

SCENARIOS = [
    "anxiety",
    "despair",
    "irritation",
    "rage",
]


def run_scenarios(mode_name: str, flags: dict) -> str:
    """Run all scenarios under a given flag configuration.

    Returns the path to the created results subfolder.
    """
    env = os.environ.copy()
    env.update(flags)

    print(f"\n=== Running mode: {mode_name} with flags {flags} ===")

    # Run all four MDP scenarios (each writes/updates model_result.csv)
    for scenario in SCENARIOS:
        script = os.path.join("02_mdp_model", f"{scenario}.py")
        print(f"Running {script}...")
        subprocess.run(["python", script], cwd=BASE_DIR, check=True, env=env)

    # Run SVM inference on the resulting appraisal data
    infer_script = os.path.join("03_model_infer", "01_svm_infer.py")
    print(f"Running {infer_script}...")
    subprocess.run(["python", infer_script], cwd=BASE_DIR, check=True, env=env)

    # Snapshot outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RESULTS_DIR, f"{mode_name}_{timestamp}")
    os.makedirs(out_dir, exist_ok=True)

    # Copy appraisal and SVM CSVs
    model_csv = os.path.join(DATA_DIR, "model_result.csv")
    if os.path.exists(model_csv):
        shutil.copy(model_csv, os.path.join(out_dir, "model_result.csv"))

    if os.path.isdir(DATA_DIR):
        for fname in os.listdir(DATA_DIR):
            if fname.startswith("svm_") and fname.endswith(".csv"):
                src = os.path.join(DATA_DIR, fname)
                dst = os.path.join(out_dir, fname)
                shutil.copy(src, dst)

    # Copy logs (if any) into a subfolder for this run
    if os.path.isdir(LOGS_DIR):
        logs_out = os.path.join(out_dir, "logs")
        os.makedirs(logs_out, exist_ok=True)
        for fname in os.listdir(LOGS_DIR):
            src = os.path.join(LOGS_DIR, fname)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(logs_out, fname))

    print(f"Saved results for mode '{mode_name}' to: {out_dir}")
    return out_dir


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1) Baseline: rule-based appraisal + state-based reward
    baseline_flags = {
        "USE_NEURAL_APPRAISAL": "0",
        "USE_EMOTION_REWARD": "0",
        # Per-step logging can be enabled manually for targeted runs
        "LOG_STEPS": "0",
    }
    run_scenarios("baseline", baseline_flags)

    # 2) Emotion-shaped: rule-based appraisal + emotion-aware reward
    emotion_flags = {
        "USE_NEURAL_APPRAISAL": "0",
        "USE_EMOTION_REWARD": "1",
        "EMOTION_REWARD_LAMBDA": "0.5",
        "LOG_STEPS": "0",
    }
    run_scenarios("emotion_shaped", emotion_flags)


if __name__ == "__main__":
    main()
