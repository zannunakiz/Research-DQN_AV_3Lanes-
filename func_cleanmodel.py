"""
Utility: clean models in `models/` by running tester validation on each .pth

Usage:
    python func_cleanmodel.py --models-dir models --max-steps 2000 [--dry-run]

Behavior:
 - For each .pth in `models_dir`, load the model into a DQNAgent
 - Run `run_tester_validation(agent, max_steps, ...)` (epsilon=0)
 - If the model fails any tester stage, increment the CSV fail count and delete the .pth
 - If it passes all stages, keep the file
 - Updates/creates `tester_stage.csv` in the models directory

This script uses the project's `CarEnvironment`, `DQNAgent`, and tester utilities.
"""
import os
import glob
import argparse
import traceback
import time


from main_environment import CarEnvironment, get_num_stages
from main_dqn_agent import DQNAgent
from main_train import run_tester_validation, _ensure_tester_stage_csv, _increment_tester_stage_fail_count
from main_constant import TEST_OBSTACLES, MEMORY_SIZE


def find_model_files(models_dir, pattern="*.pth"):
    glob_path = os.path.join(models_dir, pattern)
    return sorted(glob.glob(glob_path))


def process_model_file(model_path, env, models_dir, max_steps=2000, dry_run=False):
    print("=" * 80)
    print(f"Testing model: {model_path}")
    try:

        agent = DQNAgent(
            state_size=env.state_size,
            action_size=env.action_size,
            memory_size=int(MEMORY_SIZE),
        )


        agent.epsilon = 0.0


        try:
            agent.load(model_path)
        except Exception as e:
            print(f"ERROR: Failed to load model {model_path}: {e}")
            traceback.print_exc()
            return False, None


        print("Running tester validation (epsilon=0) ...")
        result = run_tester_validation(agent, max_steps, verbose=True, save_dir=models_dir, step_multiplier=1)


        all_passed = False
        failed_stage = None
        total_stages = None
        if isinstance(result, tuple) and len(result) >= 1:
            try:
                all_passed = bool(result[0])
                if len(result) >= 2:
                    failed_stage = result[1]
                if len(result) >= 3:
                    total_stages = result[2]
            except Exception:
                all_passed = False
        else:

            print("WARNING: Unexpected result from run_tester_validation; treating as failure.")
            all_passed = False

        if all_passed:
            print(f"[OK] Model PASSED tester validation: {os.path.basename(model_path)}")
            return True, None


        print(f"[FAIL] Model FAILED tester validation: {os.path.basename(model_path)} | failed_stage={failed_stage}")
        if total_stages is None:
            try:
                total_stages = get_num_stages(TEST_OBSTACLES)
            except Exception:
                total_stages = 0


        try:
            _increment_tester_stage_fail_count(save_dir=models_dir, total_tester_stages=int(total_stages), failed_stage=int(failed_stage) if failed_stage is not None else 0)
            print(f"Updated tester_stage.csv: incremented fail count for stage {failed_stage}")
        except Exception as e:
            print(f"Warning: could not update tester_stage.csv: {e}")

        if not dry_run:
            try:
                os.remove(model_path)
                print(f"Deleted failed model: {model_path}")
            except Exception as e:
                print(f"ERROR: Could not delete {model_path}: {e}")
        else:
            print("Dry-run mode: model would be deleted (not actually removed).")

        return False, failed_stage

    except Exception as e:
        print(f"ERROR: Exception while testing {model_path}: {e}")
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Clean models by running tester validation on each .pth")
    parser.add_argument("--models-dir", default="models", help="Directory containing .pth model files")
    parser.add_argument("--pattern", default="model_stage*.pth", help="Glob pattern to match model files")
    parser.add_argument("--max-steps", type=int, default=2000, help="Max steps per tester run")
    parser.add_argument("--dry-run", action="store_true", help="Do not delete files; just report")
    args = parser.parse_args()

    models_dir = args.models_dir
    if not os.path.isdir(models_dir):
        print(f"ERROR: models directory does not exist: {models_dir}")
        return 2


    env = CarEnvironment(curriculum_stage=0)


    try:
        total_tester_stages = get_num_stages(TEST_OBSTACLES)
        if total_tester_stages > 0:
            _ensure_tester_stage_csv(save_dir=models_dir, total_tester_stages=total_tester_stages)
            print(f"Ensured tester_stage.csv exists in {models_dir} (stages={total_tester_stages})")
    except Exception as e:
        print(f"Warning: could not ensure tester_stage.csv: {e}")

    model_files = find_model_files(models_dir, pattern=args.pattern)
    if not model_files:
        print(f"No model files found in {models_dir} matching pattern '{args.pattern}'")
        return 0

    print(f"Found {len(model_files)} model(s) to test in {models_dir}")

    passed = 0
    failed = 0
    start = time.time()
    for mf in model_files:
        ok, failed_stage = process_model_file(mf, env, models_dir, max_steps=args.max_steps, dry_run=args.dry_run)
        if ok:
            passed += 1
        else:
            failed += 1

    elapsed = time.time() - start
    print("=" * 80)
    print(f"Done. Models tested: {len(model_files)} | passed={passed} | failed={failed} | elapsed={elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
