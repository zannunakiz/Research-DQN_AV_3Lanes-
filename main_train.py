"""
Training Script for DQN Car Navigation
Trains the agent to drive straight to finish line on 3-lane road
Supports curriculum learning with multiple obstacle stages
"""

import numpy as np
from datetime import datetime
import os
import sys
import csv
import json
import time
import random
import re

from main_environment import CarEnvironment, get_num_stages
from main_dqn_agent import DQNAgent, ALGO_DQN
from func_ttc import calculate_env_ttc_ms, format_ttc_ms, summarize_ttc_samples
from main_constant import (
    DEFAULT_SCALE,
    CONSECUTIVE_STAGE_REQ,
    NEW_STAGE_EPSILON,
    OBSTACLES,
    TEST_OBSTACLES,
    TRAIN_MULTIPLIER,
    KEYONE_MULTIPLIER,
    LEARNING_RATE,
    GAMMA,
    BATCH_SIZE,
    TARGET_UPDATE_FREQ,
    GRAD_CLIP_MAX_NORM,
    TRAIN_MAX_EPSILON,
    TRAIN_MIN_EPSILON,
    TRAIN_FINAL_MIN_EPSILON,
    TRAIN_FINAL_MIN_EPSILON_SSC,
    EPSILON_DECAY,
    CONSECUTIVE_EPSILON_RECOVERY,
    CONSECUTIVE_EPSILON_RECOVERY_SSC,
    AMOUNT_EPSILON_RECOVERY,
    ENABLE_EPSILON_RECOVERY,
    INDEPENDENT_BASED,
    INDRUN_FINAL_STAGE,
    SUCCESS_BASED_REQ,
    INDEPENDENT_COUNT_REQ,
    ValidationTesterMode,
    DECISION_INTERVAL,
    MEMORY_SIZE,
    SAVE_MODEL_CHECKPOINT,
    END_EXACT,
    SENSOR_ANGLES,
)


pygame_available = False
try:
    import pygame

    pygame_available = True
except ImportError:
    pass


# matplotlib imports removed

TESTER_STAGE_CSV_NAME = "tester_stage.csv"
CONTINUE_CSV_PREFIX = "con_"
INDEPENDENT_MODEL_SUBDIR = "independent"
TESTER_STAGE_CSV_HEADERS = ["stage", "fail_count"]
PX_PER_METER = 14.0
SENSOR_LOG_NAMES = ["R2", "R1", "F", "L1", "L2", "SR", "SL"]
SENSOR_LOG_OFFSETS = {
    "R2": 20.0,
    "R1": 20.0,
    "F": 20.0,
    "L1": 20.0,
    "L2": 20.0,
    "SR": 10.0,
    "SL": 10.0,
}


def set_global_seeds(seed: int) -> None:
    """Best-effort seeding for reproducible runs (Python, NumPy, Torch)."""
    if seed is None:
        return

    try:
        seed_int = int(seed)
    except Exception:
        return

    random.seed(seed_int)
    np.random.seed(seed_int)

    try:
        import torch

        torch.manual_seed(seed_int)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_int)

        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass


def _sensor_to_meter(sensor_value: float) -> float:
    px_value = float(sensor_value) * float(DEFAULT_SCALE)
    return px_value / float(PX_PER_METER)


def _format_min_sensor(raw_min: float, sensor_max: float, sensor_name: str) -> str:
    if raw_min is None:
        return "None"
    raw_min = max(0.0, float(raw_min))
    sensor_max = max(1e-9, float(sensor_max))
    pct = (raw_min / sensor_max) * 100.0
    adjusted = max(0.0, raw_min - float(SENSOR_LOG_OFFSETS.get(sensor_name, 20.0)))
    meter_val = _sensor_to_meter(adjusted)
    return f"{raw_min:.3f}({pct:.3f}%)({meter_val:.3f}m)"


def _build_stage_model_path(
    save_dir: str,
    stage_number: int,
    episode_number: int = None,
    checkpoint: bool = False,
) -> str:
    """
    Build stage checkpoint filename.

    For final-stage independent/tester valid successes, append "-[episode]" to filename:
    model_stage6-100.pth (for episode 100)
    """
    base_name = f"model_stage{int(stage_number)}"
    suffix = "_check" if checkpoint else ""
    if episode_number is None:
        return os.path.join(save_dir, f"{base_name}{suffix}.pth")
    return os.path.join(save_dir, f"{base_name}-{int(episode_number)}{suffix}.pth")


def _build_independent_model_path(
    save_dir: str, stage_number: int, episode_number: int
) -> str:
    """Build checkpoint path for final-stage independent successes."""
    return _build_stage_model_path(
        save_dir=os.path.join(save_dir, INDEPENDENT_MODEL_SUBDIR),
        stage_number=stage_number,
        episode_number=episode_number,
    )


def _continue_csv_name(csv_name: str, continue_training: bool) -> str:
    if not continue_training:
        return csv_name
    if csv_name.startswith(CONTINUE_CSV_PREFIX):
        return csv_name
    return f"{CONTINUE_CSV_PREFIX}{csv_name}"


def _infer_stage_from_model_path(model_path: str):
    if not model_path:
        return None
    match = re.search(r"model_stage(\d+)", os.path.basename(str(model_path)))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _resolve_start_stage(current_stage, model_path: str, num_stages: int) -> int:
    stage_number = current_stage
    if stage_number is None:
        stage_number = _infer_stage_from_model_path(model_path)
    if stage_number is None:
        stage_number = 1

    try:
        stage_idx = int(stage_number) - 1
    except Exception:
        stage_idx = 0
    return min(max(stage_idx, 0), max(0, int(num_stages) - 1))


def _infer_episode_from_model_path(model_path: str):
    if not model_path:
        return None
    match = re.search(r"model_stage\d+-(\d+)\.pth$", os.path.basename(str(model_path)))
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _load_checkpoint_file(model_path: str, map_location="cpu"):
    if not model_path or not os.path.exists(model_path):
        return None
    try:
        import torch

        try:
            return torch.load(
                model_path, map_location=map_location, weights_only=False
            )
        except TypeError:
            return torch.load(model_path, map_location=map_location)
    except Exception:
        return None


def _checkpoint_training_state(checkpoint) -> dict:
    if not isinstance(checkpoint, dict):
        return {}
    state = checkpoint.get("training_state", {})
    return state if isinstance(state, dict) else {}


def _int_or_none(value):
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _resolve_resume_start_episode(cli_start_episode, training_state: dict, model_path: str) -> int:
    cli_value = _int_or_none(cli_start_episode)
    if cli_value is not None:
        return max(1, cli_value)

    next_episode = _int_or_none(training_state.get("next_episode"))
    if next_episode is not None:
        return max(1, next_episode)

    last_episode = _int_or_none(training_state.get("last_episode"))
    if last_episode is None:
        last_episode = _infer_episode_from_model_path(model_path)
    if last_episode is not None:
        return max(1, last_episode + 1)

    return 1


def _resolve_resume_stage(
    cli_current_stage, training_state: dict, model_path: str, num_stages: int
) -> int:
    cli_value = _int_or_none(cli_current_stage)
    if cli_value is not None:
        return _resolve_start_stage(cli_value, None, num_stages)

    state_stage = _int_or_none(training_state.get("current_stage"))
    if state_stage is not None:
        return _resolve_start_stage(state_stage, None, num_stages)

    state_stage_idx = _int_or_none(training_state.get("current_stage_index"))
    if state_stage_idx is not None:
        return min(max(state_stage_idx, 0), max(0, int(num_stages) - 1))

    return _resolve_start_stage(None, model_path, num_stages)


def _resolve_resume_epsilon(cli_epsilon, training_state: dict, checkpoint):
    cli_value = _float_or_none(cli_epsilon)
    if cli_value is not None:
        return max(0.0, cli_value)

    for key in ("epsilon_after_episode", "next_epsilon", "epsilon"):
        state_value = _float_or_none(training_state.get(key))
        if state_value is not None:
            return max(0.0, state_value)

    if isinstance(checkpoint, dict):
        checkpoint_value = _float_or_none(checkpoint.get("epsilon"))
        if checkpoint_value is not None:
            return max(0.0, checkpoint_value)

    return None


def _infer_checkpoint_algo(model_path: str):
    checkpoint = _load_checkpoint_file(model_path, map_location="cpu")
    try:
        algo = checkpoint.get("algo") if isinstance(checkpoint, dict) else None
        return str(algo).lower() if algo else None
    except Exception:
        return None


def should_stop_on_episode_target_valid(
    episode: int,
    num_episodes: int,
    episode_had_final_valid_success: bool,
    end_exact: bool = END_EXACT,
) -> bool:
    """
    Stop on requested episode target.

    With end_exact=True, stop immediately at episode N.
    With end_exact=False, continue after N until first final-stage VALID success.
    """
    try:
        target = int(num_episodes)
    except Exception:
        return False
    if target >= int(sys.maxsize):
        return False
    if target <= 0:
        return False
    if int(episode) < target:
        return False
    return bool(end_exact) or bool(episode_had_final_valid_success)


def _discounted_step_value(reward: float, gamma: float, power: int) -> float:
    """Per-step discounted contribution used in training return accumulation."""
    return (float(gamma) ** int(power)) * float(reward)


def _fmt_reward4(value: float) -> str:
    """Format reward/avg_reward with 4 decimal places."""
    return f"{float(value):.4f}"


def _train_dqn_run(
    num_episodes=500,
    max_steps_per_episode=999999,
    save_interval=SAVE_MODEL_CHECKPOINT,
    render_interval=100,
    verbose=True,
    visualize=False,
    render_every_n_episodes=1,
    load_model_path=None,
    continue_training=False,
    start_episode=None,
    start_epsilon=None,
    current_stage=None,
    fast_multiply=False,
    multi_valid=False,
    formula=False,
    traininfo=False,
    seed=None,
    memory_size=MEMORY_SIZE,
    save_dir="models",
    save_models=True,
    csv_log_path=None,
    csv_float_decimals=None,
    renderer=None,
    close_renderer=True,
    plot_results=True,
    neuron_mode=False,
    algo=ALGO_DQN,
):
    """
    Train the DQN agent with curriculum learning support

    Args:
        num_episodes: Number of training episodes
        max_steps_per_episode: Maximum steps per episode
        save_interval: Save model every n episodes
        render_interval: Print detailed info every n episodes
        verbose: Print training progress
        visualize: Enable real-time pygame visualization during training
        render_every_n_episodes: Render visualization every N episodes (to speed up training)
        load_model_path: Path to a model to load (optional)
        continue_training: Resume a stopped run using load_model_path, start_episode,
            start_epsilon, and current_stage.
        start_episode: First episode number to write in logs when resuming.
        start_epsilon: Epsilon value to resume from. If None, checkpoint epsilon is kept.
        current_stage: 1-based curriculum stage to resume from. If None, infer from model filename or start at stage 1.
        multi_valid: Speed up independent/tester validation runs using KEYONE_MULTIPLIER
        traininfo: Verbose per-decision-interval training info logs
        renderer: Optional GameRenderer to reuse during visualization
        close_renderer: If True, close renderer at end
        plot_results: If True, generate matplotlib plots at end (if available)
        neuron_mode: Show neural-network forward-pass details in visualization
    """

    if neuron_mode and not visualize:
        print("[INFO] --neuron requested; enabling --visualize.")
        visualize = True

    if visualize and not pygame_available:
        print(
            "WARNING: Visualization requested but pygame not installed. Running without visualization."
        )
        visualize = False

    if seed is not None:
        set_global_seeds(seed)

    if continue_training:
        if not load_model_path:
            raise ValueError("--continue requires --model <path/model.pth>")
        if not os.path.exists(load_model_path):
            raise FileNotFoundError(f"Continue model not found: {load_model_path}")

    resume_checkpoint = (
        _load_checkpoint_file(load_model_path, map_location="cpu")
        if continue_training
        else None
    )
    resume_state = _checkpoint_training_state(resume_checkpoint)

    num_stages = get_num_stages(OBSTACLES)
    start_stage_idx = (
        _resolve_resume_stage(current_stage, resume_state, load_model_path, num_stages)
        if continue_training
        else _resolve_start_stage(None, None, num_stages)
    )

    start_episode = (
        _resolve_resume_start_episode(start_episode, resume_state, load_model_path)
        if continue_training
        else _resolve_resume_start_episode(start_episode or 1, {}, None)
    )

    if continue_training and int(num_episodes) < start_episode:
        raise ValueError("--episodes must be greater than or equal to resume episode")

    env = CarEnvironment(curriculum_stage=start_stage_idx)

    if visualize:
        from main_visualize import GameRenderer, build_neuron_trace

        if renderer is None:
            renderer = GameRenderer(env, scale=DEFAULT_SCALE, neuron_mode=neuron_mode)
            print(
                "[OK] Visualization enabled - pygame window will show training progress"
            )
        else:
            renderer.env = env
            renderer.neuron_mode = bool(neuron_mode)

    agent = DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=TRAIN_MAX_EPSILON,
        epsilon_min=TRAIN_MIN_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        memory_size=int(memory_size),
        algo=algo,
    )

    if load_model_path and os.path.exists(load_model_path):
        loaded_checkpoint = agent.load(load_model_path, checkpoint=resume_checkpoint)
        if continue_training and not resume_state:
            resume_state = _checkpoint_training_state(loaded_checkpoint)

        if continue_training:
            resolved_epsilon = _resolve_resume_epsilon(
                start_epsilon, resume_state, loaded_checkpoint
            )
            if resolved_epsilon is not None:
                agent.epsilon = float(resolved_epsilon)
        elif not continue_training:
            agent.epsilon = float(TRAIN_MAX_EPSILON)
        print(f"[OK] Loaded model from {load_model_path}")
        if continue_training:
            print(f"[OK] Resume epsilon set to {agent.epsilon:.4f}")
            print(f"[OK] Replay buffer restored: {len(agent.memory)} transition(s)")
            print(f"[OK] Optimizer update counter: {int(agent.update_counter)}")
        else:
            print(f"[OK] Epsilon reset to {agent.epsilon} for training")

    os.makedirs(save_dir, exist_ok=True)

    if ValidationTesterMode:
        tester_stage_count = get_num_stages(TEST_OBSTACLES)
        if tester_stage_count > 0:
            try:
                _ensure_tester_stage_csv(
                    save_dir=save_dir,
                    total_tester_stages=tester_stage_count,
                    continue_training=continue_training,
                )
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not initialize tester stage CSV: {e}")

    csv_stage_number = start_stage_idx + 1
    default_csv_name = _continue_csv_name(
        f"model_stage{csv_stage_number}.csv", continue_training
    )
    default_csv_path = os.path.join(save_dir, default_csv_name)
    if csv_log_path is None:
        csv_log_path = default_csv_path

    episode_rewards = []
    episode_steps = []
    episode_timeframes = []
    episode_success = []
    episode_stages = []
    avg_rewards = []
    episode_mse = []
    episode_epsilons = []
    episode_progress_pct = []
    episode_streaks = []
    episode_buffer_sizes = []
    episode_time_ms = []
    episode_sensor_min_logs = []
    episode_ttc_logs = []
    episode_close_distance_logs = []
    episode_near_miss_logs = []
    episode_numbers = []

    print("=" * 60)
    algo_label = str(algo).upper()
    print(f"Starting {algo_label} Training with Curriculum Learning")
    print(f"Algorithm: {algo_label}")
    if continue_training:
        print("Mode: CONTINUE TRAINING")
    else:
        print("Mode: NEW TRAINING")
    print("=" * 60)
    print(f"State size: {env.state_size}")
    print(f"Action size: {env.action_size}")
    episodes_str = "INF" if int(num_episodes) == int(sys.maxsize) else str(num_episodes)
    print(f"Episodes: {episodes_str}")
    if continue_training:
        print(f"Start Episode: {start_episode}")
    print(f"Curriculum Stages: {num_stages}")
    print(f"Consecutive successes for stage advance: {CONSECUTIVE_STAGE_REQ}")
    print(f"Final-stage independent run: {'ON' if INDRUN_FINAL_STAGE else 'OFF'}")
    print(f"Epsilon min (normal stages): {TRAIN_MIN_EPSILON}")
    print(f"Epsilon min (final stage, SSC=0): {TRAIN_FINAL_MIN_EPSILON}")
    print(f"Epsilon min (final stage, SSC>0): {TRAIN_FINAL_MIN_EPSILON_SSC}")
    if float(NEW_STAGE_EPSILON) == 0.0:
        print("New stage epsilon: 0 (no reset; continue current epsilon)")
    else:
        print(f"New stage epsilon: {NEW_STAGE_EPSILON}")
    print(f"Replay buffer capacity (memory_size): {int(memory_size)}")
    if seed is not None:
        print(f"Seed: {int(seed)}")
    print(f"Visualization: {'ON' if visualize else 'OFF'}")
    if visualize:
        print(f"Rendering every {render_every_n_episodes} episode(s)")
    if load_model_path:
        model_mode = "Continuing from model" if continue_training else "Loaded model"
        print(f"{model_mode}: {load_model_path}")
    print("=" * 60)

    running = True

    cli_stage_override = current_stage is not None
    current_stage = start_stage_idx
    saved_stage_number = _int_or_none(resume_state.get("current_stage"))
    use_saved_stage_state = (
        bool(continue_training)
        and (
            (not cli_stage_override and saved_stage_number is None)
            or (
                saved_stage_number is not None
                and int(saved_stage_number) == int(start_stage_idx) + 1
            )
        )
    )
    stage_success_streak = (
        max(0, _int_or_none(resume_state.get("stage_success_streak")) or 0)
        if use_saved_stage_state
        else 0
    )
    stage_success_count = (
        max(0, _int_or_none(resume_state.get("stage_success_count")) or 0)
        if use_saved_stage_state
        else 0
    )
    independent_success_count = (
        max(0, _int_or_none(resume_state.get("independent_success_count")) or 0)
        if use_saved_stage_state
        else 0
    )
    epsilon_min_streak = (
        max(0, _int_or_none(resume_state.get("epsilon_min_streak")) or 0)
        if use_saved_stage_state
        else 0
    )
    stop_training = False

    def _get_stage_epsilon_min(stage_idx: int) -> float:
        """Return epsilon floor for the given curriculum stage."""
        is_final_stage_idx = (
            int(num_stages) > 0 and int(stage_idx) >= int(num_stages) - 1
        )
        if is_final_stage_idx:
            if int(stage_success_count) > 0:
                return float(TRAIN_FINAL_MIN_EPSILON_SSC)
            return float(TRAIN_FINAL_MIN_EPSILON)
        return float(TRAIN_MIN_EPSILON)

    def _sync_stage_epsilon_min(stage_idx: int, reason: str = "") -> None:
        """Apply stage-aware epsilon floor to the agent."""
        prev_min = float(agent.epsilon_min)
        next_min = _get_stage_epsilon_min(stage_idx)
        agent.epsilon_min = next_min
        if verbose and abs(prev_min - next_min) > 1e-12:
            stage_label = f"Stage {int(stage_idx) + 1}/{int(num_stages)}"
            reason_text = f" ({reason})" if reason else ""
            print(
                f"[OK] Epsilon min updated{reason_text} for {stage_label}: "
                f"{prev_min:.4f} -> {next_min:.4f}"
            )

    def _get_completed_pair_averages(rewards, episode_num: int):
        """
        Return two completed 50-episode window averages for stability display.

        Example:
        - episode 100..149 -> avg1-50 and avg51-100
        - episode 150..199 -> avg51-100 and avg101-150
        """
        completed_block_end = (int(episode_num) // 50) * 50
        if completed_block_end < 100:
            return None

        older_start = completed_block_end - 99
        older_end = completed_block_end - 50
        newer_start = completed_block_end - 49
        newer_end = completed_block_end

        older_values = rewards[older_start - 1 : older_end]
        newer_values = rewards[newer_start - 1 : newer_end]
        if not older_values or not newer_values:
            return None

        older_label = f"avg{older_start}-{older_end}"
        newer_label = f"avg{newer_start}-{newer_end}"
        older_avg = float(np.mean(older_values))
        newer_avg = float(np.mean(newer_values))
        stability_gap = float(newer_avg - older_avg)
        pair_changed_now = int(episode_num) == int(completed_block_end)

        return {
            "older_label": older_label,
            "older_avg": older_avg,
            "newer_label": newer_label,
            "newer_avg": newer_avg,
            "stability_gap": stability_gap,
            "pair_changed_now": pair_changed_now,
        }

    _sync_stage_epsilon_min(current_stage, reason="initial stage")

    train_multiplier = TRAIN_MULTIPLIER if fast_multiply else 1
    valid_multiplier = 1
    if multi_valid:
        try:
            valid_multiplier = int(KEYONE_MULTIPLIER)
        except Exception:
            valid_multiplier = 1
        if valid_multiplier < 1:
            valid_multiplier = 1
        if verbose and valid_multiplier > 1:
            print(
                f"[INFO] Multi-valid enabled: independent/tester runs speed x{valid_multiplier}."
            )

    def _training_state_payload(
        reason: str,
        episode_number: int,
        epsilon_start_value: float,
        stage_idx_for_resume=None,
        stage_success_streak_value=None,
        stage_success_count_value=None,
        independent_success_count_value=None,
        epsilon_min_streak_value=None,
    ) -> dict:
        resume_stage_idx = (
            int(current_stage)
            if stage_idx_for_resume is None
            else int(stage_idx_for_resume)
        )
        return {
            "checkpoint_version": 2,
            "save_reason": str(reason),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "last_episode": int(episode_number),
            "next_episode": int(episode_number) + 1,
            "current_stage": int(resume_stage_idx) + 1,
            "current_stage_index": int(resume_stage_idx),
            "num_stages": int(num_stages),
            "stage_success_streak": int(
                stage_success_streak
                if stage_success_streak_value is None
                else stage_success_streak_value
            ),
            "stage_success_count": int(
                stage_success_count
                if stage_success_count_value is None
                else stage_success_count_value
            ),
            "independent_success_count": int(
                independent_success_count
                if independent_success_count_value is None
                else independent_success_count_value
            ),
            "epsilon_min_streak": int(
                epsilon_min_streak
                if epsilon_min_streak_value is None
                else epsilon_min_streak_value
            ),
            "last_logged_epsilon": float(epsilon_start_value),
            "epsilon": float(agent.epsilon),
            "epsilon_after_episode": float(agent.epsilon),
            "next_epsilon": float(agent.epsilon),
            "epsilon_min": float(agent.epsilon_min),
            "epsilon_decay": float(agent.epsilon_decay),
            "buffer_len": int(len(agent.memory)),
            "update_counter": int(agent.update_counter),
            "memory_size": int(agent.memory_size),
            "batch_size": int(agent.batch_size),
            "target_update_freq": int(agent.target_update_freq),
            "decision_interval": int(DECISION_INTERVAL),
            "train_multiplier": int(train_multiplier),
            "valid_multiplier": int(valid_multiplier),
            "algo": str(algo).lower(),
            "csv_log_path": str(csv_log_path),
        }

    def _save_training_checkpoint(
        path: str,
        reason: str,
        episode_number: int,
        epsilon_start_value: float,
        **state_overrides,
    ) -> None:
        if not save_models:
            return
        checkpoint_dir = os.path.dirname(os.path.abspath(path))
        os.makedirs(checkpoint_dir, exist_ok=True)
        agent.save(
            path,
            training_state=_training_state_payload(
                reason=reason,
                episode_number=episode_number,
                epsilon_start_value=epsilon_start_value,
                **state_overrides,
            ),
        )

    csv_headers = [
        "episode",
        "stage",
        "reward",
        "avg_reward",
        "MSE",
        "timeframe",
        "steps",
        "close distance",
        "near_miss",
        "epsilon",
        "progress_pct",
        "success_rate",
        "Buffer",
        "time_ms",
        "streak",
        "reached_finish",
        "min_R2",
        "min_R1",
        "min_F",
        "min_L1",
        "min_L2",
        "min_SR",
        "min_SL",
        "min_ttc",
        "avg_ttc",
    ]

    traininfo_enabled = bool(traininfo)
    sensor_names = ["R2", "R1", "F", "L1", "L2", "SR", "SL"]
    sensor_labels = [
        f"{name}({float(angle):+g})" for name, angle in zip(sensor_names, SENSOR_ANGLES)
    ]
    action_labels = [
        "slow_left",
        "slow_straight",
        "slow_right",
        "fast_left",
        "fast_straight",
        "fast_right",
    ]

    def _ti_fmt_list(labels, values, decimals=3):
        parts = []
        for lab, val in zip(labels, values):
            if val is None:
                parts.append(f"{lab}=None")
            else:
                parts.append(f"{lab}={val:.{decimals}f}")
        return ", ".join(parts)

    def _ti_fmt_state(values):
        if values is None:
            return "None"
        return _ti_fmt_list(sensor_labels + ["SPD_N"], values, decimals=3)

    def _ti_fmt_raw(values):
        if values is None:
            return "None"
        return _ti_fmt_list(sensor_labels + ["SPD"], values, decimals=3)

    def _ti_fmt_q(values):
        if values is None:
            return "None"
        return ", ".join(
            f"{lab}={float(val):.3f}" for lab, val in zip(action_labels, values)
        )

    def _ti_print_segment(seg, tinfo):
        if not traininfo_enabled or seg is None:
            return
        seg_sep = "=" * 88
        frame_sep = "-" * 88
        print(f"\n{seg_sep}")
        print(f"[EP{seg['episode']} FRAME{seg['start_frame']}-{seg['end_frame']}]")
        print(seg_sep)
        end_frame = seg.get("end_frame")
        frames = seg.get("frames", [])

        for fr in frames:
            fno = fr.get("frame")
            print(f"[F{fno}]")
            print(f"[INPUT] raw  : {_ti_fmt_raw(fr.get('state_raw'))}")
            print(f"[INPUT] norm : {_ti_fmt_state(fr.get('state_norm'))}")
            q_line = _ti_fmt_q(fr.get("q_values"))
            act = fr.get("action")
            if act is not None:
                act_label = (
                    action_labels[act] if 0 <= act < len(action_labels) else str(act)
                )
                mode = fr.get("mode", "hold")
                eps_val = fr.get("epsilon")
                eps_str = f"{float(eps_val):.3f}" if eps_val is not None else "None"
                q_line = f"{q_line} | action={act_label} (a={act}) | mode={mode} | eps={eps_str}"
            print(f"[OUTPUT] {q_line}")
            reward = fr.get("reward")
            reward_str = f"{float(reward):+.3f}" if reward is not None else "None"
            accum = fr.get("accum_return")
            accum_str = f"{float(accum):+.3f}" if accum is not None else "None"
            print(f"[REWARD] r={reward_str} | accum={accum_str}")
            print(f"[TTC] current_ms={format_ttc_ms(fr.get('ttc_ms'))}")
            print("[TRAIN INFO] ----------------------------------------")
            disc = fr.get("discounted")
            disc_str = f"{float(disc):+.3f}" if disc is not None else "None"
            k = fr.get("discount_pow")
            k_str = "None" if k is None else str(int(k))
            print(f"[DISCOUNTED] r*gamma^{k_str}={disc_str} | accum={accum_str}")

            is_last = end_frame is not None and fno == end_frame
            if not is_last:
                print("[REPLAY BUFFER] not yet")
                print("[MINI-BATCH] not yet")
                print("[TARGET Q] not yet")
                print("[TD ERROR] not yet")
                print("[MSE LOSS] not yet")
                print("[UPDATE ONLINE NETWORK/BACKPROP] not yet")
                print("[TARGET NETWORK UPDATE] not yet")
                print(frame_sep)
                continue

            print(
                "[REPLAY BUFFER] "
                f"D=(s=[{_ti_fmt_state(seg.get('decision_state_norm'))}], "
                f"a={seg.get('decision_action')}, Rn={float(seg.get('decision_return', 0.0)):+.3f}, "
                f"s'=[{_ti_fmt_state(seg.get('next_state'))}], d={int(bool(seg.get('done')))}, "
                f"discount={float(seg.get('decision_discount', 0.0)):.3f}) | "
                f"len={int(seg.get('buffer_len', 0))}"
            )

            if not tinfo or tinfo.get("skipped"):
                buf_len = int(tinfo.get("buffer_len", 0)) if tinfo else 0
                batch_size = (
                    int(tinfo.get("batch_size", BATCH_SIZE)) if tinfo else BATCH_SIZE
                )
                print(f"[MINI-BATCH] not yet (buffer_len={buf_len}, B={batch_size})")
                print("[TARGET Q] not yet")
                print("[TD ERROR] not yet")
                print("[MSE LOSS] not yet")
                print("[UPDATE ONLINE NETWORK/BACKPROP] not yet")
                print("[TARGET NETWORK UPDATE] not yet")
                print(frame_sep)
                continue

            sample = tinfo.get("sample", {})
            sample_state = sample.get("state")
            sample_next = sample.get("next_state")
            print(
                "[MINI-BATCH] "
                f"B={tinfo.get('batch_size')}, buffer_len={tinfo.get('buffer_len')}, "
                f"sample0: a={sample.get('action')}, Rn={float(sample.get('reward', 0.0)):+.3f}, "
                f"d={int(float(sample.get('done', 0.0)))}, discount={float(sample.get('discount', 0.0)):.3f}"
            )
            print(f"[MINI-BATCH] sample0 s=[{_ti_fmt_state(sample_state)}]")
            print(f"[MINI-BATCH] sample0 s'=[{_ti_fmt_state(sample_next)}]")

            tgt = tinfo.get("target", {})
            print("[TARGET Q] formula: y = r + (1-d) * discount * maxQ_next")
            print(
                "[TARGET Q] values : "
                f"r={float(tgt.get('r', 0.0)):+.3f}, d={int(float(tgt.get('d', 0.0)))}, "
                f"discount={float(tgt.get('discount', 0.0)):.3f}, maxQ_next={float(tgt.get('max_next', 0.0)):.3f}"
            )
            print(f"[TARGET Q] result : y={float(tgt.get('y', 0.0)):+.3f}")

            q_cur = tinfo.get("q", {}).get("current", 0.0)
            td_err = tinfo.get("td_error", 0.0)
            print("[TD ERROR] formula: delta = y - Q(s,a)")
            print(
                "[TD ERROR] values : "
                f"y={float(tgt.get('y', 0.0)):+.3f}, Q={float(q_cur):+.3f}"
            )
            print(f"[TD ERROR] result : delta={float(td_err):+.3f}")

            loss_val = float(tinfo.get("loss", 0.0))
            sq_err = tinfo.get("sq_error", None)
            sq_err_str = f"{float(sq_err):.6f}" if sq_err is not None else "None"
            y_val = float(tgt.get("y", 0.0))
            q_val = float(q_cur)
            print("[MSE LOSS] formula: loss = mean((y - Q)^2)")
            print(
                "[MSE LOSS] values : "
                f"B={int(tinfo.get('batch_size', BATCH_SIZE))}, y0={y_val:+.3f}, Q0={q_val:+.3f}, "
                f"loss={loss_val:.6f}, sample_sq={sq_err_str}"
            )
            print(f"[MSE LOSS] result : loss={loss_val:.6f}")

            grad_norm = tinfo.get("grad_norm", None)
            grad_str = f"{float(grad_norm):.3f}" if grad_norm is not None else "None"
            print(
                "[UPDATE ONLINE NETWORK/BACKPROP] "
                f"grad_norm={grad_str}, clip_max={float(GRAD_CLIP_MAX_NORM):.3f}, optimizer=Adam"
            )

            updated = bool(tinfo.get("updated_target"))
            counter = int(tinfo.get("update_counter", 0))
            freq = int(tinfo.get("target_update_freq", TARGET_UPDATE_FREQ))
            if updated:
                print(
                    "[TARGET NETWORK UPDATE] "
                    f"updated=yes (counter={counter}, freq={freq})"
                )
            else:
                print(
                    f"[TARGET NETWORK UPDATE] not yet (counter={counter}, freq={freq})"
                )
            print(frame_sep)

    episode_target = int(num_episodes)
    if episode_target < int(sys.maxsize):
        if END_EXACT:
            print(
                f"[INFO] Episode target mode: stop exactly after episode {episode_target}, then save logs/model."
            )
        else:
            print(
                f"[INFO] Episode target mode: continue past episode {episode_target} until first final-stage VALID success (unless INDEPENDENT_COUNT_REQ is reached earlier)."
            )
    episode = start_episode - 1
    while running:
        episode += 1
        episode_start_perf = time.perf_counter()
        state = env.reset()
        total_reward = 0.0
        total_discounted_reward = 0.0
        steps = 0
        global_env_step = 0
        _sync_stage_epsilon_min(current_stage)

        epsilon_start = float(agent.epsilon)

        if traininfo_enabled:
            print("\n" + "=" * 88)
            print(f"[EP{episode}] START")
            print("=" * 88)

        episode_train_loss_sum = 0.0
        episode_train_loss_count = 0

        should_render = visualize and (episode % render_every_n_episodes == 0)
        paused = False

        cached_nn_output = None

        current_action = None
        action = 1
        decision_state = None
        decision_action = None
        decision_return = 0.0
        decision_horizon = 0
        episode_sensor_mins = [float("inf")] * len(SENSOR_LOG_NAMES)
        episode_ttc_samples = []
        close_distance_total = 0
        near_miss_total = 0

        ti_pending = []
        ti_current = None

        while steps < max_steps_per_episode:
            if should_render and renderer:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            running = False
                            break
                        elif event.key == pygame.K_p:
                            paused = not paused
                            print(
                                f"\n{'PAUSED' if paused else 'RESUMED'} (Press P to toggle)"
                            )
                        elif event.key == pygame.K_1 and renderer is not None:
                            try:
                                renderer.toggle_speed()
                                print(
                                    f"\nSpeed Mode: {renderer.get_speed_mode_label()}"
                                )
                            except Exception:
                                pass
                        elif event.key == pygame.K_2 and renderer is not None:
                            try:
                                renderer.toggle_slow_motion()
                                print(
                                    f"\nSpeed Mode: {renderer.get_speed_mode_label()}"
                                )
                            except Exception:
                                pass
                        elif event.key == pygame.K_i and renderer is not None:
                            try:
                                indicators_on = renderer.toggle_indicators()
                                print(
                                    "\nIndicators: "
                                    f"{'ON' if indicators_on else 'OFF'}"
                                )
                            except Exception:
                                pass

                if not running:
                    print("\nTraining interrupted by user")
                    break

                while paused and running:
                    if renderer:
                        renderer.render_cached(paused=True)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            running = False
                            paused = False
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_q:
                                running = False
                                paused = False
                            elif event.key == pygame.K_p:
                                paused = False
                                print("\nRESUMED")
                            elif event.key == pygame.K_1 and renderer is not None:
                                try:
                                    renderer.toggle_speed()
                                    print(
                                        f"\nSpeed Mode: {renderer.get_speed_mode_label()}"
                                    )
                                except Exception:
                                    pass
                            elif event.key == pygame.K_2 and renderer is not None:
                                try:
                                    renderer.toggle_slow_motion()
                                    print(
                                        f"\nSpeed Mode: {renderer.get_speed_mode_label()}"
                                    )
                                except Exception:
                                    pass
                            elif event.key == pygame.K_i and renderer is not None:
                                try:
                                    indicators_on = renderer.toggle_indicators()
                                    print(
                                        "\nIndicators: "
                                        f"{'ON' if indicators_on else 'OFF'}"
                                    )
                                except Exception:
                                    pass
                    if not running:
                        break

            inner_count = 0
            decision_updates_pending = 0
            while (
                inner_count < train_multiplier
                and steps < max_steps_per_episode
                and running
            ):
                if formula:
                    step_tag = f"E{episode}/T{steps}"
                    sep = "/" * 72
                    print(f"\n[FORMULA][{step_tag}][TIMEFRAME] {sep}")

                    try:
                        q_s = agent.get_q_values_with_calculation(state, tag=step_tag)
                        q_s_list = [f"{float(x):.3f}" for x in q_s.tolist()]
                        q_max = float(np.max(q_s))
                        a_greedy = int(np.argmax(q_s))
                        print(
                            f"[FORMULA][{step_tag}][Q_VALUES] Q(s) = policy_net(s) = [{', '.join(q_s_list)}]"
                        )
                        print(
                            f"[FORMULA][{step_tag}][Q_ARGMAX] argmax_a Q(s,a) = {a_greedy} with maxQ={q_max:.3f}"
                        )
                    except Exception as e:
                        print(f"[FORMULA][{step_tag}][WARN] Q(s) logging failed: {e}")

                is_decision_step = steps % DECISION_INTERVAL == 0
                if is_decision_step:
                    ainfo = None
                    if formula or traininfo_enabled:
                        current_action, ainfo = agent.select_action(
                            state, training=True, debug=True
                        )
                        if formula:
                            try:
                                rand_val = ainfo.get("rand", None)
                                rand_str = (
                                    "None"
                                    if rand_val is None
                                    else f"{float(rand_val):.3f}"
                                )
                                print(
                                    f"[FORMULA][{step_tag}][ACTION_SELECT] epsilon-greedy: "
                                    f"explore={bool(ainfo.get('explore'))}, rand={rand_str}, epsilon={float(ainfo.get('epsilon')):.3f}"
                                )
                            except Exception:
                                pass
                    else:
                        current_action = agent.select_action(state, training=True)
                    decision_state = np.array(state, copy=True)
                    decision_action = current_action
                    decision_return = 0.0
                    decision_horizon = 0
                    if traininfo_enabled:
                        ti_current = {
                            "episode": int(episode),
                            "start_frame": int(steps),
                            "decision_state_norm": state.tolist(),
                            "decision_action": int(current_action)
                            if current_action is not None
                            else None,
                            "frames": [],
                        }

                action = current_action if current_action is not None else 1
                frame_info = None
                if traininfo_enabled:
                    if ti_current is None:
                        ti_current = {
                            "episode": int(episode),
                            "start_frame": int(steps),
                            "decision_state_norm": state.tolist(),
                            "decision_action": int(action),
                            "frames": [],
                        }
                    mode = "hold"
                    eps_val = None
                    if is_decision_step and ainfo is not None:
                        mode = "explore" if ainfo.get("explore") else "greedy"
                        eps_val = ainfo.get("epsilon")
                    try:
                        sensor_info = env._get_sensor_angles_and_distances()
                        raw_sensors = [
                            float(s.get("distance", 0.0)) for s in sensor_info
                        ]
                    except Exception:
                        raw_sensors = [None] * len(sensor_labels)
                    raw_speed = float(getattr(env, "car_speed", 0.0))
                    raw_values = raw_sensors + [raw_speed]
                    try:
                        q_vals = agent.get_q_values(state).tolist()
                    except Exception:
                        q_vals = None
                    frame_info = {
                        "frame": int(steps),
                        "state_raw": raw_values,
                        "state_norm": state.tolist(),
                        "q_values": q_vals,
                        "action": int(action),
                        "mode": mode,
                        "epsilon": float(eps_val) if eps_val is not None else None,
                    }
                next_state, reward, done, info = env.step(
                    action, apply_steering=is_decision_step
                )
                current_ttc_ms = info.get("ttc_ms")
                if current_ttc_ms is None:
                    current_ttc_ms = calculate_env_ttc_ms(env)
                if current_ttc_ms is not None:
                    episode_ttc_samples.append(float(current_ttc_ms))
                close_distance_total += int(info.get("warning_close_count", 0))
                near_miss_total += int(info.get("near_miss", 0))
                try:
                    sensor_snapshot = env._get_sensor_angles_and_distances()
                    for sensor_idx, sensor in enumerate(sensor_snapshot):
                        if sensor_idx >= len(episode_sensor_mins):
                            break
                        dist = float(sensor.get("distance", 0.0))
                        if dist < episode_sensor_mins[sensor_idx]:
                            episode_sensor_mins[sensor_idx] = dist
                except Exception:
                    pass

                if decision_state is None:
                    decision_state = np.array(state, copy=True)
                    decision_action = action
                    decision_return = 0.0
                    decision_horizon = 0
                discount_pow = decision_horizon
                discounted_val = _discounted_step_value(
                    reward, agent.gamma, discount_pow
                )
                decision_return += discounted_val
                total_discounted_reward += discounted_val
                if (
                    traininfo_enabled
                    and ti_current is not None
                    and frame_info is not None
                ):
                    frame_info["reward"] = float(reward)
                    frame_info["discount_pow"] = int(discount_pow)
                    frame_info["discounted"] = float(discounted_val)
                    frame_info["accum_return"] = float(decision_return)
                    frame_info["done"] = bool(done)
                    frame_info["ttc_ms"] = current_ttc_ms
                    ti_current["frames"].append(frame_info)
                decision_horizon += 1

                closes_decision = (
                    done
                    or ((steps + 1) % DECISION_INTERVAL == 0)
                    or ((steps + 1) >= max_steps_per_episode)
                )
                if closes_decision and decision_horizon > 0:
                    decision_discount = agent.gamma**decision_horizon
                    agent.remember(
                        decision_state,
                        decision_action,
                        decision_return,
                        next_state,
                        done,
                        discount=decision_discount,
                    )
                    decision_updates_pending += 1
                    if traininfo_enabled and ti_current is not None:
                        ti_current["end_frame"] = int(steps)
                        if ti_current.get("decision_state_norm") is None:
                            ti_current["decision_state_norm"] = decision_state.tolist()
                        if ti_current.get("decision_action") is None:
                            ti_current["decision_action"] = (
                                int(decision_action)
                                if decision_action is not None
                                else None
                            )
                        ti_current["decision_return"] = float(decision_return)
                        ti_current["decision_discount"] = float(decision_discount)
                        ti_current["next_state"] = next_state.tolist()
                        ti_current["done"] = bool(done)
                        ti_current["buffer_len"] = int(len(agent.memory))
                        ti_pending.append(ti_current)
                        ti_current = None
                    if formula:
                        try:
                            print(
                                f"[FORMULA][{step_tag}][DECISION_STORE] "
                                f"a={int(decision_action)}, n={int(decision_horizon)}, "
                                f"R_n={float(decision_return):.3f}, discount={float(decision_discount):.3f}"
                            )
                        except Exception:
                            pass

                if formula:
                    try:
                        s_list = [f"{float(x):.3f}" for x in state.tolist()]
                        print(
                            f"[FORMULA][{step_tag}][TRANSITION] state = [{', '.join(s_list)}]"
                        )
                        print(
                            f"[FORMULA][{step_tag}][TRANSITION] action = {int(action)}"
                        )
                        print(
                            f"[FORMULA][{step_tag}][TRANSITION] reward = {float(reward):.3f}"
                        )
                        print(
                            f"[FORMULA][{step_tag}][TRANSITION] decision_accum = {float(decision_return):.3f} (n={int(decision_horizon)})"
                        )
                    except Exception as e:
                        print(
                            f"[FORMULA][{step_tag}][WARN] transition logging failed: {e}"
                        )

                state = next_state
                total_reward += reward
                steps += 1
                inner_count += 1
                global_env_step += 1

                if done:
                    break

            if decision_updates_pending > 0:
                for update_idx in range(decision_updates_pending):
                    train_tag = f"E{episode}/T{steps}/U{update_idx + 1}"
                    if formula:
                        sep = "/" * 72
                        print(f"\n[FORMULA][{train_tag}][TRAIN_STEP] {sep}")
                    loss_value = agent.train_step(
                        formula=formula, tag=train_tag, traininfo=traininfo_enabled
                    )
                    if loss_value is not None:
                        episode_train_loss_sum += float(loss_value)
                        episode_train_loss_count += 1
                    if formula:
                        sep = "/" * 72
                        loss_str = (
                            "None" if loss_value is None else f"{float(loss_value):.3f}"
                        )
                        print(
                            f"[FORMULA][{train_tag}][TRAIN_STEP_END] loss={loss_str} {sep}"
                        )
                    if traininfo_enabled:
                        seg = ti_pending.pop(0) if ti_pending else None
                        _ti_print_segment(seg, agent.last_traininfo)

            if should_render and renderer:
                try:
                    cached_nn_output = agent.get_q_values(state)
                except Exception:
                    cached_nn_output = None
                try:
                    neuron_data = (
                        build_neuron_trace(agent, state) if neuron_mode else None
                    )
                except Exception as e:
                    neuron_data = {"error": str(e)}
                render_info = env.render_info()
                if cached_nn_output is not None:
                    render_info["nn_output"] = cached_nn_output
                render_info["last_action"] = action
                render_info["near_miss_total"] = near_miss_total
                renderer.render(
                    render_info,
                    episode,
                    total_reward,
                    agent.epsilon,
                    steps,
                    neuron_data=neuron_data,
                )

            if done:
                break

        if not running:
            break

        if traininfo_enabled:
            print("=" * 88)
            print(f"[EP{episode}] END")
            print("=" * 88)

        if formula:
            try:
                eps_before = float(agent.epsilon)

                end_tag = f"E{episode}/END"
                print(
                    f"[FORMULA][{end_tag}][EPS_DECAY] eps_new = max(eps_min, eps_old*eps_decay)"
                )
                print(
                    f"[FORMULA][{end_tag}][EPS_DECAY] eps_old={eps_before:.3f}, eps_min={float(agent.epsilon_min):.3f}, eps_decay={float(agent.epsilon_decay):.3f}"
                )
            except Exception:
                pass
        agent.decay_epsilon()
        if formula:
            try:
                end_tag = f"E{episode}/END"
                print(
                    f"[FORMULA][{end_tag}][EPS_DECAY] eps_after={float(agent.epsilon):.3f}"
                )
            except Exception:
                pass

        eps_recovered = False
        if ENABLE_EPSILON_RECOVERY:
            at_min_before = agent.epsilon <= agent.epsilon_min + 0.0001
            use_ssc_recovery = (
                abs(agent.epsilon_min - TRAIN_FINAL_MIN_EPSILON_SSC) <= 1e-12
            )
            recovery_limit = (
                CONSECUTIVE_EPSILON_RECOVERY_SSC
                if use_ssc_recovery
                else CONSECUTIVE_EPSILON_RECOVERY
            )
            if at_min_before:
                epsilon_min_streak += 1
                if epsilon_min_streak >= recovery_limit:
                    agent.epsilon = AMOUNT_EPSILON_RECOVERY
                    epsilon_min_streak = 0
                    eps_recovered = True
                    if verbose:
                        print(
                            f"\n[Epsilon Recovery] Reset epsilon to {AMOUNT_EPSILON_RECOVERY} after {recovery_limit} episodes at limit"
                        )
            else:
                epsilon_min_streak = 0
        else:
            epsilon_min_streak = 0

        if formula:
            end_tag = f"E{episode}/END"
            try:
                print(
                    f"[FORMULA][{end_tag}][EPS_RECOVERY] at_min_before={bool(at_min_before)}, streak={int(epsilon_min_streak)}/{int(recovery_limit)}, recovered={bool(eps_recovered)}, eps_after={float(agent.epsilon):.3f}"
                )
            except Exception as e:
                print(f"[FORMULA][{end_tag}][WARN] eps recovery logging failed: {e}")

        avg_episode_mse = (
            (episode_train_loss_sum / episode_train_loss_count)
            if episode_train_loss_count > 0
            else 0.0
        )

        car_y = float(info.get("car_y", 0.0))
        finish_line_y = float(getattr(env, "finish_line_y", 0.0))
        raw_progress_pct = (
            (car_y / finish_line_y) * 100.0 if finish_line_y != 0 else 0.0
        )
        rounded_progress_pct = round(raw_progress_pct / 5.0) * 5
        clamped_progress_pct = min(95, max(0, rounded_progress_pct))
        progress_percentage = (
            100 if info.get("reached_finish") else clamped_progress_pct
        )
        if formula:
            end_tag = f"E{episode}/END"
            try:
                print(
                    f"[FORMULA][{end_tag}][PROGRESS_PCT] raw=(car_y/finish_y)*100 = ({car_y:.3f}/{finish_line_y:.3f})*100 = {raw_progress_pct:.3f}"
                )
                print(
                    f"[FORMULA][{end_tag}][PROGRESS_PCT] rounded=round(raw/5)*5 = round({raw_progress_pct:.3f}/5)*5 = {rounded_progress_pct:.3f}"
                )
                print(
                    f"[FORMULA][{end_tag}][PROGRESS_PCT] final=min(95,max(0,rounded)) with finish_override={bool(info.get('reached_finish'))} => {float(progress_percentage):.3f}"
                )
            except Exception as e:
                print(f"[FORMULA][{end_tag}][WARN] progress logging failed: {e}")

        try:
            world_step = int(round(float(getattr(env, "world_distance", 0.0))))
        except Exception:
            world_step = 0
        episode_rewards.append(total_discounted_reward)
        episode_numbers.append(int(episode))
        episode_steps.append(world_step)
        episode_timeframes.append(steps)
        episode_success.append(1 if info["reached_finish"] else 0)

        episode_stages.append(current_stage)
        episode_mse.append(float(avg_episode_mse))
        buffer_size_now = int(len(agent.memory))
        episode_time_ms_now = float((time.perf_counter() - episode_start_perf) * 1000.0)

        window_size = min(50, len(episode_rewards))
        avg_reward = np.mean(episode_rewards[-window_size:])
        avg_rewards.append(avg_reward)
        if formula:
            end_tag = f"E{episode}/END"
            try:
                success_rate = float(np.mean(episode_success) * 100.0)
                finish_flag = 1 if info.get("reached_finish") else 0
                print(
                    f"[FORMULA][{end_tag}][EP_STATS] reward={float(total_discounted_reward):.4f}, timeframe={int(steps)}, step={int(world_step)}, reached_finish={finish_flag}, success_rate=mean(success)*100={success_rate:.3f}"
                )
                print(
                    f"[FORMULA][{end_tag}][AVG_REWARD] avg_reward=mean(last {int(window_size)} rewards) = {float(avg_reward):.4f}"
                )
            except Exception as e:
                print(f"[FORMULA][{end_tag}][WARN] ep stats logging failed: {e}")

        if info.get("reached_finish"):
            stage_success_streak += 1
            stage_success_count += 1
        else:
            stage_success_streak = 0
        if formula:
            end_tag = f"E{episode}/END"
            try:
                print(
                    f"[FORMULA][{end_tag}][STREAKS] stage_success_streak={int(stage_success_streak)}, stage_success_count={int(stage_success_count)}"
                )
            except Exception:
                pass

        episode_epsilons.append(epsilon_start)
        episode_progress_pct.append(progress_percentage)
        episode_streaks.append(stage_success_streak)
        episode_buffer_sizes.append(buffer_size_now)
        episode_time_ms.append(episode_time_ms_now)
        episode_close_distance_logs.append(int(close_distance_total))
        episode_near_miss_logs.append(int(near_miss_total))
        sensor_ranges = getattr(env, "sensor_ranges", [100.0] * len(SENSOR_LOG_NAMES))
        min_sensor_row = {}
        for idx, sensor_name in enumerate(SENSOR_LOG_NAMES):
            raw_min = (
                episode_sensor_mins[idx]
                if idx < len(episode_sensor_mins)
                and np.isfinite(episode_sensor_mins[idx])
                else None
            )
            sensor_max = (
                float(sensor_ranges[idx]) if idx < len(sensor_ranges) else 100.0
            )
            min_sensor_row[f"min_{sensor_name}"] = _format_min_sensor(
                raw_min, sensor_max, sensor_name
            )
        episode_sensor_min_logs.append(min_sensor_row)
        ttc_summary = summarize_ttc_samples(episode_ttc_samples)
        episode_ttc_logs.append(ttc_summary)

        is_final_stage = current_stage == num_stages - 1
        episode_had_final_valid_success = False

        if verbose:
            success_rate = np.mean(episode_success) * 100
            finish_status = "FINISH" if info["reached_finish"] else "FAILED"
            stage_info = f"Stage {current_stage + 1}/{num_stages}"
            rolling_pair = _get_completed_pair_averages(episode_rewards, episode)
            reward_str = _fmt_reward4(total_discounted_reward)
            avg_reward_str = _fmt_reward4(avg_reward)
            # Show which algorithm variant is being trained (dqn/ddqn/d3qn).
            algo_label = str(algo).upper()
            episode_log = (
                f"Episode {episode:4d} | "
                f"{stage_info} | "
                f"{algo_label} | "
                f"Reward: {reward_str} | "
                f"Avg: {avg_reward_str} | "
                f"MSE: {avg_episode_mse:.3f} | "
                f"Timeframe: {steps:4d} | "
                f"Step: {int(world_step):4d} | "
                f"Eps: {epsilon_start:.3f} | "
                f"Progress: {progress_percentage:3.0f}% | "
                f"Success: {success_rate:.1f}% | "
                f"SSC: {stage_success_count:3d} | "
                f"Streak: {max(0, stage_success_streak):2d} | "
                f"Buffer: {buffer_size_now:6d} | "
                f"min_ttc: {format_ttc_ms(ttc_summary['min_ttc'])}ms | "
                f"avg_ttc: {format_ttc_ms(ttc_summary['avg_ttc'])}ms | "
                f"Time: {episode_time_ms_now:8.2f}ms | "
                f"{finish_status}"
            )
            if is_final_stage:
                episode_log += (
                    f" | VALID: {int(independent_success_count):2d}/"
                    f"{int(INDEPENDENT_COUNT_REQ)}"
                )
            if rolling_pair is not None:
                older_avg_str = _fmt_reward4(rolling_pair["older_avg"])
                newer_avg_str = _fmt_reward4(rolling_pair["newer_avg"])
                episode_log += (
                    f" | {rolling_pair['older_label']}: {older_avg_str}"
                    f" | {rolling_pair['newer_label']}: {newer_avg_str}"
                )
            print(episode_log)
            if rolling_pair is not None and rolling_pair["pair_changed_now"]:
                print(
                    f"Stability: {_fmt_reward4(rolling_pair['stability_gap'])} "
                    f"({rolling_pair['newer_label']} - {rolling_pair['older_label']})"
                )

        if formula:
            end_tag = f"E{episode}/END"
            try:
                print(
                    f"[FORMULA][{end_tag}][STAGE_STATE] current_stage={int(current_stage) + 1}/{int(num_stages)}, is_final_stage={bool(is_final_stage)}, independent_based={bool(INDEPENDENT_BASED)}"
                )
                print(
                    f"[FORMULA][{end_tag}][STAGE_STATE] success_based_req={int(SUCCESS_BASED_REQ)}, consecutive_stage_req={int(CONSECUTIVE_STAGE_REQ)}, independent_count_req={int(INDEPENDENT_COUNT_REQ)}, tester_mode={bool(ValidationTesterMode)}"
                )
            except Exception as e:
                print(f"[FORMULA][{end_tag}][WARN] stage state logging failed: {e}")

        if not is_final_stage and not INDEPENDENT_BASED:
            if stage_success_count >= SUCCESS_BASED_REQ:
                stage_model_path = _build_stage_model_path(
                    save_dir=save_dir,
                    stage_number=current_stage + 1,
                    episode_number=None,
                )
                if formula:
                    end_tag = f"E{episode}/END"
                    try:
                        print(
                            f"[FORMULA][{end_tag}][MODEL_SAVE] reason=stage_checkpoint stage={int(current_stage) + 1} path={stage_model_path}"
                        )
                    except Exception:
                        pass
                _save_training_checkpoint(
                    stage_model_path,
                    reason="stage_checkpoint",
                    episode_number=episode,
                    epsilon_start_value=epsilon_start,
                    stage_idx_for_resume=current_stage + 1,
                    stage_success_streak_value=0,
                    stage_success_count_value=0,
                    independent_success_count_value=0,
                    epsilon_min_streak_value=epsilon_min_streak,
                )

                old_stage = current_stage
                current_stage += 1
                stage_success_streak = 0
                stage_success_count = 0
                independent_success_count = 0
                _sync_stage_epsilon_min(current_stage, reason="stage advance")

                epsilon_before = float(agent.epsilon)
                epsilon_reset_applied = False
                if float(NEW_STAGE_EPSILON) != 0.0:
                    agent.epsilon = float(NEW_STAGE_EPSILON)
                    epsilon_reset_applied = True

                env.set_curriculum_stage(current_stage)

                if formula:
                    end_tag = f"E{episode}/END"
                    try:
                        print(
                            f"[FORMULA][{end_tag}][STAGE_ADVANCE] {int(old_stage) + 1} -> {int(current_stage) + 1}, epsilon_reset_applied={bool(epsilon_reset_applied)}, epsilon_before={epsilon_before:.3f}, epsilon_after={float(agent.epsilon):.3f}"
                        )
                    except Exception:
                        pass

                if renderer:
                    renderer.env = env

                if verbose:
                    if save_models:
                        print(f"[OK] Saved {stage_model_path}")
                    print("\n" + "=" * 60)
                    print(
                        f"[OK] ADVANCED TO STAGE {current_stage + 1}/{num_stages} (SUCCESS-BASED)"
                    )
                    if epsilon_reset_applied:
                        print(f"[OK] Epsilon reset to {NEW_STAGE_EPSILON}")
                    else:
                        print(f"[OK] Epsilon continues at {agent.epsilon:.4f}")
                    print("=" * 60 + "\n")
        else:
            skip_final_independent_run = (
                stage_success_streak >= CONSECUTIVE_STAGE_REQ
                and is_final_stage
                and not INDRUN_FINAL_STAGE
            )
            if stage_success_streak >= CONSECUTIVE_STAGE_REQ and not skip_final_independent_run:
                if formula:
                    end_tag = f"E{episode}/END"
                    try:
                        next_stage_str = (
                            "FINAL"
                            if is_final_stage
                            else f"{int(current_stage) + 2}/{int(num_stages)}"
                        )
                        print(
                            f"[FORMULA][{end_tag}][INDEP_TEST] start for next_stage={next_stage_str} (streak={int(stage_success_streak)}/{int(CONSECUTIVE_STAGE_REQ)})"
                        )
                    except Exception:
                        pass
                if verbose:
                    print("\n" + "=" * 60)
                    if is_final_stage:
                        print("FINAL STAGE INDEPENDENT RUN: Testing mastery")
                    else:
                        print(
                            f"INDEPENDENT RUN: Testing qualification for Stage {current_stage + 2}"
                        )
                    print("=" * 60)

                independent_success = run_independent_test(
                    env=env,
                    agent=agent,
                    max_steps=max_steps_per_episode,
                    renderer=renderer,
                    visualize=visualize,
                    verbose=verbose,
                    step_multiplier=valid_multiplier,
                    neuron_mode=neuron_mode,
                )
                if formula:
                    end_tag = f"E{episode}/END"
                    try:
                        print(
                            f"[FORMULA][{end_tag}][INDEP_TEST] result={'SUCCESS' if independent_success else 'FAIL'}"
                        )
                    except Exception:
                        pass

                if independent_success:
                    validation_passed = True
                    tester_failed_stage = None
                    tester_total_stages = 0

                    if is_final_stage and ValidationTesterMode:
                        independent_model_path = _build_independent_model_path(
                            save_dir=save_dir,
                            stage_number=current_stage + 1,
                            episode_number=episode,
                        )
                        if formula:
                            end_tag = f"E{episode}/END"
                            try:
                                print(
                                    f"[FORMULA][{end_tag}][MODEL_SAVE] reason=independent_success_checkpoint stage={int(current_stage) + 1} path={independent_model_path}"
                                )
                            except Exception:
                                pass
                        _save_training_checkpoint(
                            independent_model_path,
                            reason="independent_success_checkpoint",
                            episode_number=episode,
                            epsilon_start_value=epsilon_start,
                            stage_idx_for_resume=current_stage,
                            stage_success_streak_value=stage_success_streak,
                            stage_success_count_value=stage_success_count,
                            independent_success_count_value=independent_success_count,
                            epsilon_min_streak_value=epsilon_min_streak,
                        )
                        if verbose and save_models:
                            print(
                                "[OK] Independent success checkpoint saved: "
                                f"{independent_model_path}"
                            )

                        validation_passed, tester_failed_stage, tester_total_stages = (
                            run_tester_validation(
                                agent=agent,
                                max_steps=max_steps_per_episode,
                                verbose=verbose,
                                save_dir=save_dir,
                                step_multiplier=valid_multiplier,
                                continue_training=continue_training,
                            )
                        )
                        if formula:
                            end_tag = f"E{episode}/END"
                            try:
                                fail_stage_repr = (
                                    int(tester_failed_stage)
                                    if tester_failed_stage is not None
                                    else "None"
                                )
                                print(
                                    f"[FORMULA][{end_tag}][TESTER_VALIDATION] result={'SUCCESS' if validation_passed else 'FAIL'} failed_stage={fail_stage_repr} total={int(tester_total_stages)}"
                                )
                            except Exception:
                                pass

                    if validation_passed:
                        episode_number_for_filename = (
                            episode if is_final_stage else None
                        )
                        stage_model_path = _build_stage_model_path(
                            save_dir=save_dir,
                            stage_number=current_stage + 1,
                            episode_number=episode_number_for_filename,
                        )
                        if formula:
                            end_tag = f"E{episode}/END"
                            try:
                                print(
                                    f"[FORMULA][{end_tag}][MODEL_SAVE] reason=stage_checkpoint stage={int(current_stage) + 1} path={stage_model_path}"
                                )
                            except Exception:
                                pass
                        _save_training_checkpoint(
                            stage_model_path,
                            reason="stage_checkpoint",
                            episode_number=episode,
                            epsilon_start_value=epsilon_start,
                            stage_idx_for_resume=(
                                current_stage if is_final_stage else current_stage + 1
                            ),
                            stage_success_streak_value=(
                                stage_success_streak if is_final_stage else 0
                            ),
                            stage_success_count_value=(
                                stage_success_count if is_final_stage else 0
                            ),
                            independent_success_count_value=(
                                independent_success_count + 1
                                if is_final_stage
                                else 0
                            ),
                            epsilon_min_streak_value=epsilon_min_streak,
                        )

                        if not is_final_stage:
                            old_stage = current_stage
                            current_stage += 1
                            stage_success_streak = 0
                            stage_success_count = 0
                            independent_success_count = 0
                            _sync_stage_epsilon_min(
                                current_stage, reason="stage advance"
                            )

                            epsilon_before = float(agent.epsilon)
                            epsilon_reset_applied = False
                            if float(NEW_STAGE_EPSILON) != 0.0:
                                agent.epsilon = float(NEW_STAGE_EPSILON)
                                epsilon_reset_applied = True

                            env.set_curriculum_stage(current_stage)

                            if formula:
                                end_tag = f"E{episode}/END"
                                try:
                                    print(
                                        f"[FORMULA][{end_tag}][STAGE_ADVANCE] {int(old_stage) + 1} -> {int(current_stage) + 1}, epsilon_reset_applied={bool(epsilon_reset_applied)}, epsilon_before={epsilon_before:.3f}, epsilon_after={float(agent.epsilon):.3f}"
                                    )
                                except Exception:
                                    pass

                            if renderer:
                                renderer.env = env

                            if verbose:
                                if save_models:
                                    print(f"[OK] Saved {stage_model_path}")
                                print("\n" + "=" * 60)
                                print(
                                    f"[OK] ADVANCED TO STAGE {current_stage + 1}/{num_stages}"
                                )
                                if epsilon_reset_applied:
                                    print(f"[OK] Epsilon reset to {NEW_STAGE_EPSILON}")
                                else:
                                    print(
                                        f"[OK] Epsilon continues at {agent.epsilon:.4f}"
                                    )
                                print("=" * 60 + "\n")
                        else:
                            episode_had_final_valid_success = True
                            independent_success_count += 1

                            if formula:
                                end_tag = f"E{episode}/END"
                                try:
                                    print(
                                        f"[FORMULA][{end_tag}][FINAL_STAGE] independent_success_count={int(independent_success_count)}/{int(INDEPENDENT_COUNT_REQ)}"
                                    )
                                except Exception:
                                    pass

                            if verbose:
                                if save_models:
                                    print(f"[OK] Saved {stage_model_path}")
                                print(
                                    f"[OK] FINAL STAGE INDEPENDENT SUCCESS "
                                    f"({independent_success_count}/{INDEPENDENT_COUNT_REQ})"
                                )
                                print(
                                    "[VALID] Final-stage validated success count: "
                                    f"{independent_success_count}/{INDEPENDENT_COUNT_REQ}"
                                )

                            if independent_success_count >= INDEPENDENT_COUNT_REQ:
                                if verbose:
                                    print("\n" + "=" * 60)
                                    print(
                                        f"[OK] INDEPENDENT COUNT REACHED ({INDEPENDENT_COUNT_REQ})"
                                    )
                                    print("  Saving logs and ending training.")
                                    print("=" * 60 + "\n")
                                stop_training = True
                    else:
                        if verbose:
                            print("\n" + "=" * 60)
                            print(
                                "[FAIL] FINAL STAGE TESTER VALIDATION FAILED "
                                f"(stage {tester_failed_stage}/{tester_total_stages})"
                            )
                            print("  Continuing training at final stage")
                            print("=" * 60 + "\n")
                else:
                    if verbose:
                        print("\n" + "=" * 60)
                        if is_final_stage:
                            print("[FAIL] FINAL STAGE INDEPENDENT RUN FAILED")
                            print("  Continuing training at final stage")
                        else:
                            print(
                                f"[FAIL] INDEPENDENT RUN FAILED - Continuing Stage {current_stage + 1}"
                            )
                            print(f"  Epsilon continues at {agent.epsilon:.4f}")
                        print("=" * 60 + "\n")
            elif skip_final_independent_run:
                if formula:
                    end_tag = f"E{episode}/END"
                    try:
                        print(
                            f"[FORMULA][{end_tag}][INDEP_TEST] skipped final-stage independent run because INDRUN_FINAL_STAGE=False"
                        )
                    except Exception:
                        pass
                if verbose:
                    print("\n" + "=" * 60)
                    print(
                        "[OK] FINAL STAGE SUCCESS - independent run disabled; continuing training"
                    )
                    print("=" * 60 + "\n")

        if (not stop_training) and should_stop_on_episode_target_valid(
            episode=episode,
            num_episodes=episode_target,
            episode_had_final_valid_success=episode_had_final_valid_success,
        ):
            if END_EXACT and not episode_had_final_valid_success:
                target_model_path = _build_stage_model_path(
                    save_dir=save_dir,
                    stage_number=current_stage + 1,
                    episode_number=episode,
                    checkpoint=True,
                )
                _save_training_checkpoint(
                    target_model_path,
                    reason="episode_target_checkpoint",
                    episode_number=episode,
                    epsilon_start_value=epsilon_start,
                )
                if verbose and save_models:
                    print(f"[OK] Episode target checkpoint saved: {target_model_path}")
            if verbose:
                print("\n" + "=" * 60)
                if END_EXACT and not episode_had_final_valid_success:
                    print(
                        f"[OK] EXACT EPISODE TARGET REACHED at episode {episode} "
                        f"(target {episode_target})."
                    )
                else:
                    print(
                        f"[OK] TARGET VALID REACHED at episode {episode} "
                        f"(target {episode_target})."
                    )
                print("  Saving logs and ending training.")
                print("=" * 60 + "\n")
            stop_training = True

        should_interval_save = False
        try:
            should_interval_save = int(save_interval) > 0 and episode % int(save_interval) == 0
        except Exception:
            should_interval_save = False
        if should_interval_save and not episode_had_final_valid_success:
            interval_model_path = _build_stage_model_path(
                save_dir=save_dir,
                stage_number=current_stage + 1,
                episode_number=episode,
                checkpoint=True,
            )
            _save_training_checkpoint(
                interval_model_path,
                reason="interval_checkpoint",
                episode_number=episode,
                epsilon_start_value=epsilon_start,
            )
            if verbose and save_models:
                print(f"[OK] Interval checkpoint saved: {interval_model_path}")

        if stop_training:
            break

        if verbose and episode % render_interval == 0:
            print(f"\n--- Episode {episode} Details ---")
            print(f"Car final position: ({info['car_x']:.1f}, {info['car_y']:.1f})")
            print(f"Car final angle: {info['car_angle']:.1f} deg")
            print(f"Reached finish: {info['reached_finish']}")
            print(f"Memory size: {len(agent.memory)}")
            print("-" * 40 + "\n")

        if episode % 50 == 0 or episode == episode_target:
            try:
                if formula:
                    end_tag = f"E{episode}/END"
                    try:
                        reason = (
                            "every_50"
                            if (episode % 50 == 0)
                            else "episode_target_checkpoint"
                        )
                        print(
                            f"[FORMULA][{end_tag}][CSV_SAVE] reason={reason} path={csv_log_path}"
                        )
                    except Exception:
                        pass
                with open(csv_log_path, "w", newline="") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
                    writer.writeheader()

                    for ep_idx in range(len(episode_rewards)):
                        if csv_float_decimals is None:
                            reward_str = _fmt_reward4(episode_rewards[ep_idx])
                            avg_reward_str = (
                                _fmt_reward4(avg_rewards[ep_idx])
                                if ep_idx < len(avg_rewards)
                                else _fmt_reward4(0.0)
                            )
                            mse_str = (
                                f"{episode_mse[ep_idx]:.3f}"
                                if ep_idx < len(episode_mse)
                                else "0.000"
                            )
                            epsilon_str = f"{episode_epsilons[ep_idx]:.3f}"
                            success_rate_str = (
                                f"{(np.mean(episode_success[: ep_idx + 1]) * 100):.2f}"
                            )
                            time_ms_str = f"{episode_time_ms[ep_idx]:.2f}"
                        else:
                            d = int(csv_float_decimals)
                            reward_str = _fmt_reward4(episode_rewards[ep_idx])
                            avg_reward_str = (
                                _fmt_reward4(avg_rewards[ep_idx])
                                if ep_idx < len(avg_rewards)
                                else _fmt_reward4(0.0)
                            )
                            mse_str = (
                                f"{float(episode_mse[ep_idx]):.{d}f}"
                                if ep_idx < len(episode_mse)
                                else f"{0.0:.{d}f}"
                            )
                            epsilon_str = f"{float(episode_epsilons[ep_idx]):.{d}f}"
                            success_rate_str = f"{float(np.mean(episode_success[: ep_idx + 1]) * 100):.{d}f}"
                            time_ms_str = f"{float(episode_time_ms[ep_idx]):.{d}f}"

                        ep_num = episode_numbers[ep_idx] if ep_idx < len(episode_numbers) else ep_idx + 1

                        ep_stage = min(episode_stages[ep_idx], num_stages - 1)

                        writer.writerow(
                            {
                                "episode": ep_num,
                                "stage": ep_stage + 1,
                                "reward": reward_str,
                                "avg_reward": avg_reward_str,
                                "MSE": mse_str,
                                "timeframe": episode_timeframes[ep_idx],
                                "steps": episode_steps[ep_idx],
                                "close distance": episode_close_distance_logs[ep_idx],
                                "near_miss": episode_near_miss_logs[ep_idx],
                                "epsilon": epsilon_str,
                                "progress_pct": episode_progress_pct[ep_idx],
                                "success_rate": success_rate_str,
                                "Buffer": episode_buffer_sizes[ep_idx],
                                "time_ms": time_ms_str,
                                "streak": episode_streaks[ep_idx],
                                "reached_finish": episode_success[ep_idx],
                                "min_R2": episode_sensor_min_logs[ep_idx].get(
                                    "min_R2", "None"
                                ),
                                "min_R1": episode_sensor_min_logs[ep_idx].get(
                                    "min_R1", "None"
                                ),
                                "min_F": episode_sensor_min_logs[ep_idx].get(
                                    "min_F", "None"
                                ),
                                "min_L1": episode_sensor_min_logs[ep_idx].get(
                                    "min_L1", "None"
                                ),
                                "min_L2": episode_sensor_min_logs[ep_idx].get(
                                    "min_L2", "None"
                                ),
                                "min_SR": episode_sensor_min_logs[ep_idx].get(
                                    "min_SR", "None"
                                ),
                                "min_SL": episode_sensor_min_logs[ep_idx].get(
                                    "min_SL", "None"
                                ),
                                "min_ttc": format_ttc_ms(
                                    episode_ttc_logs[ep_idx].get("min_ttc")
                                ),
                                "avg_ttc": format_ttc_ms(
                                    episode_ttc_logs[ep_idx].get("avg_ttc")
                                ),
                            }
                        )

                if verbose and episode % 50 == 0:
                    print(f"[OK] CSV log updated: {csv_log_path}")
            except Exception as e:
                if verbose:
                    print(f"Warning: Could not save CSV log: {e}")

    try:
        with open(csv_log_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
            writer.writeheader()

            for ep_idx in range(len(episode_rewards)):
                if csv_float_decimals is None:
                    reward_str = _fmt_reward4(episode_rewards[ep_idx])
                    avg_reward_str = (
                        _fmt_reward4(avg_rewards[ep_idx])
                        if ep_idx < len(avg_rewards)
                        else _fmt_reward4(0.0)
                    )
                    mse_str = (
                        f"{episode_mse[ep_idx]:.3f}"
                        if ep_idx < len(episode_mse)
                        else "0.000"
                    )
                    epsilon_str = f"{episode_epsilons[ep_idx]:.3f}"
                    success_rate_str = (
                        f"{(np.mean(episode_success[: ep_idx + 1]) * 100):.2f}"
                    )
                    time_ms_str = f"{episode_time_ms[ep_idx]:.2f}"
                else:
                    d = int(csv_float_decimals)
                    reward_str = _fmt_reward4(episode_rewards[ep_idx])
                    avg_reward_str = (
                        _fmt_reward4(avg_rewards[ep_idx])
                        if ep_idx < len(avg_rewards)
                        else _fmt_reward4(0.0)
                    )
                    mse_str = (
                        f"{float(episode_mse[ep_idx]):.{d}f}"
                        if ep_idx < len(episode_mse)
                        else f"{0.0:.{d}f}"
                    )
                    epsilon_str = f"{float(episode_epsilons[ep_idx]):.{d}f}"
                    success_rate_str = (
                        f"{float(np.mean(episode_success[: ep_idx + 1]) * 100):.{d}f}"
                    )
                    time_ms_str = f"{float(episode_time_ms[ep_idx]):.{d}f}"

                ep_num = episode_numbers[ep_idx] if ep_idx < len(episode_numbers) else ep_idx + 1
                ep_stage = min(episode_stages[ep_idx], num_stages - 1)

                writer.writerow(
                    {
                        "episode": ep_num,
                        "stage": ep_stage + 1,
                        "reward": reward_str,
                        "avg_reward": avg_reward_str,
                        "MSE": mse_str,
                        "timeframe": episode_timeframes[ep_idx],
                        "steps": episode_steps[ep_idx],
                        "close distance": episode_close_distance_logs[ep_idx],
                        "near_miss": episode_near_miss_logs[ep_idx],
                        "epsilon": epsilon_str,
                        "progress_pct": episode_progress_pct[ep_idx],
                        "success_rate": success_rate_str,
                        "Buffer": episode_buffer_sizes[ep_idx],
                        "time_ms": time_ms_str,
                        "streak": episode_streaks[ep_idx],
                        "reached_finish": episode_success[ep_idx],
                        "min_R2": episode_sensor_min_logs[ep_idx].get("min_R2", "None"),
                        "min_R1": episode_sensor_min_logs[ep_idx].get("min_R1", "None"),
                        "min_F": episode_sensor_min_logs[ep_idx].get("min_F", "None"),
                        "min_L1": episode_sensor_min_logs[ep_idx].get("min_L1", "None"),
                        "min_L2": episode_sensor_min_logs[ep_idx].get("min_L2", "None"),
                        "min_SR": episode_sensor_min_logs[ep_idx].get("min_SR", "None"),
                        "min_SL": episode_sensor_min_logs[ep_idx].get("min_SL", "None"),
                        "min_ttc": format_ttc_ms(
                            episode_ttc_logs[ep_idx].get("min_ttc")
                        ),
                        "avg_ttc": format_ttc_ms(
                            episode_ttc_logs[ep_idx].get("avg_ttc")
                        ),
                    }
                )

        print(f"\n[OK] Final CSV log saved: {csv_log_path}")
    except Exception as e:
        print(f"Warning: Could not save final CSV log: {e}")

    if close_renderer and renderer:
        renderer.close()
        renderer = None

    if plot_results:
        plot_training_results(
            episode_rewards, avg_rewards, episode_success, agent.loss_history
        )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Stage: {current_stage + 1}/{num_stages}")
    print(f"Final success rate (all episodes): {np.mean(episode_success) * 100:.1f}%")
    if save_models:
        print("SAVED MODEL AND TRAINING LOGS!")
    else:
        print("SAVED TRAINING LOGS!")
    print(" ")
    print("[ENDING] CONGRATULATIONS on the final stage and you MASTERED IT!  OK OK OK")
    print("=" * 60)

    return agent, episode_rewards, episode_success, renderer, running


def train_dqn(
    num_episodes=500,
    max_steps_per_episode=999999,
    save_interval=SAVE_MODEL_CHECKPOINT,
    render_interval=100,
    verbose=True,
    visualize=False,
    render_every_n_episodes=1,
    load_model_path=None,
    continue_training=False,
    start_episode=None,
    start_epsilon=None,
    current_stage=None,
    fast_multiply=False,
    multi_valid=False,
    formula=False,
    traininfo=False,
    seed=None,
    memory_size=MEMORY_SIZE,
    neuron_mode=False,
    algo=ALGO_DQN,
):
    """Train one curriculum-learning run, optionally resuming from a checkpoint."""

    agent, rewards, success, _renderer, _running = _train_dqn_run(
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        save_interval=save_interval,
        render_interval=render_interval,
        verbose=verbose,
        visualize=visualize,
        render_every_n_episodes=render_every_n_episodes,
        load_model_path=load_model_path,
        continue_training=continue_training,
        start_episode=start_episode,
        start_epsilon=start_epsilon,
        current_stage=current_stage,
        fast_multiply=fast_multiply,
        multi_valid=multi_valid,
        formula=formula,
        traininfo=traininfo,
        seed=seed,
        memory_size=memory_size,
        renderer=None,
        close_renderer=True,
        plot_results=True,
        neuron_mode=neuron_mode,
        algo=algo,
    )
    return agent, rewards, success


def _get_tester_stage_csv_path(save_dir: str, continue_training: bool = False) -> str:
    return os.path.join(
        save_dir, _continue_csv_name(TESTER_STAGE_CSV_NAME, continue_training)
    )


def _read_tester_stage_fail_counts(csv_path: str) -> dict:
    fail_counts = {}
    if not os.path.exists(csv_path):
        return fail_counts

    with open(csv_path, "r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                stage = int(row.get("stage", 0))
                fail_count = int(row.get("fail_count", 0))
            except Exception:
                continue
            if stage <= 0:
                continue
            fail_counts[stage] = max(0, fail_count)

    return fail_counts


def _write_tester_stage_fail_counts(
    csv_path: str, total_tester_stages: int, fail_counts: dict
) -> None:
    directory = os.path.dirname(csv_path) or "."
    os.makedirs(directory, exist_ok=True)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=TESTER_STAGE_CSV_HEADERS)
        writer.writeheader()
        for stage in range(1, int(total_tester_stages) + 1):
            writer.writerow(
                {"stage": stage, "fail_count": max(0, int(fail_counts.get(stage, 0)))}
            )


def _ensure_tester_stage_csv(
    save_dir: str, total_tester_stages: int, continue_training: bool = False
) -> str:
    csv_path = _get_tester_stage_csv_path(save_dir, continue_training)
    fail_counts = _read_tester_stage_fail_counts(csv_path)
    _write_tester_stage_fail_counts(
        csv_path=csv_path,
        total_tester_stages=total_tester_stages,
        fail_counts=fail_counts,
    )
    return csv_path


def _increment_tester_stage_fail_count(
    save_dir: str,
    total_tester_stages: int,
    failed_stage: int,
    continue_training: bool = False,
) -> str:
    csv_path = _ensure_tester_stage_csv(
        save_dir=save_dir,
        total_tester_stages=total_tester_stages,
        continue_training=continue_training,
    )

    stage = int(failed_stage)
    if stage < 1 or stage > int(total_tester_stages):
        return csv_path

    fail_counts = _read_tester_stage_fail_counts(csv_path)
    fail_counts[stage] = int(fail_counts.get(stage, 0)) + 1
    _write_tester_stage_fail_counts(
        csv_path=csv_path,
        total_tester_stages=total_tester_stages,
        fail_counts=fail_counts,
    )
    return csv_path


def run_tester_validation(
    agent,
    max_steps,
    verbose=True,
    save_dir="models",
    step_multiplier=1,
    continue_training=False,
):
    """
    Run epsilon=0 tester validation across every TEST_OBSTACLES stage.
    step_multiplier executes multiple env steps per loop (like key '1').

    Returns:
        Tuple: (all_passed, failed_stage_1_based_or_none, total_tester_stages)
    """
    total_tester_stages = get_num_stages(TEST_OBSTACLES)
    if total_tester_stages <= 0:
        return True, None, 0

    tester_stage_csv_path = _get_tester_stage_csv_path(save_dir, continue_training)
    try:
        tester_stage_csv_path = _ensure_tester_stage_csv(
            save_dir=save_dir,
            total_tester_stages=total_tester_stages,
            continue_training=continue_training,
        )
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not initialize tester stage CSV: {e}")

    if verbose:
        print("  ValidationTesterMode enabled: running TEST_OBSTACLES...")
        print(f"  Tester stage fail log: {tester_stage_csv_path}")

    for tester_stage in range(total_tester_stages):
        tester_env = CarEnvironment(
            curriculum_stage=tester_stage, obstacles_config=TEST_OBSTACLES
        )
        stage_success = run_independent_test(
            env=tester_env,
            agent=agent,
            max_steps=max_steps,
            renderer=None,
            visualize=False,
            verbose=False,
            step_multiplier=step_multiplier,
        )

        if verbose:
            result = "SUCCESS" if stage_success else "FAILED"
            print(
                f"    TEST_OBSTACLE {tester_stage + 1}/{total_tester_stages}: {result}"
            )

        if not stage_success:
            try:
                _increment_tester_stage_fail_count(
                    save_dir=save_dir,
                    total_tester_stages=total_tester_stages,
                    failed_stage=tester_stage + 1,
                    continue_training=continue_training,
                )
            except Exception as e:
                if verbose:
                    print(f"  Warning: Could not update tester stage CSV: {e}")
            return False, tester_stage + 1, total_tester_stages

    return True, None, total_tester_stages


def run_independent_test(
    env,
    agent,
    max_steps,
    renderer=None,
    visualize=False,
    verbose=True,
    step_multiplier=1,
    neuron_mode=False,
):
    """
    Run a single INDEPENDENT episode with epsilon=0 and no training.
    This is used to test if the agent qualifies to advance to the next curriculum stage.

    IMPORTANT: This does NOT affect training at all:
    - No episode counter increment
    - No reward recording
    - No model saving
    - No experience storage
    - No epsilon decay

    Args:
        env: The CarEnvironment instance
        agent: The DQNAgent instance
        max_steps: Maximum steps for the test run
        renderer: Optional GameRenderer instance
        visualize: Whether to render the test
        verbose: Whether to print progress
        step_multiplier: Execute multiple env steps per render/loop (like key '1')
        neuron_mode: Show neural-network forward-pass details in visualization

    Returns:
        True if the agent reached the finish line, False otherwise
    """
    if neuron_mode and visualize:
        from main_visualize import build_neuron_trace

    original_epsilon = agent.epsilon
    agent.epsilon = 0.0

    state = env.reset()
    done = False
    steps = 0

    try:
        step_multiplier = int(step_multiplier)
    except Exception:
        step_multiplier = 1
    if step_multiplier < 1:
        step_multiplier = 1

    if verbose:
        if step_multiplier > 1:
            print(
                f"  Running INDEPENDENT test (epsilon=0, no training, speed x{step_multiplier})..."
            )
        else:
            print("  Running INDEPENDENT test (epsilon=0, no training)...")

    current_action = agent.select_action(state, training=False)

    while not done and steps < max_steps:
        steps_this_frame = step_multiplier
        for _ in range(steps_this_frame):
            if done or steps >= max_steps:
                break

            is_decision_step = steps % DECISION_INTERVAL == 0
            if is_decision_step:
                current_action = agent.select_action(state, training=False)
            action = current_action

            next_state, reward, done, info = env.step(
                action, apply_steering=is_decision_step
            )

            state = next_state
            steps += 1

        if visualize and renderer:
            try:
                neuron_data = build_neuron_trace(agent, state) if neuron_mode else None
            except Exception as e:
                neuron_data = {"error": str(e)}
            render_info = env.render_info()
            renderer.render(
                render_info,
                0,
                0,
                0,
                steps,
                paused=False,
                neuron_data=neuron_data,
            )

    try:
        world_step = int(round(float(getattr(env, "world_distance", 0.0))))
    except Exception:
        world_step = 0

    agent.epsilon = original_epsilon

    success = info.get("reached_finish", False)
    if verbose:
        result = "SUCCESS" if success else "FAILED"
        print(
            f"  INDEPENDENT RUN Result: {result} (Timeframe: {steps}, Step: {world_step})"
        )

    return success


def plot_training_results(rewards, avg_rewards, success, loss_history):
    """Plot training results (no-op; matplotlib usage removed)"""
    pass


def _safe_experiment_label(text: str) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if not s:
        return ""
    keep = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        elif ch.isspace():
            keep.append("_")
    return "".join(keep).strip("_")


def _read_episode_log_csv(csv_path: str):
    rows = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _compute_metrics_from_episode_log(rows, last_n: int = 50):
    if not rows:
        return {
            "episodes": 0,
            "final_success_rate": 0.0,
            "last_n": int(last_n),
            "last_n_success_rate": 0.0,
            "last_n_avg_reward": 0.0,
            "last_n_avg_mse": 0.0,
            "max_stage": 0,
            "episode_reached_max_stage": None,
        }

    def _to_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    def _to_int(x, default=0):
        try:
            return int(float(x))
        except Exception:
            return int(default)

    def _to_bool(x):
        if isinstance(x, bool):
            return x
        s = str(x).strip().lower()
        return s in ("1", "true", "t", "yes", "y")

    rewards = [_to_float(r.get("reward", 0.0)) for r in rows]
    mses = [_to_float(r.get("MSE", 0.0)) for r in rows]
    stages = [_to_int(r.get("stage", 0)) for r in rows]
    finishes = [_to_bool(r.get("reached_finish", False)) for r in rows]

    n_total = len(rows)
    window = max(1, min(int(last_n), n_total))
    last_rewards = rewards[-window:]
    last_mses = mses[-window:]
    last_finishes = finishes[-window:]

    max_stage = max(stages) if stages else 0
    ep_reach_max = None
    if max_stage > 0:
        for idx, st in enumerate(stages, start=1):
            if st == max_stage:
                ep_reach_max = idx
                break

    return {
        "episodes": n_total,
        "final_success_rate": float(np.mean(finishes)) if finishes else 0.0,
        "last_n": window,
        "last_n_success_rate": float(np.mean(last_finishes)) if last_finishes else 0.0,
        "last_n_avg_reward": float(np.mean(last_rewards)) if last_rewards else 0.0,
        "last_n_avg_mse": float(np.mean(last_mses)) if last_mses else 0.0,
        "max_stage": int(max_stage),
        "episode_reached_max_stage": ep_reach_max,
    }


def fine_tune_memory_size(
    memory_sizes,
    runs_per_size: int = 3,
    episodes_per_run: int = 200,
    last_n: int = 50,
    base_seed: int = 123,
    output_root: str = "tuning",
    label: str = "",
    max_steps_per_episode: int = 999999,
    fast_multiply: bool = False,
    multi_valid: bool = False,
    verbose: bool = False,
):
    """Fine-tune replay buffer size (memory_size) and save logs for a paper."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label_safe = _safe_experiment_label(label)
    exp_name = f"fine_tune_memory_{timestamp}" + (
        f"_{label_safe}" if label_safe else ""
    )
    exp_dir = os.path.join(output_root, exp_name)
    runs_dir = os.path.join(exp_dir, "runs")
    os.makedirs(runs_dir, exist_ok=True)

    memory_sizes_list = [int(x) for x in memory_sizes]
    total_runs = int(len(memory_sizes_list) * int(runs_per_size))
    print(
        "[TUNE] Fine-tuning mode: memory_size sweep\n"
        f"[TUNE] memory_sizes={memory_sizes_list} | runs_per_size={int(runs_per_size)} | episodes_per_run={int(episodes_per_run)}\n"
        f"[TUNE] total_runs={total_runs} | fast_multiply={bool(fast_multiply)} | multi_valid={bool(multi_valid)} | last_n_window={int(last_n)}\n"
        f"[TUNE] output_dir={exp_dir}",
        flush=True,
    )

    config = {
        "type": "fine_tune_memory_size",
        "timestamp": timestamp,
        "label": label,
        "command_argv": sys.argv,
        "memory_sizes": [int(x) for x in memory_sizes],
        "runs_per_size": int(runs_per_size),
        "episodes_per_run": int(episodes_per_run),
        "last_n_window": int(last_n),
        "base_seed": int(base_seed),
        "max_steps_per_episode": int(max_steps_per_episode),
        "fast_multiply": bool(fast_multiply),
        "multi_valid": bool(multi_valid),
        "fixed_hparams": {
            "learning_rate": float(LEARNING_RATE),
            "gamma": float(GAMMA),
            "epsilon_start": float(TRAIN_MAX_EPSILON),
            "epsilon_min": float(TRAIN_MIN_EPSILON),
            "epsilon_min_final_stage": float(TRAIN_FINAL_MIN_EPSILON),
            "epsilon_min_final_stage_ssc": float(TRAIN_FINAL_MIN_EPSILON_SSC),
            "epsilon_decay": float(EPSILON_DECAY),
            "batch_size": int(BATCH_SIZE),
            "target_update_freq": int(TARGET_UPDATE_FREQ),
        },
    }

    def _fmt3(x) -> str:
        return f"{float(x):.3f}"

    def _format_floats_recursive(obj):
        if isinstance(obj, float):
            return _fmt3(obj)
        if isinstance(obj, dict):
            return {k: _format_floats_recursive(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_format_floats_recursive(v) for v in obj]
        return obj

    with open(os.path.join(exp_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(_format_floats_recursive(config), f, indent=2)

    run_rows = []

    run_counter = 0
    for mem_index, mem_size in enumerate(memory_sizes_list):
        for run_index in range(int(runs_per_size)):
            run_counter += 1
            seed = int(base_seed) + (mem_index * 1000) + run_index
            run_id = f"mem{int(mem_size)}_run{run_index + 1}_seed{seed}"
            run_dir = os.path.join(runs_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            episode_log_csv = os.path.join(run_dir, "episode_log.csv")

            print(
                f"[TUNE] Run {run_counter}/{total_runs} | memory_size={int(mem_size)} | run={int(run_index + 1)}/{int(runs_per_size)} | seed={int(seed)}",
                flush=True,
            )
            t0 = time.perf_counter()
            train_error = None
            try:
                _train_dqn_run(
                    num_episodes=int(episodes_per_run),
                    max_steps_per_episode=int(max_steps_per_episode),
                    save_interval=999999,
                    render_interval=50,
                    verbose=bool(verbose),
                    visualize=False,
                    render_every_n_episodes=1,
                    load_model_path=None,
                    fast_multiply=bool(fast_multiply),
                    multi_valid=bool(multi_valid),
                    formula=False,
                    seed=seed,
                    memory_size=int(mem_size),
                    save_dir=run_dir,
                    save_models=False,
                    csv_log_path=episode_log_csv,
                    csv_float_decimals=3,
                    renderer=None,
                    close_renderer=True,
                    plot_results=False,
                )
            except Exception as e:
                train_error = str(e)
            duration_s = time.perf_counter() - t0

            metrics_error = None
            try:
                rows = _read_episode_log_csv(episode_log_csv)
                metrics = _compute_metrics_from_episode_log(rows, last_n=int(last_n))
            except Exception as e:
                metrics_error = str(e)
                metrics = _compute_metrics_from_episode_log([], last_n=int(last_n))

            run_summary = {
                "memory_size": int(mem_size),
                "run": int(run_index + 1),
                "seed": int(seed),
                "episodes": int(metrics.get("episodes", 0)),
                "max_stage": int(metrics.get("max_stage", 0)),
                "episode_reached_max_stage": metrics.get(
                    "episode_reached_max_stage", None
                ),
                "final_success_rate": float(metrics.get("final_success_rate", 0.0)),
                "last_n": int(metrics.get("last_n", int(last_n))),
                "last_n_success_rate": float(metrics.get("last_n_success_rate", 0.0)),
                "last_n_avg_reward": float(metrics.get("last_n_avg_reward", 0.0)),
                "last_n_avg_mse": float(metrics.get("last_n_avg_mse", 0.0)),
                "duration_seconds": float(duration_s),
                "episode_log_csv": episode_log_csv,
                "train_error": train_error,
                "metrics_error": metrics_error,
            }
            run_rows.append(run_summary)

            run_summary_out = dict(run_summary)
            run_summary_out["final_success_rate"] = _fmt3(
                run_summary_out["final_success_rate"]
            )
            run_summary_out["last_n_success_rate"] = _fmt3(
                run_summary_out["last_n_success_rate"]
            )
            run_summary_out["last_n_avg_reward"] = _fmt3(
                run_summary_out["last_n_avg_reward"]
            )
            run_summary_out["last_n_avg_mse"] = _fmt3(run_summary_out["last_n_avg_mse"])
            run_summary_out["duration_seconds"] = _fmt3(
                run_summary_out["duration_seconds"]
            )

            with open(
                os.path.join(run_dir, "run_summary.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(run_summary_out, f, indent=2)

            if train_error:
                print(
                    f"[TUNE] Run {run_counter}/{total_runs} FAILED | error={train_error}",
                    flush=True,
                )
            else:
                print(
                    f"[TUNE] Run {run_counter}/{total_runs} OK | "
                    f"duration={duration_s:.1f}s | "
                    f"final_success_rate={run_summary['final_success_rate']:.3f} | "
                    f"last{int(run_summary['last_n'])}_success_rate={run_summary['last_n_success_rate']:.3f} | "
                    f"max_stage={int(run_summary['max_stage'])}",
                    flush=True,
                )

    runs_csv_path = os.path.join(exp_dir, "runs_summary.csv")
    if run_rows:
        fieldnames = list(run_rows[0].keys())
        with open(runs_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in run_rows:
                row_out = dict(row)
                row_out["final_success_rate"] = _fmt3(row_out["final_success_rate"])
                row_out["last_n_success_rate"] = _fmt3(row_out["last_n_success_rate"])
                row_out["last_n_avg_reward"] = _fmt3(row_out["last_n_avg_reward"])
                row_out["last_n_avg_mse"] = _fmt3(row_out["last_n_avg_mse"])
                row_out["duration_seconds"] = _fmt3(row_out["duration_seconds"])
                writer.writerow(row_out)

    def _mean_std(values):
        arr = np.array(values, dtype=np.float64)
        return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    agg_rows = []
    for mem_size in memory_sizes_list:
        subset = [
            r
            for r in run_rows
            if int(r["memory_size"]) == int(mem_size) and not r.get("train_error")
        ]
        if not subset:
            agg_rows.append(
                {
                    "memory_size": int(mem_size),
                    "n_runs": 0,
                    "final_success_rate_mean": _fmt3(0.0),
                    "final_success_rate_std": _fmt3(0.0),
                    "last_n_success_rate_mean": _fmt3(0.0),
                    "last_n_success_rate_std": _fmt3(0.0),
                    "last_n_avg_reward_mean": _fmt3(0.0),
                    "last_n_avg_reward_std": _fmt3(0.0),
                    "last_n_avg_mse_mean": _fmt3(0.0),
                    "last_n_avg_mse_std": _fmt3(0.0),
                    "max_stage_mean": _fmt3(0.0),
                    "max_stage_std": _fmt3(0.0),
                    "duration_seconds_mean": _fmt3(0.0),
                    "duration_seconds_std": _fmt3(0.0),
                }
            )
            continue

        fs_mean, fs_std = _mean_std([r["final_success_rate"] for r in subset])
        ln_mean, ln_std = _mean_std([r["last_n_success_rate"] for r in subset])
        rw_mean, rw_std = _mean_std([r["last_n_avg_reward"] for r in subset])
        mse_mean, mse_std = _mean_std([r["last_n_avg_mse"] for r in subset])
        ms_mean, ms_std = _mean_std([r["max_stage"] for r in subset])
        dur_mean, dur_std = _mean_std([r["duration_seconds"] for r in subset])

        agg_rows.append(
            {
                "memory_size": int(mem_size),
                "n_runs": int(len(subset)),
                "final_success_rate_mean": _fmt3(fs_mean),
                "final_success_rate_std": _fmt3(fs_std),
                "last_n_success_rate_mean": _fmt3(ln_mean),
                "last_n_success_rate_std": _fmt3(ln_std),
                "last_n_avg_reward_mean": _fmt3(rw_mean),
                "last_n_avg_reward_std": _fmt3(rw_std),
                "last_n_avg_mse_mean": _fmt3(mse_mean),
                "last_n_avg_mse_std": _fmt3(mse_std),
                "max_stage_mean": _fmt3(ms_mean),
                "max_stage_std": _fmt3(ms_std),
                "duration_seconds_mean": _fmt3(dur_mean),
                "duration_seconds_std": _fmt3(dur_std),
            }
        )

    agg_csv_path = os.path.join(exp_dir, "aggregate_by_memory_size.csv")
    if agg_rows:
        fieldnames = list(agg_rows[0].keys())
        with open(agg_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in agg_rows:
                writer.writerow(row)

    print(f"[TUNE] Saved: {runs_csv_path}", flush=True)
    print(f"[TUNE] Saved: {agg_csv_path}", flush=True)
    return exp_dir


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DQN Car Training with Curriculum Learning"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=500,
        help="Number of training episodes (default: 500)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable real-time pygame visualization during training",
    )
    parser.add_argument(
        "--neuron",
        action="store_true",
        help="Show detailed neural-network forward-pass panel while training visualization is running",
    )
    parser.add_argument(
        "--render-every",
        type=int,
        default=1,
        help="Render every N episodes when visualizing (default: 1)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=SAVE_MODEL_CHECKPOINT,
        help=(
            "Save *_check.pth checkpoint every N episodes; set to 0 to disable "
            f"(default: {SAVE_MODEL_CHECKPOINT})."
        ),
    )
    parser.add_argument(
        "--no-verbose", action="store_true", help="Disable verbose training output"
    )
    parser.add_argument(
        "--loadmodel",
        type=str,
        default=None,
        dest="model",
        help="Backward-compatible alias for --model",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        dest="model",
        help="Path to model checkpoint for --continue",
    )
    parser.add_argument(
        "--continue",
        action="store_true",
        dest="continue_train",
        help="Resume training from --model using checkpoint metadata; --epistart, --epsilon, and --curstage are optional overrides",
    )
    parser.add_argument(
        "--epistart",
        type=int,
        default=None,
        help="Optional episode override for --continue; checkpoint next_episode is used when omitted",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Optional epsilon override for --continue; checkpoint epsilon is used when omitted",
    )
    parser.add_argument(
        "--curstage",
        type=int,
        default=None,
        help="Optional 1-based curriculum stage override for --continue",
    )
    parser.add_argument(
        "--multiply",
        action="store_true",
        help="Speed up training by batching TRAIN_MULTIPLIER env steps per NN training step",
    )
    parser.add_argument(
        "--multivalid",
        action="store_true",
        help="Speed up INDEPENDENT/tester validation runs using KEYONE_MULTIPLIER steps per loop (like key '1')",
    )
    parser.add_argument(
        "--formula",
        action="store_true",
        help="Print detailed formula calculations during training (VERY verbose)",
    )
    parser.add_argument(
        "--traininfo",
        action="store_true",
        help="Print per-decision-interval training info (readable verbose log)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible training runs (default: None)",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=MEMORY_SIZE,
        help=f"Replay buffer capacity (default: {MEMORY_SIZE})",
    )

    parser.add_argument(
        "--fine-tune-memory",
        action="store_true",
        help="Run fine-tuning sweep over replay buffer memory_size (saves under ./tuning)",
    )
    parser.add_argument(
        "--tune-memory-sizes",
        nargs="+",
        type=int,
        default=[10000, 25000, 50000, MEMORY_SIZE, 200000],
        help="Memory sizes to sweep (default: 10k 25k 50k 100k 200k)",
    )
    parser.add_argument(
        "--tune-runs",
        type=int,
        default=3,
        help="Number of runs (different seeds) per memory_size (default: 3)",
    )
    parser.add_argument(
        "--tune-episodes",
        type=int,
        default=200,
        help="Episodes per tuning run (default: 200)",
    )
    parser.add_argument(
        "--tune-last-n",
        type=int,
        default=50,
        help="Window size for 'last-N' metrics in summaries (default: 50)",
    )
    parser.add_argument(
        "--tune-base-seed",
        type=int,
        default=123,
        help="Base seed; actual seed = base + mem_index*1000 + run_index (default: 123)",
    )
    parser.add_argument(
        "--tune-dir",
        type=str,
        default="tuning",
        help="Output root directory for fine-tuning results (default: tuning)",
    )
    parser.add_argument(
        "--tune-label",
        type=str,
        default="",
        help="Optional label appended to tuning folder name (default: '')",
    )
    parser.add_argument(
        "--tune-verbose",
        action="store_true",
        help="Verbose per-episode logs during fine-tuning (default: off)",
    )
    algo_group = parser.add_mutually_exclusive_group()
    algo_group.add_argument(
        "--ddqn",
        action="store_true",
        help="Use Double DQN (van Hasselt et al., 2016) instead of vanilla DQN",
    )
    algo_group.add_argument(
        "--d3qn",
        action="store_true",
        help="Use Dueling Double DQN (D3QN: Wang et al., 2016 + Double) instead of vanilla DQN",
    )

    args = parser.parse_args()

    if args.neuron and not args.visualize:
        print("[INFO] --neuron requested; enabling --visualize.")
        args.visualize = True

    # Resolve algorithm choice
    if args.continue_train and not args.model:
        parser.error("--continue requires --model <path/model.pth>")

    if args.d3qn:
        algo = "d3qn"
    elif args.ddqn:
        algo = "ddqn"
    else:
        algo = _infer_checkpoint_algo(args.model) if args.continue_train else None
        if algo not in ("dqn", "ddqn", "d3qn"):
            algo = "dqn"

    if args.fine_tune_memory:
        exp_dir = fine_tune_memory_size(
            memory_sizes=args.tune_memory_sizes,
            runs_per_size=int(args.tune_runs),
            episodes_per_run=int(args.tune_episodes),
            last_n=int(args.tune_last_n),
            base_seed=int(args.tune_base_seed),
            output_root=str(args.tune_dir),
            label=str(args.tune_label),
            max_steps_per_episode=999999,
            fast_multiply=bool(args.multiply),
            multi_valid=bool(args.multivalid),
            verbose=bool(args.tune_verbose),
        )
        print(f"[OK] Fine-tuning complete. Results saved under: {exp_dir}")
        raise SystemExit(0)

    agent, rewards, success = train_dqn(
        num_episodes=args.episodes,
        max_steps_per_episode=999999,
        save_interval=args.save_interval,
        render_interval=50,
        verbose=not args.no_verbose,
        visualize=args.visualize,
        render_every_n_episodes=args.render_every,
        load_model_path=args.model,
        continue_training=args.continue_train,
        start_episode=args.epistart,
        start_epsilon=args.epsilon,
        current_stage=args.curstage,
        fast_multiply=args.multiply,
        multi_valid=args.multivalid,
        formula=args.formula,
        traininfo=args.traininfo,
        seed=args.seed,
        memory_size=args.memory_size,
        neuron_mode=args.neuron,
        algo=algo,
    )
