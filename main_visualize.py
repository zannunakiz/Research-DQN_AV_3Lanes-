"""
Pygame Visualization for DQN Car Navigation
Real-time rendering with scrolling road effect
Single-window display: Road + two-column Info panel
"""

import pygame
import math
import os
import csv
import re
import random
import time
import numpy as np

from main_environment import CarEnvironment, get_num_stages
from main_dqn_agent import DQNAgent, ALGO_DQN, SUPPORTED_ALGOS
from func_scale import to_kmh
from func_ttc import calculate_env_ttc_ms, format_ttc_ms, summarize_ttc_samples

try:
    from main_constant import (
        SCREEN_HEIGHT,
        CAR_STATIC_Y_POS,
        DEFAULT_SCALE,
        OBSTACLES,
        TEST_OBSTACLES,
        ALLSTAGE_CONSECUTIVE_REQ,
        KEYONE_MULTIPLIER,
        USE_PNG,
        LANE_CENTER_REWARD_WIDTH,
        SHOW_CENTERLANE_REWARD_INDICATOR,
        CENTERLANE_REWARD_INDICATOR_COLOR,
        LEFT_LR_OFFSETX,
        RIGHT_LR_OFFSETX,
        CENTER_LR_OFFSETX,
        DECISION_INTERVAL,
        MEMORY_SIZE,
        CAR_MAX_SPEED,
        EPSILON_DECAY,
        LEARNING_RATE,
        GAMMA,
        BATCH_SIZE,
        TARGET_UPDATE_FREQ,
        TRAIN_MAX_EPSILON,
        TRAIN_MIN_EPSILON,
        FONT_TITLE,
        FONT_SUBTITLE,
        FONT_TEXT,
        startRandom,
        gapRandom,
        maxRandom,
        visualize_logs_sec,
    )
except ImportError:
    SCREEN_HEIGHT = 600
    CAR_STATIC_Y_POS = 150
    DEFAULT_SCALE = 1.5
    OBSTACLES = [[]]
    TEST_OBSTACLES = [[]]
    ALLSTAGE_CONSECUTIVE_REQ = 2
    KEYONE_MULTIPLIER = 5
    USE_PNG = False
    DECISION_INTERVAL = 10
    MEMORY_SIZE = 100000
    CAR_MAX_SPEED = 3.2444444444444445
    EPSILON_DECAY = 0.998
    LEARNING_RATE = 0.001
    GAMMA = 0.99
    BATCH_SIZE = 64
    TARGET_UPDATE_FREQ = 10
    TRAIN_MAX_EPSILON = 1.0
    TRAIN_MIN_EPSILON = 0.001
    FONT_TITLE = 24
    FONT_SUBTITLE = 20
    FONT_TEXT = 18
    startRandom = 400
    gapRandom = 125
    maxRandom = 50
    visualize_logs_sec = 2


WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
DARK_GRAY = (50, 50, 50)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 100, 255)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)
BRIGHT_PURPLE = (210, 120, 255)
PANEL_BG = (28, 31, 36)
PANEL_BG_ALT = (28, 31, 36)
PANEL_HEADER = (19, 22, 26)
PANEL_BORDER = (92, 102, 116)
TEXT_MUTED = (190, 198, 210)

VISUALIZE_LOG_DIR = "visualize_logs"
VISUALIZE_LOG_BASENAME = "visualize"
EVALUATE_LOG_NAME = "evaluate.csv"
VISUALIZE_LOG_PATTERN = re.compile(r"^visualize-(\d+)\.csv$")
EVALUATE_MODEL_PATTERN = re.compile(r"-(\d+)\.pth$", re.IGNORECASE)
NEURON_INPUT_NAMES = ["R2", "R1", "F", "L1", "L2", "SR", "SL", "Speed"]
NEURON_ACTION_NAMES = ["SlowL", "SlowS", "SlowR", "FastL", "FastS", "FastR"]
PX_PER_METER = 14.0
SENSOR_LOG_NAMES = ["R2", "R1", "F", "L1", "L2", "SR", "SL"]
SENSOR_LOG_OFFSETS = {"R2": 20.0, "R1": 20.0, "F": 20.0, "L1": 20.0, "L2": 20.0, "SR": 10.0, "SL": 10.0}


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


def build_neuron_trace(
    agent,
    state,
    max_neurons_per_hidden_layer=6,
    max_inputs_per_neuron=8,
):
    """Build a compact forward-pass trace from normalized input to Q-values."""
    state_array = np.asarray(state, dtype=np.float64).reshape(-1)
    activations = state_array
    linear_layers = [
        module
        for module in agent.policy_net.network
        if hasattr(module, "weight") and hasattr(module, "bias")
    ]

    trace_layers = []
    total_params = 0
    for layer_index, module in enumerate(linear_layers, start=1):
        weights = module.weight.detach().cpu().numpy().astype(np.float64, copy=False)
        bias = module.bias.detach().cpu().numpy().astype(np.float64, copy=False)
        total_params += int(weights.size + bias.size)

        pre_activation = weights.dot(activations) + bias
        is_output_layer = layer_index == len(linear_layers)
        post_activation = (
            pre_activation
            if is_output_layer
            else np.maximum(pre_activation, 0.0)
        )

        if is_output_layer:
            selected_indices = list(range(len(pre_activation)))
        else:
            ranked_indices = np.argsort(np.abs(post_activation))[::-1]
            selected_indices = [
                int(idx) for idx in ranked_indices[: int(max_neurons_per_hidden_layer)]
            ]

        neuron_rows = []
        for neuron_index in selected_indices:
            products = weights[neuron_index] * activations
            ranked_inputs = np.argsort(np.abs(products))[::-1]
            contribution_rows = []
            for input_index in ranked_inputs[: int(max_inputs_per_neuron)]:
                input_index = int(input_index)
                input_name = (
                    NEURON_INPUT_NAMES[input_index]
                    if layer_index == 1 and input_index < len(NEURON_INPUT_NAMES)
                    else f"a{layer_index - 1}[{input_index}]"
                )
                contribution_rows.append(
                    {
                        "input_index": input_index,
                        "input_name": input_name,
                        "weight": float(weights[neuron_index, input_index]),
                        "input": float(activations[input_index]),
                        "product": float(products[input_index]),
                    }
                )

            output_label = (
                NEURON_ACTION_NAMES[neuron_index]
                if is_output_layer and neuron_index < len(NEURON_ACTION_NAMES)
                else f"n{neuron_index}"
            )
            neuron_rows.append(
                {
                    "index": int(neuron_index),
                    "label": output_label,
                    "bias": float(bias[neuron_index]),
                    "z": float(pre_activation[neuron_index]),
                    "activation": float(post_activation[neuron_index]),
                    "contributions": contribution_rows,
                }
            )

        trace_layers.append(
            {
                "index": int(layer_index),
                "type": "output" if is_output_layer else "hidden",
                "input_size": int(weights.shape[1]),
                "output_size": int(weights.shape[0]),
                "weight_shape": tuple(int(v) for v in weights.shape),
                "bias_shape": tuple(int(v) for v in bias.shape),
                "activation_min": float(np.min(post_activation)),
                "activation_max": float(np.max(post_activation)),
                "activation_mean": float(np.mean(post_activation)),
                "active_count": int(np.sum(post_activation > 0.0))
                if not is_output_layer
                else None,
                "neurons": neuron_rows,
                "q_values": post_activation.tolist() if is_output_layer else None,
            }
        )

        activations = post_activation

    q_values = trace_layers[-1]["q_values"] if trace_layers else []
    return {
        "input": state_array.tolist(),
        "input_names": NEURON_INPUT_NAMES[: len(state_array)],
        "layers": trace_layers,
        "q_values": q_values,
        "total_params": int(total_params),
    }


class RandomObstacleGenerator:
    """Generate finite random obstacle rows for --random visualization."""

    def __init__(
        self,
        start_y=startRandom,
        gap_y=gapRandom,
        min_vehicles_per_row=1,
        max_vehicles_per_row=2,
        max_rows=maxRandom,
        lookahead_y=1500,
        cleanup_behind_y=300,
        rng=None,
    ):
        self.start_y = float(start_y)
        self.gap_y = float(gap_y)
        self.min_vehicles_per_row = int(min_vehicles_per_row)
        self.max_vehicles_per_row = int(max_vehicles_per_row)
        self.max_rows = int(max(1, max_rows))
        self.lookahead_y = float(lookahead_y)
        self.cleanup_behind_y = float(cleanup_behind_y)
        self.rng = rng if rng is not None else random
        self.next_spawn_y = self.start_y
        self.rows_spawned = 0
        self.total_obstacles_spawned = 0

    def reset(self):
        self.next_spawn_y = self.start_y
        self.rows_spawned = 0
        self.total_obstacles_spawned = 0

    def _choose_row_lanes(self, lane_count):
        lane_count = int(max(1, lane_count))
        min_count = max(1, min(self.min_vehicles_per_row, lane_count))
        max_count = max(min_count, min(self.max_vehicles_per_row, lane_count))
        vehicle_count = self.rng.randint(min_count, max_count)
        return sorted(self.rng.sample(range(lane_count), vehicle_count))

    def build_next_row_configs(self, lane_count):
        lanes = self._choose_row_lanes(lane_count)
        return [{"lane": lane, "y": self.next_spawn_y} for lane in lanes]

    def build_all_configs(self, lane_count):
        configs = []
        self.reset()
        while self.rows_spawned < self.max_rows:
            row_configs = self.build_next_row_configs(lane_count)
            configs.extend(row_configs)
            self.rows_spawned += 1
            self.total_obstacles_spawned += len(row_configs)
            self.next_spawn_y += self.gap_y
        return configs

    def append_all_obstacles(self, env):
        configs = self.build_all_configs(env.lane_count)
        return env.append_obstacles(configs)

    def append_due_obstacles(self, env):
        added = 0
        while (
            self.rows_spawned < self.max_rows
            and self.next_spawn_y <= float(env.car_y) + self.lookahead_y
        ):
            row_configs = self.build_next_row_configs(env.lane_count)
            added += env.append_obstacles(row_configs)
            self.rows_spawned += 1
            self.total_obstacles_spawned += len(row_configs)
            self.next_spawn_y += self.gap_y

        self.cleanup_obstacles(env)
        return added

    def cleanup_obstacles(self, env):
        cutoff_y = float(env.car_y) - self.cleanup_behind_y
        before_count = len(env.obstacles)
        env.obstacles = [
            obs for obs in env.obstacles if float(obs.get("y", 0.0)) >= cutoff_y
        ]
        return before_count - len(env.obstacles)


def get_next_visualize_csv_path(log_dir=VISUALIZE_LOG_DIR):
    """Create log directory if needed and return next visualize-<N>.csv path."""
    abs_log_dir = os.path.abspath(log_dir)
    os.makedirs(abs_log_dir, exist_ok=True)

    max_index = 0
    for entry in os.listdir(abs_log_dir):
        match = VISUALIZE_LOG_PATTERN.match(entry)
        if not match:
            continue
        try:
            max_index = max(max_index, int(match.group(1)))
        except Exception:
            continue

    next_index = max_index + 1
    filename = f"{VISUALIZE_LOG_BASENAME}-{next_index}.csv"
    return os.path.join(abs_log_dir, filename)


def get_evaluate_csv_path(log_dir=VISUALIZE_LOG_DIR):
    """Create log directory if needed and return fixed evaluate.csv path."""
    abs_log_dir = os.path.abspath(log_dir)
    os.makedirs(abs_log_dir, exist_ok=True)
    return os.path.join(abs_log_dir, EVALUATE_LOG_NAME)


def get_visualize_csv_headers(include_model=False):
    """Return visualize/evaluate CSV headers in the canonical order."""
    headers = [
        "episode",
        "close distance",
        "near_miss",
        "MSE",
        "Reward",
        "Avg reward",
        "time",
        "timeframe",
        "steps",
        "progress_pct",
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
    if include_model:
        return ["model"] + headers
    return headers


def extract_model_number(model_path):
    """Return numeric suffix after the last '-' for eligible .pth models."""
    basename = os.path.basename(str(model_path))
    if "_check" in basename or not basename.lower().endswith(".pth"):
        return None
    match = EVALUATE_MODEL_PATTERN.search(basename)
    if not match:
        return None
    return int(match.group(1))


def get_evaluable_model_paths(models_dir="models"):
    """Return non-checkpoint model paths sorted newest-to-oldest by suffix number."""
    abs_models_dir = os.path.abspath(models_dir)
    if not os.path.isdir(abs_models_dir):
        return []

    model_entries = []
    for entry in os.listdir(abs_models_dir):
        path = os.path.join(abs_models_dir, entry)
        if not os.path.isfile(path):
            continue
        model_number = extract_model_number(entry)
        if model_number is None:
            continue
        model_entries.append((model_number, path))

    model_entries.sort(key=lambda item: (item[0], os.path.basename(item[1])), reverse=True)
    return model_entries


def build_min_sensor_logs(env, episode_sensor_mins):
    """Format per-sensor minimum distances for visualize-compatible CSV rows."""
    sensor_ranges = getattr(env, "sensor_ranges", [100.0] * len(SENSOR_LOG_NAMES))
    return {
        f"min_{sensor_name}": _format_min_sensor(
            episode_sensor_mins[idx]
            if idx < len(episode_sensor_mins) and np.isfinite(episode_sensor_mins[idx])
            else None,
            float(sensor_ranges[idx]) if idx < len(sensor_ranges) else 100.0,
            sensor_name,
        )
        for idx, sensor_name in enumerate(SENSOR_LOG_NAMES)
    }


def build_visualize_episode_row(
    episode: int,
    close_distance: int,
    mse: float,
    reward: float,
    avg_reward: float,
    time_ms: int,
    timeframe: int,
    steps: int,
    min_ttc=None,
    avg_ttc=None,
    min_sensor_logs: dict = None,
    near_miss: int = 0,
    progress_pct: float = 0.0,
    reached_finish: int = 0,
    model=None,
):
    """Build one visualize CSV row with fixed formatting."""
    row = {}
    if model is not None:
        row["model"] = int(model) if str(model).isdigit() else str(model)

    row.update({
        "episode": int(episode),
        "close distance": int(close_distance),
        "near_miss": int(near_miss),
        "MSE": f"{float(mse):.3f}",
        "Reward": f"{float(reward):.3f}",
        "Avg reward": f"{float(avg_reward):.3f}",
        "time": int(time_ms),
        "timeframe": int(timeframe),
        "steps": int(steps),
        "progress_pct": float(progress_pct),
        "reached_finish": int(reached_finish),
        "min_ttc": format_ttc_ms(min_ttc),
        "avg_ttc": format_ttc_ms(avg_ttc),
    })
    min_sensor_logs = min_sensor_logs or {}
    for sensor_name in SENSOR_LOG_NAMES:
        row[f"min_{sensor_name}"] = min_sensor_logs.get(f"min_{sensor_name}", "None")
    return row


class ExperimentObstaclePlanner:
    """State container for experiment obstacle controls."""

    LANE_LABEL_TO_INDEX = {"left": 0, "center": 1, "right": 2}
    LANE_INDEX_TO_LABEL = {0: "left", 1: "center", 2: "right"}

    def __init__(self):
        self.selected_lanes = set()
        self.distance = 125
        self.to_spawn_list = []

    def toggle_lane(self, lane_label):
        lane = str(lane_label).strip().lower()
        if lane not in self.LANE_LABEL_TO_INDEX:
            return False
        if lane in self.selected_lanes:
            self.selected_lanes.remove(lane)
        else:
            self.selected_lanes.add(lane)
        return True

    def increment_distance(self):
        self.distance += 5
        return self.distance

    def decrement_distance(self):
        self.distance = max(5, self.distance - 5)
        return self.distance

    def can_add(self):
        return len(self.selected_lanes) > 0

    def add_current_selection(self):
        if not self.can_add():
            return False
        ordered_lanes = sorted(
            self.selected_lanes, key=lambda name: self.LANE_LABEL_TO_INDEX[name]
        )
        self.to_spawn_list.append({"lanes": ordered_lanes, "distance": self.distance})
        return True

    def clear_spawn_list(self):
        self.to_spawn_list = []

    def build_spawn_plan(self, current_car_y):
        """Build cumulative spawn Y values from current car position."""
        if not self.to_spawn_list:
            return []

        plan = []
        previous_y = None
        for entry in self.to_spawn_list:
            distance = float(entry["distance"])
            if previous_y is None:
                spawn_y = float(current_car_y) + 200.0 + distance
            else:
                spawn_y = previous_y + distance
            lanes = list(entry["lanes"])
            plan.append({"lanes": lanes, "distance": distance, "spawn_y": spawn_y})
            previous_y = spawn_y
        return plan

    def build_obstacle_configs(self, current_car_y):
        configs = []
        for row in self.build_spawn_plan(current_car_y):
            for lane_label in row["lanes"]:
                lane_index = self.LANE_LABEL_TO_INDEX.get(lane_label)
                if lane_index is not None:
                    configs.append({"lane": lane_index, "y": row["spawn_y"]})
        return configs

    def snapshot(self):
        return {
            "selected_lanes": sorted(
                self.selected_lanes, key=lambda name: self.LANE_LABEL_TO_INDEX[name]
            ),
            "distance": self.distance,
            "to_spawn_list": [
                {"lanes": list(entry["lanes"]), "distance": int(entry["distance"])}
                for entry in self.to_spawn_list
            ],
        }


class GameRenderer:
    """Pygame renderer for car environment (single window with 3 sections)."""

    def __init__(
        self,
        env,
        scale=DEFAULT_SCALE,
        experiment_mode=False,
        neuron_mode=False,
    ):
        pygame.init()

        self.env = env
        self.scale = scale
        self.experiment_mode = bool(experiment_mode)
        self.neuron_mode = bool(neuron_mode)


        self.road_display_width = int(env.road_width * scale)
        self.road_display_height = SCREEN_HEIGHT


        self.car_screen_y = CAR_STATIC_Y_POS


        self.info_left_width = 320
        self.info_right_width = 320
        self.info_panel_width = self.info_left_width + self.info_right_width
        self.experiment_panel_width = 360 if self.experiment_mode else 0
        self.neuron_panel_width = 760 if self.neuron_mode else 0
        self.window_width = (
            self.road_display_width
            + self.info_panel_width
            + self.experiment_panel_width
            + self.neuron_panel_width
        )
        self.window_height = self.road_display_height
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        if self.experiment_mode and self.neuron_mode:
            pygame.display.set_caption(
                "DQN Car Navigation - Road + Info + Obstacle + Neuron Trace"
            )
        elif self.experiment_mode:
            pygame.display.set_caption(
                "DQN Car Navigation - Road + Info + Obstacle Controls (Experiment)"
            )
        elif self.neuron_mode:
            pygame.display.set_caption("DQN Car Navigation - Road + Info + Neuron Trace")
        else:
            pygame.display.set_caption("DQN Car Navigation - Road + Info")

        self.clock = pygame.time.Clock()
        self.base_fps = 60
        self.speed_multiplier = 1
        self.render_fps_multiplier = 1.0
        self.speed_mode = "normal"
        self.show_sensor_tip_labels = True
        self.show_near_miss_indicator = True
        self.font_text = pygame.font.Font(None, int(FONT_TEXT))
        self.font_subtitle = pygame.font.Font(None, int(FONT_SUBTITLE))
        self.font_title = pygame.font.Font(None, int(FONT_TITLE))
        self.font = self.font_text
        self.font_large = self.font_title


        self.car_img = None
        self.obstacle_img = None
        if USE_PNG:
            try:
                self.car_img = pygame.image.load("car.png").convert_alpha()
                self.obstacle_img = self.car_img.copy()
            except (FileNotFoundError, pygame.error) as e:
                print(
                    f"Warning: Could not load car.png ({e}). Falling back to rectangle rendering."
                )
                self.car_img = None
                self.obstacle_img = None


        self.last_render_info = None
        self.last_episode = 0
        self.last_total_reward = 0
        self.last_epsilon = 0
        self.last_timeframe = 0
        self.last_paused = False
        self.last_experiment_data = None
        self.last_neuron_data = None
        self.experiment_button_rects = {}

    def _set_speed_mode(self, mode):
        mode_str = str(mode).strip().lower()
        if mode_str not in {"normal", "fast", "slow"}:
            mode_str = "normal"

        self.speed_mode = mode_str
        if self.speed_mode == "fast":
            self.speed_multiplier = int(max(1, KEYONE_MULTIPLIER))
            self.render_fps_multiplier = 1.0
        elif self.speed_mode == "slow":
            self.speed_multiplier = 1
            self.render_fps_multiplier = 0.25
        else:
            self.speed_multiplier = 1
            self.render_fps_multiplier = 1.0

    def toggle_speed(self):
        """Toggle fast simulation mode (key '1')."""
        if self.speed_mode == "fast":
            self._set_speed_mode("normal")
        else:
            self._set_speed_mode("fast")
        return self.speed_multiplier

    def toggle_slow_motion(self):
        """Toggle slow-motion mode (key '2')."""
        if self.speed_mode == "slow":
            self._set_speed_mode("normal")
        else:
            self._set_speed_mode("slow")
        return self.get_speed_mode_label()

    def toggle_indicators(self):
        next_state = not bool(self.show_sensor_tip_labels)
        self.show_sensor_tip_labels = next_state
        self.show_near_miss_indicator = next_state
        return next_state

    def get_speed_mode_label(self):
        if self.speed_mode == "fast":
            return f"FAST x{int(self.speed_multiplier)}"
        if self.speed_mode == "slow":
            return "SLOW x0.25"
        return "NORMAL x1"

    def get_steps_per_frame(self):
        return int(max(1, self.speed_multiplier))

    def get_effective_render_fps(self):
        return int(max(1, round(float(self.base_fps) * float(self.render_fps_multiplier))))

    def get_camera_offset(self, car_y):
        """Calculate camera offset for scrolling effect"""
        return car_y - (CAR_STATIC_Y_POS / self.scale)

    def world_to_screen(self, x, y, camera_offset):
        """Convert world coordinates to screen coordinates with camera offset"""
        screen_x = int(x * self.scale)
        adjusted_y = y - camera_offset
        screen_y = int(self.road_display_height - adjusted_y * self.scale)
        return screen_x, screen_y

    def draw_road(self, camera_offset, finish_line_y):
        """Draw the 3-lane road with scrolling effect"""
        pygame.draw.rect(
            self.screen,
            DARK_GRAY,
            (0, 0, self.road_display_width, self.road_display_height),
        )


        try:
            if SHOW_CENTERLANE_REWARD_INDICATOR:

                for i in range(self.env.lane_count):
                    lane_center_world = (i * self.env.lane_width) + (
                        self.env.lane_width / 2
                    )

                    try:
                        if i == 0:
                            lane_center_world = float(lane_center_world) + float(
                                LEFT_LR_OFFSETX
                            )
                        elif i == 1:
                            lane_center_world = float(lane_center_world) + float(
                                CENTER_LR_OFFSETX
                            )
                        elif i == 2:
                            lane_center_world = float(lane_center_world) + float(
                                RIGHT_LR_OFFSETX
                            )
                    except Exception:
                        lane_center_world = (i * self.env.lane_width) + (
                            self.env.lane_width / 2
                        )
                    rect_width_px = int(LANE_CENTER_REWARD_WIDTH * self.scale)
                    rect_height_px = self.road_display_height
                    rect_x = int(lane_center_world * self.scale) - rect_width_px // 2
                    rect_y = 0

                    overlay = pygame.Surface(
                        (rect_width_px, rect_height_px), pygame.SRCALPHA
                    )
                    overlay.fill(CENTERLANE_REWARD_INDICATOR_COLOR)
                    self.screen.blit(overlay, (rect_x, rect_y))
        except NameError:

            pass

        lane_positions = self.env.get_lane_positions()
        for lane_x in lane_positions:
            screen_x = int(lane_x * self.scale)
            dash_length = 30
            gap_length = 20
            world_y_start = camera_offset
            world_y_end = camera_offset + (self.road_display_height / self.scale)

            y_world = int(world_y_start / (dash_length + gap_length)) * (
                dash_length + gap_length
            )
            while y_world < world_y_end + (dash_length + gap_length):
                _, screen_y_start = self.world_to_screen(0, y_world, camera_offset)
                _, screen_y_end = self.world_to_screen(
                    0, y_world + dash_length, camera_offset
                )

                if -10 < screen_y_start < self.road_display_height + 10:
                    pygame.draw.line(
                        self.screen,
                        WHITE,
                        (screen_x, max(0, screen_y_start)),
                        (screen_x, min(self.road_display_height, screen_y_end)),
                        2,
                    )
                y_world += dash_length + gap_length

        pygame.draw.line(self.screen, YELLOW, (0, 0), (0, self.road_display_height), 4)
        pygame.draw.line(
            self.screen,
            YELLOW,
            (self.road_display_width - 2, 0),
            (self.road_display_width - 2, self.road_display_height),
            4,
        )

        if not self.experiment_mode and math.isfinite(float(finish_line_y)):
            _, finish_screen_y = self.world_to_screen(0, finish_line_y, camera_offset)
            if -20 < finish_screen_y < self.road_display_height + 20:
                gradient_rows = [
                    ((35, 110, 55), 5),
                    ((55, 155, 75), 3),
                    ((90, 210, 110), 1),
                    ((55, 155, 75), -1),
                    ((35, 110, 55), -3),
                ]
                for color, offset in gradient_rows:
                    y = finish_screen_y + offset
                    if 0 <= y < self.road_display_height:
                        pygame.draw.line(
                            self.screen,
                            color,
                            (0, y),
                            (self.road_display_width, y),
                            2,
                        )

        _, start_screen_y = self.world_to_screen(0, 50, camera_offset)
        if -10 < start_screen_y < self.road_display_height + 10:
            pygame.draw.line(
                self.screen,
                GREEN,
                (0, start_screen_y),
                (self.road_display_width, start_screen_y),
                2,
            )

    def draw_obstacle_car(
        self, obs_x, obs_y, obs_width, obs_height, camera_offset, color=RED
    ):
        """Draw an obstacle car (PNG with color masking or rectangle fallback)"""
        screen_x, screen_y = self.world_to_screen(obs_x, obs_y, camera_offset)

        if not (-100 < screen_y < self.road_display_height + 100):
            return

        if USE_PNG and self.obstacle_img is not None:

            scaled_width = int(obs_width * self.scale)
            scaled_height = int(obs_height * self.scale)
            obs_img_scaled = pygame.transform.scale(
                self.obstacle_img, (scaled_width, scaled_height)
            )


            mask_surface = pygame.Surface(
                (scaled_width, scaled_height), pygame.SRCALPHA
            )
            mask_surface.fill(color)


            mask_surface.blit(
                obs_img_scaled, (0, 0), special_flags=pygame.BLEND_RGBA_MIN
            )


            obs_rect = mask_surface.get_rect(center=(screen_x, screen_y))
            self.screen.blit(mask_surface, obs_rect)

        else:

            obs_surface = pygame.Surface(
                (obs_width * self.scale, obs_height * self.scale), pygame.SRCALPHA
            )
            pygame.draw.rect(
                obs_surface,
                color,
                (0, 0, obs_width * self.scale, obs_height * self.scale),
            )
            border_color = (180, 0, 0) if color == RED else (180, 145, 0)
            pygame.draw.rect(
                obs_surface,
                border_color,
                (0, 0, obs_width * self.scale, obs_height * self.scale),
                2,
            )

            obs_rect = obs_surface.get_rect(center=(screen_x, screen_y))
            self.screen.blit(obs_surface, obs_rect)

    def draw_car(self, car_x, car_y, car_angle, car_width, car_height, camera_offset):
        """Draw the car (PNG with color masking or rectangle fallback)"""
        screen_x, screen_y = self.world_to_screen(car_x, car_y, camera_offset)

        if USE_PNG and self.car_img is not None:


            scaled_width = int(car_width * self.scale)
            scaled_height = int(car_height * self.scale)
            car_img_scaled = pygame.transform.scale(
                self.car_img, (scaled_width, scaled_height)
            )


            mask_surface = pygame.Surface(
                (scaled_width, scaled_height), pygame.SRCALPHA
            )
            mask_surface.fill(BLUE)


            mask_surface.blit(
                car_img_scaled, (0, 0), special_flags=pygame.BLEND_RGBA_MIN
            )


            center_x = scaled_width / 2
            pygame.draw.polygon(
                mask_surface,
                YELLOW,
                [(center_x, 5), (center_x - 8, 20), (center_x + 8, 20)],
            )


            rotation_angle = car_angle - 90
            rotated_car = pygame.transform.rotate(mask_surface, rotation_angle)
            rotated_rect = rotated_car.get_rect(center=(screen_x, screen_y))
            self.screen.blit(rotated_car, rotated_rect)

        else:

            car_surface = pygame.Surface(
                (car_width * self.scale, car_height * self.scale), pygame.SRCALPHA
            )
            pygame.draw.rect(
                car_surface,
                BLUE,
                (0, 0, car_width * self.scale, car_height * self.scale),
            )
            pygame.draw.rect(
                car_surface,
                LIGHT_BLUE,
                (0, 0, car_width * self.scale, car_height * self.scale),
                2,
            )

            center_x = car_width * self.scale / 2
            pygame.draw.polygon(
                car_surface,
                YELLOW,
                [(center_x, 5), (center_x - 8, 20), (center_x + 8, 20)],
            )

            rotation_angle = car_angle - 90
            rotated_car = pygame.transform.rotate(car_surface, rotation_angle)
            rotated_rect = rotated_car.get_rect(center=(screen_x, screen_y))

            self.screen.blit(rotated_car, rotated_rect)

    def draw_near_miss_box(self, corners, camera_offset):
        if not corners:
            return
        screen_points = [
            self.world_to_screen(float(x), float(y), camera_offset) for x, y in corners
        ]
        if len(screen_points) >= 3:
            pygame.draw.polygon(self.screen, LIGHT_BLUE, screen_points, width=2)

    def draw_sensors(self, car_x, car_y, sensors, camera_offset, show_labels=False):
        """Draw sensor rays"""
        screen_x, screen_y = self.world_to_screen(car_x, car_y, camera_offset)
        sensor_tip_labels = ["R2", "R1", "F", "L1", "L2", "SR", "SL"]

        for i, sensor in enumerate(sensors):
            angle = sensor["angle"]
            distance = sensor["distance"]
            normalized = sensor.get("normalized", 1.0)

            rad = math.radians(angle)
            end_x = car_x + distance * math.cos(rad)
            end_y = car_y + distance * math.sin(rad)
            end_screen_x, end_screen_y = self.world_to_screen(
                end_x, end_y, camera_offset
            )

            if i == 2:
                from main_constant import OBSTACLE_WARNING_DISTANCE_FRONT

                if distance < OBSTACLE_WARNING_DISTANCE_FRONT:
                    color = RED
                elif normalized is None:
                    color = GREEN
                elif normalized > 0.5:
                    color = GREEN
                else:
                    color = GREEN
            elif i == 5 or i == 6:
                from main_constant import OBSTACLE_WARNING_DISTANCE_SIDES

                if distance < OBSTACLE_WARNING_DISTANCE_SIDES:
                    color = RED
                elif normalized is None:
                    color = GREEN
                elif normalized > 0.5:
                    color = GREEN
                else:
                    color = GREEN
            else:
                if normalized is None:
                    color = GREEN
                elif normalized > 0.5:
                    color = GREEN
                else:
                    color = GREEN

            pygame.draw.line(
                self.screen,
                color,
                (screen_x, screen_y),
                (end_screen_x, end_screen_y),
                2,
            )
            pygame.draw.circle(self.screen, color, (end_screen_x, end_screen_y), 3)
            if show_labels and i < len(sensor_tip_labels):
                label = self.font.render(sensor_tip_labels[i], True, WHITE)
                self.screen.blit(label, (end_screen_x + 4, end_screen_y - 10))

    def draw_sensor_tip_labels_overlay(self, car_x, car_y, sensors, camera_offset):
        """Draw sensor tip labels as top-most overlay (above panels)."""
        sensor_tip_labels = ["R2", "R1", "F", "L1", "L2", "SR", "SL"]
        for i, sensor in enumerate(sensors):
            if i >= len(sensor_tip_labels):
                continue
            angle = sensor["angle"]
            distance = sensor["distance"]
            rad = math.radians(angle)
            end_x = car_x + distance * math.cos(rad)
            end_y = car_y + distance * math.sin(rad)
            end_screen_x, end_screen_y = self.world_to_screen(end_x, end_y, camera_offset)
            label = self.font.render(sensor_tip_labels[i], True, WHITE)
            self.screen.blit(label, (end_screen_x + 4, end_screen_y - 10))

    @staticmethod
    def _fmt_decimal_comma(value, decimals=1, show_plus=False):
        numeric = float(value)
        if show_plus:
            text = f"{numeric:+.{int(decimals)}f}"
        else:
            text = f"{numeric:.{int(decimals)}f}"
        return text.replace(".", ",")

    def _format_sensor_calc_parts(self, sensor_name, sensor):
        base_distance = float(sensor.get("base_distance", sensor.get("distance", 0.0)))
        noise_value = float(sensor.get("noise", 0.0))
        measured_distance = float(sensor.get("distance", base_distance))

        base_text = self._fmt_decimal_comma(base_distance, decimals=1, show_plus=False)
        noise_text = self._fmt_decimal_comma(abs(noise_value), decimals=1, show_plus=False)
        measured_text = self._fmt_decimal_comma(
            measured_distance,
            decimals=1,
            show_plus=False,
        )
        noise_prefix = "+" if noise_value >= 0 else "-"
        left_text = f"{sensor_name}: {base_text}"
        noise_text_colored = f"{noise_prefix}{noise_text}"
        right_text = f" = {measured_text}"
        return left_text, noise_text_colored, right_text

    def draw_info_panels(
        self,
        info,
        episode=0,
        total_reward=0,
        epsilon=0,
        fps=0,
        timeframe=0,
        paused=False,
    ):
        """Draw the two-column info area (left: sensors/NN/actions, right: stats/state/controls)."""
        left_x = self.road_display_width
        right_x = left_x + self.info_left_width
        sensor_panel_x = left_x
        car_info_panel_x = right_x
        panel_content_pad = 36

        pygame.draw.rect(
            self.screen,
            PANEL_BG,
            (left_x, 0, self.info_left_width, self.window_height),
        )
        pygame.draw.rect(
            self.screen,
            PANEL_BG_ALT,
            (right_x, 0, self.info_right_width, self.window_height),
        )

        pygame.draw.rect(
            self.screen,
            PANEL_HEADER,
            (left_x, 0, self.info_left_width, 44),
        )
        pygame.draw.rect(
            self.screen,
            PANEL_HEADER,
            (right_x, 0, self.info_right_width, 44),
        )

        pygame.draw.line(
            self.screen, PANEL_BORDER, (right_x, 0), (right_x, self.window_height), 2
        )
        pygame.draw.line(
            self.screen,
            PANEL_BORDER,
            (left_x, 44),
            (right_x + self.info_right_width, 44),
            1,
        )


        title = self.font_title.render("CAR INFO", True, WHITE)
        self.screen.blit(title, (car_info_panel_x + panel_content_pad, 12))

        if paused:
            pause_text = self.font_title.render("PAUSED", True, RED)
            self.screen.blit(pause_text, (car_info_panel_x + panel_content_pad + 150, 12))
            y_left = 62
        else:
            y_left = 62

        line_height = 20
        target_speed = float(info.get("target_speed", info["speed"]))
        speed_delta = float(info.get("speed_delta", 0.0))
        speed_kmh = to_kmh(float(info["speed"]))
        current_ttc_ms = info.get("ttc_ms")
        ttc_counting = bool(info.get("ttc_counting", current_ttc_ms is not None))
        near_miss_active = bool(info.get("near_miss_active", info.get("near_miss", 0)))
        near_miss_total = info.get("near_miss_total", None)
        in_center_zone = info.get("in_lane_center_zone", None)
        info_timeframe = info.get("timeframe", timeframe)
        world_step = info.get("world_step", None)
        if world_step is None:
            world_distance = info.get("world_distance", 0.0)
            try:
                world_step = int(round(float(world_distance)))
            except Exception:
                world_step = 0
        try:
            info_timeframe = int(info_timeframe)
        except Exception:
            info_timeframe = int(timeframe)

        center_zone_text = "Center Zone: N/A"
        if in_center_zone is not None:
            center_zone_text = f"Center Zone: {'IN' if in_center_zone else 'OUT'}"

        left_lines = [
            f"Episode: {episode}",
            f"Timeframe: {info_timeframe}",
            f"Step: {int(world_step)}",
            f"Total Reward: {total_reward:.2f}",
            f"Epsilon: {epsilon:.3f}",
            f"FPS: {fps:.0f}",
            f"Speed Mode: {self.get_speed_mode_label()}",
            "",
            "CAR STATE",
            f"Position X: {info['car_x']:.1f}",
            f"Position Y: {info['car_y']:.1f}",
            f"Angle: {info['car_angle']:.1f} deg",
            f"Speed: {info['speed']:.2f}",
            f"Km/h: {speed_kmh:.1f}",
            f"Target Speed: {target_speed:.2f}",
            f"Delta/Timeframe: {speed_delta:+.4f}",
            f"TTC Counting: {ttc_counting}",
            f"TTC: {format_ttc_ms(current_ttc_ms) if ttc_counting else 'None'} ms",
            f"Count near_miss: {'YES' if near_miss_active else 'NO'}",
            f"Near Miss Count: {int(near_miss_total)}"
            if near_miss_total is not None
            else "",
            center_zone_text,
            "",
            "CONTROLS",
            "P: Pause/Resume",
            "R: Reset Episode",
            "Q: Quit",
            "I: Toggle Indicators",
            f"1: Speed x{KEYONE_MULTIPLIER}",
            "2: Slow Motion",
        ]

        for line in left_lines:
            if line in {"CAR STATE", "CONTROLS"}:
                color = LIGHT_BLUE
            elif line.startswith(("P:", "R:", "Q:", "I:", "1:", "2:")):
                color = TEXT_MUTED
            else:
                color = WHITE
            font_to_use = self.font_subtitle if line in {"CAR STATE", "CONTROLS"} else self.font_text
            text = font_to_use.render(line, True, color)
            self.screen.blit(text, (car_info_panel_x + panel_content_pad, y_left))
            y_left += line_height


        right_title = self.font_title.render("SENSORS / NETWORKS", True, WHITE)
        self.screen.blit(right_title, (sensor_panel_x + panel_content_pad, 12))
        y_right = 62

        last_action = info.get("last_action", None)
        actions_header = self.font_subtitle.render("ACTIONS", True, LIGHT_BLUE)
        self.screen.blit(actions_header, (sensor_panel_x + panel_content_pad, y_right))
        y_right += 22
        action_names = [
            "Slow Left",
            "Slow Straight",
            "Slow Right",
            "Fast Left",
            "Fast Straight",
            "Fast Right",
        ]
        for i, action_name in enumerate(action_names):
            action_text = self.font.render(f"{i + 1} = {action_name}", True, TEXT_MUTED)
            self.screen.blit(action_text, (sensor_panel_x + panel_content_pad, y_right))
            y_right += 18

        if last_action is not None and last_action in range(6):
            current_name = action_names[last_action]
            current_text = self.font.render(
                f"Current Action: {last_action + 1} ({current_name})", True, ORANGE
            )
            self.screen.blit(current_text, (sensor_panel_x + panel_content_pad, y_right))
            y_right += 18

        nn = info.get("nn_input", None)
        nn_out = info.get("nn_output", None)
        if nn is not None or nn_out is not None:
            y_right += 12
            text = self.font_subtitle.render("NETWORKS", True, LIGHT_BLUE)
            self.screen.blit(text, (sensor_panel_x + panel_content_pad, y_right))
            y_right += 22

        if nn is not None:
            text = self.font_subtitle.render("NN INPUT", True, LIGHT_BLUE)
            self.screen.blit(text, (sensor_panel_x + panel_content_pad, y_right))
            y_right += 18
            nn_labels = [
                "R2",
                "R1",
                "F",
                "L1",
                "L2",
                "SR",
                "SL",
                "Speed",
            ]
            for i, value in enumerate(nn):
                label = nn_labels[i] if i < len(nn_labels) else f"Input-{i}"
                line = f"{label} = {float(value):.2f}"
                text = self.font.render(line, True, TEXT_MUTED)
                self.screen.blit(text, (sensor_panel_x + panel_content_pad, y_right))
                y_right += 18
            y_right += 12

        if nn_out is not None:
            try:
                outputs = [float(v) for v in nn_out]
            except TypeError:
                outputs = None

            if outputs is not None:
                text = self.font_subtitle.render("NN OUTPUT (Q)", True, LIGHT_BLUE)
                self.screen.blit(text, (sensor_panel_x + panel_content_pad, y_right))
                y_right += 18

                action_names = ["SlowL", "SlowS", "SlowR", "FastL", "FastS", "FastR"]
                for i, q in enumerate(outputs):
                    name = action_names[i] if i < len(action_names) else f"A{i}"
                    line = f"Q[{name}]: {q:.2f}"
                    text = self.font.render(line, True, TEXT_MUTED)
                    self.screen.blit(text, (sensor_panel_x + panel_content_pad, y_right))
                    y_right += 18

        y_right += 10
        text = self.font_subtitle.render("SENSORS", True, LIGHT_BLUE)
        self.screen.blit(text, (sensor_panel_x + panel_content_pad, y_right))
        y_right += line_height

        sensor_names = [
            "Front-R2",
            "Front-R1",
            "Front",
            "Front-L1",
            "Front-L2",
            "Side-R",
            "Side-L",
        ]
        for i, sensor in enumerate(info["sensors"]):
            normalized = sensor.get("normalized", None)
            is_front = i == 2
            is_side = i == 5 or i == 6
            sensor_color = self._get_sensor_color(
                sensor["distance"],
                normalized,
                is_front_sensor=is_front,
                is_side_sensor=is_side,
            )
            left_text, noise_text, right_text = self._format_sensor_calc_parts(
                sensor_names[i], sensor
            )
            x = sensor_panel_x + panel_content_pad
            left_surface = self.font.render(left_text, True, sensor_color)
            self.screen.blit(left_surface, (x, y_right))
            x += left_surface.get_width()
            noise_surface = self.font.render(noise_text, True, BRIGHT_PURPLE)
            self.screen.blit(noise_surface, (x, y_right))
            x += noise_surface.get_width()
            right_surface = self.font.render(right_text, True, sensor_color)
            self.screen.blit(right_surface, (x, y_right))
            y_right += 20

    def _draw_experiment_button(self, rect, label, enabled=True, active=False):
        if not enabled:
            bg_color = (65, 65, 65)
            border_color = (120, 120, 120)
            text_color = (170, 170, 170)
        elif active:
            bg_color = (0, 155, 0)
            border_color = (190, 255, 190)
            text_color = WHITE
        else:
            bg_color = (40, 90, 170)
            border_color = (130, 180, 255)
            text_color = WHITE

        pygame.draw.rect(self.screen, bg_color, rect, border_radius=5)
        pygame.draw.rect(self.screen, border_color, rect, width=2, border_radius=5)
        text = self.font.render(label, True, text_color)
        text_rect = text.get_rect(center=rect.center)
        self.screen.blit(text, text_rect)

    def draw_experiment_panel(self, experiment_data):
        """Draw obstacle control panel used in --experiment mode."""
        if not self.experiment_mode:
            return

        if experiment_data is None:
            experiment_data = {"selected_lanes": [], "distance": 125, "to_spawn_list": []}

        selected_lanes = set(experiment_data.get("selected_lanes", []))
        distance = int(experiment_data.get("distance", 125))
        to_spawn_list = list(experiment_data.get("to_spawn_list", []))

        panel_x = self.road_display_width + self.info_panel_width
        panel_width = self.experiment_panel_width
        panel_rect = pygame.Rect(panel_x, 0, panel_width, self.window_height)

        pygame.draw.rect(self.screen, (30, 30, 30), panel_rect)
        pygame.draw.line(
            self.screen, WHITE, (panel_x, 0), (panel_x, self.window_height), 2
        )

        self.experiment_button_rects = {}

        title = self.font_large.render("Obstacle Controls", True, WHITE)
        self.screen.blit(title, (panel_x + 20, 20))

        y = 60
        subtitle = self.font.render("3 Path", True, YELLOW)
        self.screen.blit(subtitle, (panel_x + 20, y))
        y += 24

        lane_labels = ["left", "center", "right"]
        lane_width = (panel_width - 50) // 3
        lane_gap = 5
        lane_x = panel_x + 20
        for lane_name in lane_labels:
            rect = pygame.Rect(lane_x, y, lane_width, 36)
            self.experiment_button_rects[f"lane_{lane_name}"] = rect
            self._draw_experiment_button(
                rect, lane_name.upper(), enabled=True, active=lane_name in selected_lanes
            )
            lane_x += lane_width + lane_gap

        y += 52
        distance_title = self.font.render("Distance", True, YELLOW)
        self.screen.blit(distance_title, (panel_x + 20, y))
        y += 24

        minus_rect = pygame.Rect(panel_x + 20, y, 60, 36)
        plus_rect = pygame.Rect(panel_x + panel_width - 80, y, 60, 36)
        value_rect = pygame.Rect(panel_x + 90, y, panel_width - 180, 36)

        self.experiment_button_rects["distance_minus"] = minus_rect
        self.experiment_button_rects["distance_plus"] = plus_rect

        self._draw_experiment_button(minus_rect, "-", enabled=True, active=False)
        self._draw_experiment_button(plus_rect, "+", enabled=True, active=False)

        pygame.draw.rect(self.screen, (50, 50, 50), value_rect, border_radius=5)
        pygame.draw.rect(self.screen, (140, 140, 140), value_rect, width=2, border_radius=5)
        value_text = self.font_large.render(str(distance), True, WHITE)
        value_text_rect = value_text.get_rect(center=value_rect.center)
        self.screen.blit(value_text, value_text_rect)

        y += 50
        can_add = len(selected_lanes) > 0
        add_rect = pygame.Rect(panel_x + 20, y, panel_width - 40, 38)
        self.experiment_button_rects["add_list"] = add_rect
        self._draw_experiment_button(
            add_rect, "ADD LIST", enabled=can_add, active=False
        )

        y += 50
        spawn_rect = pygame.Rect(panel_x + 20, y, (panel_width - 50) // 2, 38)
        clear_rect = pygame.Rect(
            spawn_rect.right + 10, y, (panel_width - 50) // 2, 38
        )
        self.experiment_button_rects["spawn"] = spawn_rect
        self.experiment_button_rects["clear_spawn"] = clear_rect

        self._draw_experiment_button(spawn_rect, "SPAWN", enabled=True, active=False)
        self._draw_experiment_button(
            clear_rect, "CLEAR SPAWN", enabled=True, active=False
        )

        y += 54
        box_label = self.font.render("To Spawn Lists", True, YELLOW)
        self.screen.blit(box_label, (panel_x + 20, y))
        y += 22

        list_rect = pygame.Rect(panel_x + 20, y, panel_width - 40, self.window_height - y - 20)
        pygame.draw.rect(self.screen, (20, 20, 20), list_rect)
        pygame.draw.rect(self.screen, (120, 120, 120), list_rect, width=2)

        row_height = 20
        max_rows = max(1, (list_rect.height - 10) // row_height)
        start_index = max(0, len(to_spawn_list) - max_rows)
        visible_rows = to_spawn_list[start_index:]

        row_y = list_rect.y + 6
        for idx, row in enumerate(visible_rows, start=start_index + 1):
            lanes = row.get("lanes", [])
            distance_val = int(row.get("distance", 0))
            row_text = f"{idx}. {'+'.join(lanes)} | d={distance_val}"
            text = self.font.render(row_text, True, WHITE)
            self.screen.blit(text, (list_rect.x + 8, row_y))
            row_y += row_height

    def draw_neuron_panel(self, neuron_data):
        """Draw forward-pass neuron details used in --neuron mode."""
        if not self.neuron_mode:
            return

        panel_x = (
            self.road_display_width
            + self.info_panel_width
            + self.experiment_panel_width
        )
        panel_width = self.neuron_panel_width
        panel_rect = pygame.Rect(panel_x, 0, panel_width, self.window_height)

        pygame.draw.rect(self.screen, (24, 26, 30), panel_rect)
        pygame.draw.line(
            self.screen, WHITE, (panel_x, 0), (panel_x, self.window_height), 2
        )

        title = self.font_large.render("Neuron Trace", True, WHITE)
        self.screen.blit(title, (panel_x + 18, 18))

        y = 52
        line_height = 16
        max_y = self.window_height - 18

        def _clip_text(text, width, indent=0):
            max_chars = max(20, (int(width) - 12 - int(indent)) // 7)
            clipped = str(text)
            if len(clipped) > max_chars:
                clipped = clipped[: max_chars - 3] + "..."
            return clipped

        def draw_line(text, color=WHITE, indent=0):
            nonlocal y
            if y > max_y:
                return
            clipped = _clip_text(text, panel_width - 30, indent)
            surface = self.font.render(clipped, True, color)
            self.screen.blit(surface, (panel_x + 18 + indent, y))
            y += line_height

        def draw_column_line(text, x, y_pos, width, color=WHITE, indent=0):
            if y_pos > max_y:
                return y_pos
            clipped = _clip_text(text, width, indent)
            surface = self.font.render(clipped, True, color)
            self.screen.blit(surface, (x + indent, y_pos))
            return y_pos + line_height

        if not neuron_data:
            draw_line("Waiting for first network pass...", YELLOW)
            return
        if neuron_data.get("error"):
            draw_line("Neuron trace error:", RED)
            draw_line(neuron_data.get("error"), WHITE, 8)
            return

        total_params = int(neuron_data.get("total_params", 0))
        draw_line(f"Policy net params: {total_params}", YELLOW)
        draw_line("Forward: y = W*x + b, hidden uses ReLU", LIGHT_BLUE)
        y += 4

        layers = neuron_data.get("layers", [])
        hidden_layers = [
            layer for layer in layers if str(layer.get("type", "hidden")) != "output"
        ]
        output_layers = [
            layer for layer in layers if str(layer.get("type", "hidden")) == "output"
        ]

        draw_line("-- Layers (top active neurons) --", YELLOW)
        column_top = y
        column_gap = 18
        column_width = (panel_width - 36 - column_gap) // 2
        left_x = panel_x + 18
        right_x = left_x + column_width + column_gap
        pygame.draw.line(
            self.screen,
            (80, 80, 80),
            (right_x - 9, column_top),
            (right_x - 9, self.window_height - 18),
            1,
        )

        def draw_layer_column(column_layers, x, start_y, width, title):
            y_col = start_y
            y_col = draw_column_line(title, x, y_col, width, YELLOW)
            for layer in column_layers:
                layer_index = int(layer.get("index", 0))
                layer_kind = layer.get("type", "hidden")
                in_size = int(layer.get("input_size", 0))
                out_size = int(layer.get("output_size", 0))
                active_count = layer.get("active_count", None)
                active_text = (
                    f", active={int(active_count)}/{out_size}"
                    if active_count is not None
                    else ""
                )
                y_col = draw_column_line(
                    f"L{layer_index} {layer_kind}: {in_size}->{out_size}{active_text}",
                    x,
                    y_col,
                    width,
                    LIGHT_BLUE,
                )
                y_col = draw_column_line(
                    "a min/mean/max = "
                    f"{float(layer.get('activation_min', 0.0)):+.4f} / "
                    f"{float(layer.get('activation_mean', 0.0)):+.4f} / "
                    f"{float(layer.get('activation_max', 0.0)):+.4f}",
                    x,
                    y_col,
                    width,
                    WHITE,
                    8,
                )

                neurons = layer.get("neurons", [])
                max_neurons_to_draw = 6 if layer_kind == "output" else 2
                for neuron in neurons[:max_neurons_to_draw]:
                    y_col = draw_column_line(
                        f"{neuron.get('label')} "
                        f"b={float(neuron.get('bias', 0.0)):+.4f} "
                        f"z={float(neuron.get('z', 0.0)):+.4f} "
                        f"a={float(neuron.get('activation', 0.0)):+.4f}",
                        x,
                        y_col,
                        width,
                        WHITE,
                        8,
                    )
                    for contrib in neuron.get("contributions", [])[:2]:
                        y_col = draw_column_line(
                            f"{contrib.get('input_name')} "
                            f"w={float(contrib.get('weight', 0.0)):+.4f} "
                            f"in={float(contrib.get('input', 0.0)):+.4f} "
                            f"prod={float(contrib.get('product', 0.0)):+.4f}",
                            x,
                            y_col,
                            width,
                            GRAY,
                            20,
                        )
                y_col += 4
                if y_col > max_y:
                    break
            return y_col

        draw_layer_column(hidden_layers, left_x, column_top, column_width, "Hidden")
        draw_layer_column(output_layers, right_x, column_top, column_width, "Output")

    def _get_sensor_color(
        self, distance, normalized=None, is_front_sensor=False, is_side_sensor=False
    ):
        """Get color based on sensor normalized value"""
        if is_front_sensor:
            from main_constant import OBSTACLE_WARNING_DISTANCE_FRONT

            if distance < OBSTACLE_WARNING_DISTANCE_FRONT:
                return RED

        if is_side_sensor:
            from main_constant import OBSTACLE_WARNING_DISTANCE_SIDES

            if distance < OBSTACLE_WARNING_DISTANCE_SIDES:
                return RED

        return GREEN

    def render(
        self,
        info,
        episode=0,
        total_reward=0,
        epsilon=0,
        timeframe=0,
        paused=False,
        experiment_data=None,
        neuron_data=None,
    ):
        """Render the entire scene"""

        self.last_render_info = info
        self.last_episode = episode
        self.last_total_reward = total_reward
        self.last_epsilon = epsilon
        self.last_timeframe = timeframe
        self.last_paused = paused
        self.last_experiment_data = experiment_data
        self.last_neuron_data = neuron_data

        self.screen.fill(BLACK)

        camera_offset = self.get_camera_offset(info["car_y"])
        finish_line_y = info.get("finish_line_y", self.env.finish_line_y)
        self.draw_road(camera_offset, finish_line_y)

        ttc_counting = bool(info.get("ttc_counting", False))
        ttc_target_index = info.get("ttc_target_index", None)
        for obstacle_index, obstacle in enumerate(info["obstacles"]):
            obstacle_color = (
                YELLOW
                if ttc_counting and obstacle_index == ttc_target_index
                else RED
            )
            self.draw_obstacle_car(
                obstacle["x"],
                obstacle["y"],
                obstacle["width"],
                obstacle["height"],
                camera_offset,
                color=obstacle_color,
            )

        self.draw_sensors(
            info["car_x"],
            info["car_y"],
            info["sensors"],
            camera_offset,
            show_labels=False,
        )
        self.draw_car(
            info["car_x"],
            info["car_y"],
            info["car_angle"],
            info["car_width"],
            info["car_height"],
            camera_offset,
        )
        if self.show_near_miss_indicator:
            self.draw_near_miss_box(info.get("near_miss_corners"), camera_offset)

        fps = self.clock.get_fps()
        self.draw_info_panels(
            info, episode, total_reward, epsilon, fps, timeframe, paused
        )
        self.draw_experiment_panel(experiment_data)
        self.draw_neuron_panel(neuron_data)
        if self.show_sensor_tip_labels:
            self.draw_sensor_tip_labels_overlay(
                info["car_x"],
                info["car_y"],
                info["sensors"],
                camera_offset,
            )

        pygame.display.flip()


        self.clock.tick(self.get_effective_render_fps())

    def render_cached(self, paused=None):
        """Re-render using cached state (for pause)."""
        if self.last_render_info is not None:
            self.render(
                self.last_render_info,
                self.last_episode,
                self.last_total_reward,
                self.last_epsilon,
                self.last_timeframe,
                self.last_paused if paused is None else paused,
                self.last_experiment_data,
                self.last_neuron_data,
            )

    def handle_events(self, paused=False):
        """Handle pygame events"""
        running = True
        reset = False
        manual_action = None
        pause_toggle = False
        experiment_actions = []

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    running = False
                elif event.key == pygame.K_r:
                    reset = True
                elif event.key == pygame.K_p:
                    pause_toggle = True
                elif event.key == pygame.K_1:
                    self.toggle_speed()
                    print(f"Speed Mode: {self.get_speed_mode_label()}")
                elif event.key == pygame.K_2:
                    self.toggle_slow_motion()
                    print(f"Speed Mode: {self.get_speed_mode_label()}")
                elif event.key == pygame.K_i:
                    indicators_on = self.toggle_indicators()
                    print(
                        "Indicators: "
                        f"{'ON' if indicators_on else 'OFF'}"
                    )
                elif event.key == pygame.K_LEFT:
                    manual_action = 3
                elif event.key == pygame.K_UP:
                    manual_action = 4
                elif event.key == pygame.K_DOWN:
                    manual_action = 1
                elif event.key == pygame.K_RIGHT:
                    manual_action = 5
            elif (
                self.experiment_mode
                and event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
            ):
                mouse_pos = event.pos
                for button_id, rect in self.experiment_button_rects.items():
                    if rect.collidepoint(mouse_pos):
                        if button_id.startswith("lane_"):
                            lane_name = button_id.replace("lane_", "", 1)
                            experiment_actions.append(
                                {"type": "toggle_lane", "lane": lane_name}
                            )
                        elif button_id == "distance_minus":
                            experiment_actions.append({"type": "distance_minus"})
                        elif button_id == "distance_plus":
                            experiment_actions.append({"type": "distance_plus"})
                        elif button_id == "add_list":
                            experiment_actions.append({"type": "add_list"})
                        elif button_id == "spawn":
                            experiment_actions.append({"type": "spawn"})
                        elif button_id == "clear_spawn":
                            experiment_actions.append({"type": "clear_spawn"})
                        break

        return running, reset, manual_action, pause_toggle, experiment_actions

    def close(self):
        """Close pygame window"""
        pygame.quit()


def load_checkpoint_and_algo(model_path, fallback_algo=ALGO_DQN):
    """Load a checkpoint once and infer its saved algorithm when available."""
    import torch

    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")

    checkpoint_algo = None
    if isinstance(checkpoint, dict):
        checkpoint_algo = checkpoint.get("algo")
        config = checkpoint.get("agent_config", {})
        if not checkpoint_algo and isinstance(config, dict):
            checkpoint_algo = config.get("algo")

    checkpoint_algo = str(checkpoint_algo).lower() if checkpoint_algo else None
    if checkpoint_algo in SUPPORTED_ALGOS:
        return checkpoint, checkpoint_algo
    return checkpoint, fallback_algo


def create_visualize_agent(env, algo=ALGO_DQN):
    """Create an inference agent with the same defaults used by visualization."""
    return DQNAgent(
        state_size=env.state_size,
        action_size=env.action_size,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        epsilon=TRAIN_MAX_EPSILON,
        epsilon_min=TRAIN_MIN_EPSILON,
        epsilon_decay=EPSILON_DECAY,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
        memory_size=MEMORY_SIZE,
        algo=algo,
    )


def run_evaluation_episode(agent, env, step_multiplier=1):
    """Run one headless episode and return visualize-compatible metrics."""
    try:
        step_multiplier = int(step_multiplier)
    except (TypeError, ValueError):
        step_multiplier = 1
    step_multiplier = max(1, step_multiplier)

    state = env.reset()
    total_reward = 0.0
    done = False
    close_distance_total = 0
    near_miss_total = 0
    step_count = 0
    current_vis_action = 1
    episode_sensor_mins = [float("inf")] * len(SENSOR_LOG_NAMES)
    episode_ttc_samples = []
    info = {
        "car_x": getattr(env, "car_x", 0.0),
        "car_y": getattr(env, "car_y", 0.0),
        "finish_line_y": getattr(env, "finish_line_y", 2000),
        "world_distance": getattr(env, "world_distance", 0.0),
        "reached_finish": False,
    }

    start_time = time.perf_counter()

    while not done:
        for _ in range(step_multiplier):
            if step_count % DECISION_INTERVAL == 0:
                current_vis_action = agent.select_action(state, training=False)

            is_decision_step = step_count % DECISION_INTERVAL == 0
            next_state, reward, done, info = env.step(
                current_vis_action, apply_steering=is_decision_step
            )

            current_ttc_ms = info.get("ttc_ms")
            if current_ttc_ms is None:
                current_ttc_ms = calculate_env_ttc_ms(env)
            if current_ttc_ms is not None:
                episode_ttc_samples.append(float(current_ttc_ms))

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

            total_reward += reward
            close_distance_total += int(info.get("warning_close_count", 0))
            near_miss_total += int(info.get("near_miss", 0))
            state = next_state
            step_count += 1

            if done:
                break

    duration_ms = int(round((time.perf_counter() - start_time) * 1000.0))
    finish_y = info.get("finish_line_y", getattr(env, "finish_line_y", 2000))
    raw_prog = (info["car_y"] / finish_y) * 100.0 if finish_y != 0 else 0.0
    rounded_prog = round(raw_prog / 5.0) * 5
    progress_pct = (
        100.0 if info.get("reached_finish") else min(95.0, max(0.0, rounded_prog))
    )
    world_step = int(
        round(float(info.get("world_distance", getattr(env, "world_distance", 0.0))))
    )
    ttc_summary = summarize_ttc_samples(episode_ttc_samples)

    seconds = int(duration_ms / 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return {
        "close_distance": int(close_distance_total),
        "near_miss": int(near_miss_total),
        "reward": float(total_reward),
        "time_ms": int(duration_ms),
        "time_str": f"{hours:02d}:{minutes:02d}:{seconds:02d}",
        "timeframe": int(step_count),
        "steps": int(world_step),
        "progress_pct": float(progress_pct),
        "reached_finish": 1 if info.get("reached_finish") else 0,
        "min_ttc": ttc_summary["min_ttc"],
        "avg_ttc": ttc_summary["avg_ttc"],
        "min_sensor_logs": build_min_sensor_logs(env, episode_sensor_mins),
        "final_info": info,
    }


def run_evamodel(models_dir="models", algo=ALGO_DQN, log_dir=VISUALIZE_LOG_DIR):
    """Evaluate every non-check model in models_dir once, newest suffix first."""
    model_entries = get_evaluable_model_paths(models_dir)
    evaluate_csv_path = get_evaluate_csv_path(log_dir)
    step_multiplier = int(max(1, KEYONE_MULTIPLIER))

    print("\n=== Evaluate Models Mode ===")
    print(f"Models dir: {os.path.abspath(models_dir)}")
    print(f"Model order: newest numeric suffix first; *_check.pth skipped")
    print(f"Speed mode: FAST x{step_multiplier} (same as key '1')")
    print(f"Evaluate CSV: {evaluate_csv_path}")

    if not model_entries:
        print("No eligible .pth models found.")
        return

    print(f"Found {len(model_entries)} eligible model(s).")
    env = CarEnvironment(obstacles_config=OBSTACLES)
    agent = None
    agent_algo = None
    visualize_rewards = []
    rows_written = 0

    with open(evaluate_csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=get_visualize_csv_headers(include_model=True),
        )
        csv_writer.writeheader()

        for model_index, (model_number, model_path) in enumerate(model_entries, start=1):
            print(
                f"\n[{model_index}/{len(model_entries)}] Evaluating model {model_number}: "
                f"{model_path}"
            )
            try:
                checkpoint, checkpoint_algo = load_checkpoint_and_algo(model_path, algo)
                if agent is None or agent_algo != checkpoint_algo:
                    agent = create_visualize_agent(env, checkpoint_algo)
                    agent_algo = checkpoint_algo
                agent.load(model_path, checkpoint=checkpoint, load_replay_buffer=False)
                agent.epsilon = 0.0
            except Exception as exc:
                print(f"[ERROR] Skipping model {model_number}: {exc}")
                continue

            metrics = run_evaluation_episode(
                agent,
                env,
                step_multiplier=step_multiplier,
            )
            visualize_rewards.append(float(metrics["reward"]))
            avg_reward_all = (
                float(sum(visualize_rewards) / len(visualize_rewards))
                if visualize_rewards
                else 0.0
            )
            rows_written += 1
            row = build_visualize_episode_row(
                model=model_number,
                episode=rows_written,
                close_distance=metrics["close_distance"],
                near_miss=metrics["near_miss"],
                mse=0.0,
                reward=metrics["reward"],
                avg_reward=avg_reward_all,
                time_ms=metrics["time_ms"],
                timeframe=metrics["timeframe"],
                steps=metrics["steps"],
                progress_pct=metrics["progress_pct"],
                reached_finish=metrics["reached_finish"],
                min_ttc=metrics["min_ttc"],
                avg_ttc=metrics["avg_ttc"],
                min_sensor_logs=metrics["min_sensor_logs"],
            )
            csv_writer.writerow(row)
            csv_file.flush()

            final_info = metrics["final_info"]
            print(f"Model {model_number} finished!")
            print(f"  Total Reward: {metrics['reward']:.2f}")
            print("  MSE: 0.000 (Inference)")
            print(f"  Close Distance: {metrics['close_distance']}")
            print(f"  Near Miss: {metrics['near_miss']}")
            print(f"  Avg Reward: {avg_reward_all:.2f}")
            print(f"  Time: {metrics['time_str']}")
            print(f"  Timeframe: {metrics['timeframe']}")
            print(f"  Step: {metrics['steps']}")
            print(f"  min_ttc: {format_ttc_ms(metrics['min_ttc'])} ms")
            print(f"  avg_ttc: {format_ttc_ms(metrics['avg_ttc'])} ms")
            print(f"  Progress: {metrics['progress_pct']:.1f}%")
            print(
                f"  Final Position: "
                f"({final_info.get('car_x', 0.0):.1f}, {final_info.get('car_y', 0.0):.1f})"
            )
            print(
                f"  CSV Row -> model={model_number}, episode={rows_written}, "
                f"close distance={metrics['close_distance']}, near_miss={metrics['near_miss']}, "
                f"MSE=0.000, Reward={metrics['reward']:.3f}, Avg reward={avg_reward_all:.3f}, "
                f"time(ms)={metrics['time_ms']}, timeframe={metrics['timeframe']}, "
                f"steps={metrics['steps']}, progress_pct={metrics['progress_pct']:.1f}, "
                f"reached_finish={metrics['reached_finish']}, "
                f"min_ttc={format_ttc_ms(metrics['min_ttc'])}, "
                f"avg_ttc={format_ttc_ms(metrics['avg_ttc'])}"
            )

    print(f"\nEvaluate models ended. Rows written: {rows_written}.")


def run_visualization(
    model_path=None,
    episodes=10,
    manual_mode=False,
    allstage=False,
    tester=False,
    experiment=False,
    random_mode=False,
    neuron_mode=False,
    algo=ALGO_DQN,
):
    """Run visualization with trained model or manual control"""
    experiment_mode = bool(experiment)
    random_mode = bool(random_mode)
    if random_mode and experiment_mode:
        print("Random mode ignores --experiment.")
        experiment_mode = False
    if experiment_mode and allstage:
        print("Experiment mode ignores --allstage.")
        allstage = False
    if random_mode and allstage:
        print("Random mode ignores --allstage.")
        allstage = False
    if random_mode and tester:
        print("Random mode ignores --tester.")
        tester = False
    if experiment_mode:

        obstacles_cfg = [[]]
    elif random_mode:
        obstacles_cfg = [[]]
    else:
        obstacles_cfg = TEST_OBSTACLES if tester else OBSTACLES

    if experiment_mode:
        env = CarEnvironment(obstacles_config=obstacles_cfg, disable_finish=True)
    elif random_mode:
        env = CarEnvironment(obstacles_config=obstacles_cfg, disable_finish=False)
    elif allstage:
        env = CarEnvironment(curriculum_stage=0, obstacles_config=obstacles_cfg)
    else:
        env = CarEnvironment(obstacles_config=obstacles_cfg)

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
        memory_size=MEMORY_SIZE,
        algo=algo,
    )

    if model_path and os.path.exists(model_path):
        agent.load(model_path)
        agent.epsilon = 0.0
        print(f"Loaded model from {model_path}")
    else:
        print("No model loaded, using random actions or manual control")
        agent.epsilon = 0.0

    renderer = GameRenderer(
        env,
        scale=DEFAULT_SCALE,
        experiment_mode=experiment_mode,
        neuron_mode=neuron_mode,
    )
    planner = ExperimentObstaclePlanner() if experiment_mode else None
    random_generator = RandomObstacleGenerator() if random_mode else None
    visualize_csv_path = get_next_visualize_csv_path()
    visualize_rewards = []
    print(f"Visualize CSV: {visualize_csv_path}")

    print("\n=== Visualization Mode ===")
    print("Controls:")
    print("  Arrow Keys: Manual control (Left/Up/Right/Down)")
    print("  P: Pause/Resume")
    print("  R: Reset episode")
    print(f"  1: Toggle speed x{KEYONE_MULTIPLIER}")
    print("  2: Toggle slow motion")
    print("  Q: Quit")
    print("  I: Toggle Indicators")
    if experiment_mode:
        print("  Mouse: Use Obstacle Controls panel")
        print("Obstacle source: EXPERIMENT (empty start)")
    elif random_mode:
        print(
            f"Obstacle source: RANDOM finite (start={int(startRandom)}, gap={int(gapRandom)}, "
            f"rows={int(maxRandom)}, 1-2 vehicle(s) per row)"
        )
    else:
        print(f"Obstacle source: {'TEST_OBSTACLES' if tester else 'OBSTACLES'}")
    if neuron_mode:
        print("Neuron trace: ON (normalized input, hidden activations, weights/bias, Q)")
    print("=" * 30)

    episode = 0
    running = True
    paused = False


    num_stages = get_num_stages(obstacles_cfg)
    current_stage = 0
    consecutive_success = 0
    csv_headers = get_visualize_csv_headers()
    completed_episode_rows = []

    def save_visualize_csv(current_row=None):
        try:
            with open(visualize_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_headers)
                writer.writeheader()
                for r in completed_episode_rows:
                    writer.writerow(r)
                if current_row is not None:
                    writer.writerow(current_row)
        except Exception as e:
            print(f"Error saving visualize CSV: {e}")

    # Write initial header
    save_visualize_csv()

    info = {}
    total_reward = 0
    close_distance_total = 0
    near_miss_total = 0
    step_count = 0
    episode_sensor_mins = [float("inf")] * len(SENSOR_LOG_NAMES)
    episode_ttc_samples = []

    def get_current_episode_row(curr_duration_ms):
        finish_y = info.get("finish_line_y", getattr(env, "finish_line_y", 2000))
        car_y = info.get("car_y", getattr(env, "car_y", 0.0))
        raw_prog = (car_y / finish_y) * 100.0 if finish_y != 0 else 0.0
        rounded_prog = round(raw_prog / 5.0) * 5
        prog_pct = (
            100.0 if info.get("reached_finish") else min(95.0, max(0.0, rounded_prog))
        )
        temp_rewards = visualize_rewards + [total_reward]
        avg_rew = (
            float(sum(temp_rewards) / len(temp_rewards))
            if temp_rewards
            else 0.0
        )
        w_step = int(
            round(float(info.get("world_distance", getattr(env, "world_distance", 0.0))))
        )
        t_summary = summarize_ttc_samples(episode_ttc_samples)
        return build_visualize_episode_row(
            episode=episode,
            close_distance=close_distance_total,
            near_miss=near_miss_total,
            mse=0.0,
            reward=total_reward,
            avg_reward=avg_rew,
            time_ms=curr_duration_ms,
            timeframe=step_count,
            steps=w_step,
            progress_pct=prog_pct,
            reached_finish=1 if info.get("reached_finish") else 0,
            min_ttc=t_summary["min_ttc"],
            avg_ttc=t_summary["avg_ttc"],
            min_sensor_logs=build_min_sensor_logs(env, episode_sensor_mins),
        )

    while running and (experiment_mode or episode < episodes):
        state = env.reset()
        if random_generator is not None:
            random_generator.append_all_obstacles(env)
            state = env._get_state()
        total_reward = 0
        done = False
        close_distance_total = 0
        near_miss_total = 0
        episode += 1

        if allstage:
            print(
                f"\nStarting Episode {episode} (Stage {current_stage + 1}/{num_stages}, streak {consecutive_success}/{ALLSTAGE_CONSECUTIVE_REQ})"
            )
        else:
            print(f"\nStarting Episode {episode}")

        start_ticks = pygame.time.get_ticks()
        step_count = 0
        action = 4 if manual_mode else 1
        nn_output = None
        neuron_data = None


        current_vis_action = action
        episode_sensor_mins = [float("inf")] * len(SENSOR_LOG_NAMES)
        episode_ttc_samples = []

        info = {
            "car_x": getattr(env, "car_x", 0.0),
            "car_y": getattr(env, "car_y", 0.0),
            "reached_finish": False,
            "finish_line_y": getattr(env, "finish_line_y", 2000.0),
            "world_distance": getattr(env, "world_distance", 0.0),
        }
        last_log_ticks = start_ticks

        while running and not done:
            running, reset, manual_action, pause_toggle, experiment_actions = (
                renderer.handle_events(paused)
            )

            if planner is not None and experiment_actions:
                for experiment_action in experiment_actions:
                    action_type = experiment_action.get("type")
                    if action_type == "toggle_lane":
                        planner.toggle_lane(experiment_action.get("lane"))
                    elif action_type == "distance_minus":
                        planner.decrement_distance()
                    elif action_type == "distance_plus":
                        planner.increment_distance()
                    elif action_type == "add_list":
                        if planner.add_current_selection():
                            latest = planner.to_spawn_list[-1]
                            lanes_text = "+".join(latest["lanes"])
                            print(
                                f"[Experiment] Added: lanes={lanes_text}, distance={latest['distance']}"
                            )
                        else:
                            print("[Experiment] Select at least one path before ADD LIST.")
                    elif action_type == "spawn":
                        spawn_plan = planner.build_spawn_plan(env.car_y)
                        obstacle_configs = planner.build_obstacle_configs(env.car_y)
                        if obstacle_configs:
                            added_count = env.append_obstacles(obstacle_configs)
                            print(
                                f"[Experiment] Spawned {added_count} obstacle(s) from {len(spawn_plan)} list item(s)."
                            )
                        else:
                            print("[Experiment] To Spawn Lists is empty.")
                    elif action_type == "clear_spawn":
                        planner.clear_spawn_list()
                        print("[Experiment] Cleared To Spawn Lists.")

            if pause_toggle:
                paused = not paused
                print(f"{'PAUSED' if paused else 'RESUMED'}")

            if paused:
                if renderer.last_render_info is None:
                    paused_info = env.render_info()
                    if nn_output is not None:
                        paused_info["nn_output"] = nn_output
                    paused_info["last_action"] = action
                    paused_info["near_miss_total"] = near_miss_total
                    renderer.render(
                        paused_info,
                        episode,
                        total_reward,
                        agent.epsilon,
                        step_count,
                        paused=True,
                        experiment_data=planner.snapshot() if planner else None,
                        neuron_data=neuron_data,
                    )
                else:
                    renderer.render(
                        renderer.last_render_info,
                        episode,
                        total_reward,
                        agent.epsilon,
                        step_count,
                        paused=True,
                        experiment_data=planner.snapshot() if planner else None,
                        neuron_data=neuron_data,
                    )
                continue

            if reset:
                break


            steps_this_frame = renderer.get_steps_per_frame()
            last_action = action

            for _ in range(steps_this_frame):
                if manual_mode:
                    if manual_action is not None:
                        last_action = manual_action
                    if last_action is None:
                        last_action = 4
                else:

                    if step_count % DECISION_INTERVAL == 0:
                        current_vis_action = agent.select_action(state, training=False)
                    last_action = current_vis_action


                is_decision_step = step_count % DECISION_INTERVAL == 0
                next_state, reward, done, info = env.step(
                    last_action, apply_steering=is_decision_step
                )
                current_ttc_ms = info.get("ttc_ms")
                if current_ttc_ms is None:
                    current_ttc_ms = calculate_env_ttc_ms(env)
                if current_ttc_ms is not None:
                    episode_ttc_samples.append(float(current_ttc_ms))
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
                if random_generator is not None:
                    random_generator.append_due_obstacles(env)
                    next_state = env._get_state()

                total_reward += reward
                close_distance_total += int(info.get("warning_close_count", 0))
                near_miss_total += int(info.get("near_miss", 0))
                state = next_state
                step_count += 1

                if done:
                    break

            action = last_action


            try:
                nn_output = agent.get_q_values(state)
            except Exception:
                nn_output = None
            try:
                neuron_data = build_neuron_trace(agent, state) if neuron_mode else None
            except Exception as e:
                neuron_data = {"error": str(e)}

            render_info = env.render_info()
            if nn_output is not None:
                render_info["nn_output"] = nn_output
            render_info["last_action"] = action
            render_info["near_miss_total"] = near_miss_total
            renderer.render(
                render_info,
                episode,
                total_reward,
                agent.epsilon,
                step_count,
                paused,
                planner.snapshot() if planner else None,
                neuron_data,
            )

            if not paused:
                now_ticks = pygame.time.get_ticks()
                if now_ticks - last_log_ticks >= visualize_logs_sec * 1000:
                    current_row = get_current_episode_row(now_ticks - start_ticks)
                    save_visualize_csv(current_row)
                    last_log_ticks = now_ticks

        if done:
            end_ticks = pygame.time.get_ticks()
            duration_ms = end_ticks - start_ticks
            seconds = int(duration_ms / 1000)
            m, s = divmod(seconds, 60)
            h, m = divmod(m, 60)
            time_str = f"{h:02d}:{m:02d}:{s:02d}"

            finish_y = info.get("finish_line_y", getattr(env, "finish_line_y", 2000))
            raw_prog = (info["car_y"] / finish_y) * 100.0 if finish_y != 0 else 0.0
            rounded_prog = round(raw_prog / 5.0) * 5
            progress_pct = (
                100.0 if info.get("reached_finish") else min(95.0, max(0.0, rounded_prog))
            )
            mse_value = 0.0
            visualize_rewards.append(float(total_reward))
            avg_reward_all = (
                float(sum(visualize_rewards) / len(visualize_rewards))
                if visualize_rewards
                else 0.0
            )
            world_step = int(
                round(float(info.get("world_distance", getattr(env, "world_distance", 0.0))))
            )
            ttc_summary = summarize_ttc_samples(episode_ttc_samples)

            final_row = build_visualize_episode_row(
                episode=episode,
                close_distance=close_distance_total,
                near_miss=near_miss_total,
                mse=mse_value,
                reward=total_reward,
                avg_reward=avg_reward_all,
                time_ms=duration_ms,
                timeframe=step_count,
                steps=world_step,
                progress_pct=progress_pct,
                reached_finish=1 if info.get("reached_finish") else 0,
                min_ttc=ttc_summary["min_ttc"],
                avg_ttc=ttc_summary["avg_ttc"],
                min_sensor_logs=build_min_sensor_logs(env, episode_sensor_mins),
            )
            completed_episode_rows.append(final_row)
            save_visualize_csv()

            print(f"Episode {episode} finished!")
            print(f"  Total Reward: {total_reward:.2f}")
            print("  MSE: 0.000 (Inference)")
            print(f"  Close Distance: {close_distance_total}")
            print(f"  Near Miss: {near_miss_total}")
            print(f"  Avg Reward: {avg_reward_all:.2f}")
            print(f"  Time: {time_str}")
            print(f"  Timeframe: {step_count}")
            print(f"  Step: {world_step}")
            print(f"  min_ttc: {format_ttc_ms(ttc_summary['min_ttc'])} ms")
            print(f"  avg_ttc: {format_ttc_ms(ttc_summary['avg_ttc'])} ms")
            print(f"  Progress: {progress_pct:.1f}%")
            print(f"  Final Position: ({info['car_x']:.1f}, {info['car_y']:.1f})")
            print(
                f"  CSV Row -> episode={episode}, close distance={close_distance_total}, "
                f"near_miss={near_miss_total}, "
                f"MSE={mse_value:.3f}, Reward={float(total_reward):.3f}, Avg reward={avg_reward_all:.3f}, "
                f"time(ms)={int(duration_ms)}, timeframe={int(step_count)}, steps={int(world_step)}, "
                f"progress_pct={float(progress_pct):.1f}, reached_finish={1 if info.get('reached_finish') else 0}, "
                f"min_ttc={format_ttc_ms(ttc_summary['min_ttc'])}, avg_ttc={format_ttc_ms(ttc_summary['avg_ttc'])}"
            )

            if allstage:
                if info.get("reached_finish"):
                    consecutive_success += 1
                else:
                    consecutive_success = 0


                if (
                    current_stage < (num_stages - 1)
                    and consecutive_success >= ALLSTAGE_CONSECUTIVE_REQ
                ):
                    current_stage += 1
                    consecutive_success = 0
                    try:
                        env.set_curriculum_stage(current_stage)
                        renderer.env = env
                    except Exception as e:
                        print(f"Warning: could not advance stage: {e}")
                    else:
                        print(f"[OK] ADVANCED TO STAGE {current_stage + 1}/{num_stages}")


            end_pause_ms = int(max(50, 1000 / max(1, renderer.get_steps_per_frame())))
            pygame.time.wait(end_pause_ms)

    renderer.close()
    print("\nVisualization ended.")


def run_speedtest_visualization():
    """
    Run endless-road speed test mode.

    Controls:
      - Up Arrow: queue FAST straight decision (action 4)
      - Down Arrow: queue SLOW straight decision (action 1)
      - Queued decision is applied only at the next decision boundary
        (step % DECISION_INTERVAL == 0).
    """
    env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
    env.max_steps = 10**12
    renderer = GameRenderer(env, scale=DEFAULT_SCALE, experiment_mode=False)

    print("\n=== Speed Test Mode ===")
    print("Controls:")
    print("  UP: Queue FAST straight decision")
    print("  DOWN: Queue SLOW straight decision")
    print("  P: Pause/Resume")
    print("  R: Reset")
    print(f"  1: Toggle speed x{KEYONE_MULTIPLIER}")
    print("  2: Toggle slow motion")
    print("  Q: Quit")
    print("=" * 30)

    state = env.reset()
    total_reward = 0.0
    step_count = 0
    episode = 1
    running = True
    paused = False
    nn_output = None


    current_decision_action = 4
    pending_decision_action = 4

    def _decision_label(action_id):
        return "FAST" if int(action_id) == 4 else "SLOW"

    while running:
        running, reset, manual_action, pause_toggle, _experiment_actions = (
            renderer.handle_events(paused)
        )

        if manual_action in [1, 4]:
            pending_decision_action = int(manual_action)
            print(
                f"[SpeedTest] Queued {_decision_label(pending_decision_action)} decision "
                f"(applies at next decision step)."
            )

        if pause_toggle:
            paused = not paused
            print(f"{'PAUSED' if paused else 'RESUMED'}")

        if reset:
            state = env.reset()
            total_reward = 0.0
            step_count = 0
            current_decision_action = pending_decision_action
            nn_output = None
            print("[SpeedTest] Reset.")
            continue

        if paused:
            if renderer.last_render_info is None:
                paused_info = env.render_info()
                if nn_output is not None:
                    paused_info["nn_output"] = nn_output
                paused_info["last_action"] = current_decision_action
                renderer.render(
                    paused_info,
                    episode,
                    total_reward,
                    0.0,
                    step_count,
                    paused=True,
                    experiment_data=None,
                )
            else:
                renderer.render(
                    renderer.last_render_info,
                    episode,
                    total_reward,
                    0.0,
                    step_count,
                    paused=True,
                    experiment_data=None,
                )
            continue

        steps_this_frame = renderer.get_steps_per_frame()
        for _ in range(steps_this_frame):
            is_decision_step = step_count % DECISION_INTERVAL == 0
            if is_decision_step:
                prev_action = current_decision_action
                current_decision_action = pending_decision_action
                if current_decision_action != prev_action:
                    print(
                        f"[SpeedTest] Timeframe {step_count}: applied "
                        f"{_decision_label(current_decision_action)} decision."
                    )

            next_state, reward, done, _info = env.step(
                current_decision_action, apply_steering=is_decision_step
            )
            state = next_state
            total_reward += reward
            step_count += 1

            if done:

                state = env.reset()
                total_reward = 0.0
                step_count = 0
                current_decision_action = pending_decision_action
                print("[SpeedTest] Auto-reset after terminal state.")
                break

        render_info = env.render_info()
        if nn_output is not None:
            render_info["nn_output"] = nn_output
        render_info["last_action"] = current_decision_action
        renderer.render(
            render_info,
            episode,
            total_reward,
            0.0,
            step_count,
            paused=False,
            experiment_data=None,
        )

    renderer.close()
    print("\nSpeed test ended.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DQN Car Navigation Visualization")
    parser.add_argument(
        "--model",
        type=str,
        default="models/best_model.pth",
        help="Path to trained model",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to visualize"
    )
    parser.add_argument(
        "--manual", action="store_true", help="Use manual keyboard control"
    )
    parser.add_argument(
        "--allstage",
        action="store_true",
        help="Run visualization across all curriculum stages (advance after consecutive finishes)",
    )
    parser.add_argument(
        "--tester",
        action="store_true",
        help="Use TEST_OBSTACLES from main_constant.py instead of OBSTACLES",
    )
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Enable obstacle experiment mode (no finish line, custom spawn controls)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Enable finite random obstacles (startRandom/gapRandom/maxRandom, 1-2 vehicles per row)",
    )
    parser.add_argument(
        "--neuron",
        action="store_true",
        help="Show detailed neural-network forward-pass panel (input, weights, bias, hidden activations, Q-values)",
    )
    parser.add_argument(
        "--speedtest",
        action="store_true",
        help="Run endless speed test (Up/Down queue fast/slow straight decisions at interval boundaries)",
    )
    parser.add_argument(
        "--evamodel",
        action="store_true",
        help="Headless-evaluate every non-check model in ./models once and write visualize_logs/evaluate.csv",
    )

    algo_group = parser.add_mutually_exclusive_group()
    algo_group.add_argument(
        "--ddqn",
        action="store_true",
        help="Use Double DQN network architecture for visualization",
    )
    algo_group.add_argument(
        "--d3qn",
        action="store_true",
        help="Use Dueling Double DQN (D3QN) network architecture for visualization",
    )

    args = parser.parse_args()

    # Resolve algorithm choice
    if args.d3qn:
        algo = "d3qn"
    elif args.ddqn:
        algo = "ddqn"
    else:
        algo = "dqn"

    if args.speedtest and args.evamodel:
        parser.error("--speedtest cannot be used with --evamodel")

    if args.speedtest:
        run_speedtest_visualization()
    elif args.evamodel:
        run_evamodel(algo=algo)
    else:
        run_visualization(
            model_path=args.model,
            episodes=args.episodes,
            manual_mode=args.manual,
            allstage=args.allstage,
            tester=args.tester,
            experiment=args.experiment,
            random_mode=args.random,
            neuron_mode=args.neuron,
            algo=algo,
        )
