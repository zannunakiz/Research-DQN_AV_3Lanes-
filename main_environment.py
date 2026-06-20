"""
DQN Environment for 2D Car Navigation
3-lane straight road with car using 7 sensors
Goal: Drive straight to finish line with scrolling road effect
Supports curriculum learning with multiple obstacle stages
"""

import numpy as np
import math
import random
from func_ttc import calculate_env_ttc_details
from main_constant import (
    FINISH_DISTANCE,
    ROAD_WIDTH,
    LANE_COUNT,
    CAR_WIDTH,
    CAR_HEIGHT,
    nearmiss_distance,
    OBSTACLE_WIDTH,
    OBSTACLE_HEIGHT,
    OBSTACLES,
    OBSTACLE_SPEED,
    REWARD_PROGRESS,
    REWARD_LANE_CENTER_MAX,
    REWARD_STRAIGHT_ANGLE,
    REWARD_FINISH,
    LANE_CENTER_REWARD_WIDTH,
    LEFT_LR_OFFSETX,
    RIGHT_LR_OFFSETX,
    CENTER_LR_OFFSETX,
    LEFT_OBSTACLE_OFFSETX,
    RIGHT_OBSTACLE_OFFSETX,
    CENTER_OBSTACLE_OFFSETX,
    REWARD_FAST_CLEAR,
    PENALTY_SLOW_WHEN_CLEAR,
    PENALTY_COLLISION,
    PENALTY_TIMEOUT,
    PENALTY_NOT_IN_CENTER,
    PENALTY_WARNING_DISTANCE_FRONT,
    PENALTY_WARNING_DISTANCE_SIDES,
    STRAIGHT_ANGLE_THRESHOLD,
    OBSTACLE_WARNING_DISTANCE_FRONT,
    OBSTACLE_WARNING_DISTANCE_SIDES,

    SENSOR_R2,
    SENSOR_R1,
    SENSOR_F,
    SENSOR_L1,
    SENSOR_L2,
    SENSOR_SR,
    SENSOR_SL,
    SENSOR_ANGLES,
    SENSOR_NOISE_RANGE,

    CAR_MAX_SPEED,
    CAR_MIN_SPEED,
    SPEED_UP,
    SPEED_DOWN,
    TURNING_ANGLE,
    DECISION_INTERVAL,
)


def get_flat_obstacles(obstacles_config, stage=None):
    """
    Get a flat list of obstacles from the OBSTACLES configuration.
    Supports both old format (flat list) and new curriculum format (list of lists).

    Args:
        obstacles_config: The OBSTACLES configuration from main_constant.py
        stage: The curriculum stage index (0-based). If None, returns all obstacles
               for backward compatibility with flat format.

    Returns:
        A flat list of obstacle dictionaries
    """
    if not obstacles_config:
        return []


    if obstacles_config and isinstance(obstacles_config[0], list):

        if stage is not None and 0 <= stage < len(obstacles_config):
            return obstacles_config[stage]
        else:

            return obstacles_config[0] if obstacles_config else []
    else:

        return obstacles_config


def get_num_stages(obstacles_config):
    """
    Get the number of curriculum stages from OBSTACLES configuration.

    Args:
        obstacles_config: The OBSTACLES configuration from main_constant.py

    Returns:
        Number of stages (1 for flat format, len(list) for curriculum format)
    """
    if not obstacles_config:
        return 1


    if obstacles_config and isinstance(obstacles_config[0], list):
        return len(obstacles_config)
    else:

        return 1


class CarEnvironment:
    def __init__(
        self,
        road_length=None,
        road_width=None,
        lane_count=None,
        curriculum_stage=0,
        obstacles_config=None,
        disable_finish=False,
    ):

        self.road_width = road_width if road_width is not None else ROAD_WIDTH
        self.lane_count = lane_count if lane_count is not None else LANE_COUNT
        self.lane_width = self.road_width // self.lane_count


        self.obstacles_config = (
            obstacles_config if obstacles_config is not None else OBSTACLES
        )
        self.num_stages = get_num_stages(self.obstacles_config)
        self.current_stage = min(curriculum_stage, self.num_stages - 1)
        self.disable_finish = bool(disable_finish)


        self.car_width = CAR_WIDTH
        self.car_height = CAR_HEIGHT
        self.nearmiss_distance = float(nearmiss_distance)
        self.max_speed = CAR_MAX_SPEED
        self.min_speed = CAR_MIN_SPEED

        self.min_speed_kmh = 55.0
        self.max_speed_kmh = 75.0
        kmh_span = max(self.max_speed_kmh - self.min_speed_kmh, 1e-9)
        self.world_speed_per_kmh = (self.max_speed - self.min_speed) / kmh_span
        self.speed_up_kmh = float(SPEED_UP)
        self.speed_down_kmh = float(SPEED_DOWN)
        self.last_target_speed = self.min_speed
        self.last_speed_delta = 0.0
        self.speed_interval_steps_remaining = 0


        self.base_car_angle = 90.0
        self.steering_step = float(TURNING_ANGLE)
        self.decision_interval = max(1, int(DECISION_INTERVAL))
        self.steering_offset = 0.0
        self.steering_target_offset = 0.0
        self.steering_anim_rate = self.steering_step / float(self.decision_interval)


        try:
            self.obstacle_speed = min(OBSTACLE_SPEED, self.max_speed - 0.1)
            if self.obstacle_speed < 0:
                self.obstacle_speed = 0.0
        except NameError:
            self.obstacle_speed = 0.0


        self.obstacle_lane_change_speed = max(1.0, self.lane_width / 10.0)


        self.num_sensors = 7

        self.sensor_ranges = [
            SENSOR_R2,
            SENSOR_R1,
            SENSOR_F,
            SENSOR_L1,
            SENSOR_L2,
            SENSOR_SR,
            SENSOR_SL,
        ]

        self.sensor_range = max(self.sensor_ranges)
        self.sensor_noise_values = self._build_sensor_noise_values(SENSOR_NOISE_RANGE)


        self.state_size = self.num_sensors + 1


        self.action_size = 6


        self.last_action = 1


        self.max_steps = 999999


        self._setup_stage_obstacles()


        self.reset()

    def _clamp_lane_index(self, lane_index):
        if lane_index is None:
            return None
        try:
            lane_value = int(lane_index)
        except (TypeError, ValueError):
            return None
        return max(0, min(self.lane_count - 1, lane_value))

    def _lane_center_x(self, lane_index):
        return (lane_index * self.lane_width) + (self.lane_width // 2)

    def _lane_center_metrics(self):
        """Return lane-center geometry and whether car is inside reward zone."""
        lane_index = self._clamp_lane_index(int(self.car_x // self.lane_width))
        if lane_index is None:
            lane_index = self.lane_count // 2

        lane_center = self._lane_center_x(lane_index)
        try:
            if lane_index == 0:
                lane_center = float(lane_center) + float(LEFT_LR_OFFSETX)
            elif lane_index == 1:
                lane_center = float(lane_center) + float(CENTER_LR_OFFSETX)
            elif lane_index == 2:
                lane_center = float(lane_center) + float(RIGHT_LR_OFFSETX)
        except Exception:
            lane_center = self._lane_center_x(lane_index)

        distance_from_center = abs(self.car_x - lane_center)
        lane_center_reward_width = min(
            float(self.lane_width), float(LANE_CENTER_REWARD_WIDTH)
        )
        lane_center_reward_half_width = max(lane_center_reward_width / 2.0, 1e-6)
        in_center_reward_zone = distance_from_center <= lane_center_reward_half_width
        return (
            lane_center,
            distance_from_center,
            lane_center_reward_half_width,
            in_center_reward_zone,
        )

    @staticmethod
    def _clamp(value, min_value, max_value):
        return max(min_value, min(value, max_value))

    @staticmethod
    def _build_sensor_noise_values(noise_values):
        """Build discrete noise options and always include zero-noise option."""
        options = []
        seen = set()
        try:
            for value in list(noise_values):
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(numeric_value) and numeric_value not in seen:
                    options.append(numeric_value)
                    seen.add(numeric_value)
        except TypeError:
            pass

        if 0.0 not in seen:
            options.append(0.0)
        return options if options else [0.0]

    def _sample_sensor_noise(self):
        return float(random.choice(self.sensor_noise_values))

    def _measure_sensor(self, sensor_index, apply_noise=True):
        """Measure one sensor and optionally apply discrete random noise."""
        rel_angle = SENSOR_ANGLES[sensor_index]
        abs_angle = self.car_angle + rel_angle
        max_range = float(self.sensor_ranges[sensor_index])
        base_distance = float(self._cast_ray(abs_angle, max_range))
        noise = self._sample_sensor_noise() if apply_noise else 0.0
        noisy_distance = self._clamp(base_distance + noise, 0.0, max_range)
        normalized = self._clamp(noisy_distance / max_range, 0.0, 1.0)
        return {
            "index": int(sensor_index),
            "angle": float(abs_angle),
            "base_distance": float(base_distance),
            "noise": float(noise),
            "distance": float(noisy_distance),
            "normalized": float(normalized),
        }

    def _start_speed_interval(self, decision_delta_kmh):
        """Prepare a new speed target to be reached at the end of one decision interval."""
        delta_world = float(decision_delta_kmh) * float(self.world_speed_per_kmh)
        target_speed = self._clamp(
            float(self.car_speed) + delta_world, self.min_speed, self.max_speed
        )
        self.last_target_speed = target_speed
        self.speed_interval_steps_remaining = max(1, int(self.decision_interval))

    def _apply_interval_speed_dynamics(self):
        """Move speed gradually toward the current interval target over remaining steps."""
        if self.speed_interval_steps_remaining <= 0:
            self.last_target_speed = float(self.car_speed)
            self.last_speed_delta = 0.0
            return

        steps_left = float(self.speed_interval_steps_remaining)
        speed_error = float(self.last_target_speed) - float(self.car_speed)
        speed_delta = speed_error / steps_left
        proposed_speed = self._clamp(
            float(self.car_speed) + speed_delta, self.min_speed, self.max_speed
        )

        self.last_speed_delta = proposed_speed - float(self.car_speed)
        self.car_speed = proposed_speed
        self.speed_interval_steps_remaining -= 1

    def _resolve_obstacle_speed(self, speed_value):
        """Resolve obstacle speed from config with safe fallback to default speed."""
        default_speed = float(getattr(self, "obstacle_speed", 0.0))
        max_obstacle_speed = max(0.0, float(self.max_speed) - 0.1)
        if speed_value is None:
            return min(default_speed, max_obstacle_speed)
        try:
            resolved_speed = float(speed_value)
        except (TypeError, ValueError):
            return min(default_speed, max_obstacle_speed)
        if resolved_speed < 0:
            return 0.0
        return min(resolved_speed, max_obstacle_speed)

    def _build_obstacle_state(self, obs_config):
        if not isinstance(obs_config, dict):
            return None

        lane = self._clamp_lane_index(obs_config.get("lane"))
        y_pos = obs_config.get("y")
        if lane is None or y_pos is None:
            return None
        try:
            y_pos = float(y_pos)
        except (TypeError, ValueError):
            return None

        x_pos = self._lane_center_x(lane)

        try:
            if lane == 0:
                x_pos = float(x_pos) + float(LEFT_OBSTACLE_OFFSETX)
            elif lane == 1:
                x_pos = float(x_pos) + float(CENTER_OBSTACLE_OFFSETX)
            elif lane == 2:
                x_pos = float(x_pos) + float(RIGHT_OBSTACLE_OFFSETX)
        except Exception:

            x_pos = self._lane_center_x(lane)
        obstacle = {
            "x": x_pos,
            "y": y_pos,
            "width": OBSTACLE_WIDTH,
            "height": OBSTACLE_HEIGHT,
            "lane": lane,
            "speed": self._resolve_obstacle_speed(obs_config.get("speed")),
        }

        switch_lane = (
            self._clamp_lane_index(obs_config.get("switch"))
            if "switch" in obs_config
            else None
        )
        stopat_lane = (
            self._clamp_lane_index(obs_config.get("stopat"))
            if "stopat" in obs_config
            else None
        )

        pattern = [lane]
        if switch_lane is not None:
            pattern.append(switch_lane)
        if stopat_lane is not None:
            pattern.append(stopat_lane)

        if len(pattern) >= 2:
            repeat = obs_config.get("repeat", None)
            delay = obs_config.get("delay", 0)

            if repeat is None:
                repeat = 1
            try:
                repeat = int(repeat)
            except (TypeError, ValueError):
                repeat = 0
            repeat = max(0, repeat)

            try:
                delay = int(delay)
            except (TypeError, ValueError):
                delay = 0
            delay = max(0, delay)

            has_change = any(
                pattern[i] != pattern[i - 1] for i in range(1, len(pattern))
            )
            if repeat > 0 and has_change:
                obstacle.update(
                    {
                        "move_enabled": True,
                        "pattern_lanes": pattern,
                        "pattern_index": 0,
                        "cycles_done": 0,
                        "repeat": repeat,
                        "delay": delay,
                        "wait_steps": delay,
                        "moving": False,
                        "target_lane": None,
                        "target_x": None,
                        "target_index": None,
                        "lane_change_speed": self.obstacle_lane_change_speed,
                    }
                )

        return obstacle

    def _next_obstacle_target(self, obs):
        if not obs.get("move_enabled"):
            return None, None
        if obs.get("cycles_done", 0) >= obs.get("repeat", 0):
            return None, None

        pattern = obs.get("pattern_lanes", [])
        if not pattern:
            return None, None

        idx = obs.get("pattern_index", 0)
        if idx < len(pattern) - 1:
            return pattern[idx + 1], idx + 1


        return pattern[0], 0

    def _apply_obstacle_target(self, obs, target_lane, target_index):
        obs["lane"] = target_lane
        obs["pattern_index"] = target_index
        if target_index == len(obs.get("pattern_lanes", [])) - 1:
            obs["cycles_done"] = obs.get("cycles_done", 0) + 1

        obs["moving"] = False
        obs["target_lane"] = None
        obs["target_x"] = None
        obs["target_index"] = None

        if obs.get("cycles_done", 0) >= obs.get("repeat", 0):
            obs["move_enabled"] = False

    def _update_obstacle_lane_switch(self, obs):
        if not obs.get("move_enabled"):
            return

        if obs.get("moving"):
            target_x = obs.get("target_x")
            if target_x is None:
                obs["moving"] = False
                return

            speed = obs.get("lane_change_speed", self.obstacle_lane_change_speed)
            dx = target_x - obs["x"]
            if abs(dx) <= speed:
                obs["x"] = target_x
                target_lane = obs.get("target_lane")
                target_index = obs.get("target_index")
                if target_lane is not None and target_index is not None:
                    self._apply_obstacle_target(obs, target_lane, target_index)
                    if obs.get("move_enabled"):
                        obs["wait_steps"] = obs.get("delay", 0)
                else:
                    obs["moving"] = False
            else:
                obs["x"] += speed if dx > 0 else -speed
            return

        wait_steps = obs.get("wait_steps", 0)
        if wait_steps > 0:
            obs["wait_steps"] = wait_steps - 1
            return

        target_lane, target_index = self._next_obstacle_target(obs)
        if target_lane is None:
            obs["move_enabled"] = False
            return

        target_x = self._lane_center_x(target_lane)

        try:
            if target_lane == 0:
                target_x = float(target_x) + float(LEFT_OBSTACLE_OFFSETX)
            elif target_lane == 1:
                target_x = float(target_x) + float(CENTER_OBSTACLE_OFFSETX)
            elif target_lane == 2:
                target_x = float(target_x) + float(RIGHT_OBSTACLE_OFFSETX)
        except Exception:
            target_x = self._lane_center_x(target_lane)
        if abs(target_x - obs["x"]) < 1e-6:
            self._apply_obstacle_target(obs, target_lane, target_index)
            if obs.get("move_enabled"):
                obs["wait_steps"] = obs.get("delay", 0)
            return

        obs["target_lane"] = target_lane
        obs["target_index"] = target_index
        obs["target_x"] = target_x
        obs["moving"] = True

    def _setup_stage_obstacles(self):
        """Setup obstacles for the current curriculum stage"""

        stage_obstacles = get_flat_obstacles(
            self.obstacles_config, self.current_stage
        )


        self.initial_obstacles = []
        for obs_config in stage_obstacles:
            if not isinstance(obs_config, dict):
                continue
            lane = obs_config.get("lane")
            y_pos = obs_config.get("y")
            if lane is None or y_pos is None:
                continue
            self.initial_obstacles.append(
                {
                    "lane": lane,
                    "y": y_pos,
                    "speed": obs_config.get("speed"),
                    "switch": obs_config.get("switch"),
                    "stopat": obs_config.get("stopat"),
                    "repeat": obs_config.get("repeat"),
                    "delay": obs_config.get("delay"),
                }
            )


        self.obstacles = []
        self._refresh_finish_line()

    def _refresh_finish_line(self):
        """Recompute finish-line metadata after obstacle changes."""
        if self.obstacles:
            max_obstacle_y = max(obs["y"] for obs in self.obstacles)
        elif self.initial_obstacles:
            max_obstacle_y = max(obs["y"] for obs in self.initial_obstacles)
        else:
            max_obstacle_y = 100.0


        self.road_length = max_obstacle_y + FINISH_DISTANCE
        if self.disable_finish:
            self.finish_line_y = float("inf")
        else:
            self.finish_line_y = self.road_length

    def clear_obstacles(self):
        """Clear active and initial obstacle sets."""
        self.initial_obstacles = []
        self.obstacles = []
        self._refresh_finish_line()

    def append_obstacles(self, obstacle_configs):
        """
        Append runtime obstacles from lightweight configs.

        Args:
            obstacle_configs: iterable of dicts with at least lane and y.

        Returns:
            Number of successfully added obstacles.
        """
        added = 0
        if obstacle_configs is None:
            return added

        for obs_config in obstacle_configs:
            obstacle_state = self._build_obstacle_state(obs_config)
            if obstacle_state is not None:
                self.obstacles.append(obstacle_state)
                added += 1

        self._refresh_finish_line()
        return added

    def set_curriculum_stage(self, stage):
        """
        Change the curriculum stage and reinitialize obstacles.

        Args:
            stage: The new stage index (0-based)

        Returns:
            True if stage changed, False if invalid stage
        """
        if 0 <= stage < self.num_stages:
            self.current_stage = stage
            self._setup_stage_obstacles()
            return True
        return False

    def reset(self):
        """Reset environment to initial state"""

        center_lane = self.lane_count // 2
        self.car_x = (center_lane * self.lane_width) + (self.lane_width // 2)
        self.car_y = 50
        self.car_angle = self.base_car_angle
        self.car_speed = self.min_speed
        self.last_target_speed = self.min_speed
        self.last_speed_delta = 0.0
        self.speed_interval_steps_remaining = 0
        self.steering_offset = 0.0
        self.steering_target_offset = 0.0

        self.steps = 0
        self.timeframe = 0
        self.world_distance = 0.0
        self.world_step = 0
        self.done = False
        self.reached_finish = False
        self.last_action = 1
        self.in_center_reward_zone = True
        self.center_distance = 0.0
        self.center_half_width = max(
            min(float(self.lane_width), float(LANE_CENTER_REWARD_WIDTH)) / 2.0, 1e-6
        )
        self.warning_front = False
        self.warning_side_right = False
        self.warning_side_left = False
        self.warning_close_count_step = 0
        self.near_miss_active = False
        self.near_miss_count_step = 0


        self.obstacles = []
        for initial_obs in self.initial_obstacles:
            obstacle_state = self._build_obstacle_state(initial_obs)
            if obstacle_state is not None:
                self.obstacles.append(obstacle_state)
        self._refresh_finish_line()

        return self._get_state()

    def _get_sensor_readings(self):
        """
        Get readings from 7 sensors:
        - 5 front sensors at angles: -60, -30, 0, 30, 60 degrees from car direction
        - 2 side sensors at -90 and 90 degrees
        Note: In pygame coordinates, negative angles point RIGHT, positive angles point LEFT
        """

        readings = []
        for sensor_index in range(len(self.sensor_ranges)):
            measured = self._measure_sensor(sensor_index, apply_noise=True)
            readings.append(measured["normalized"])

        return np.array(readings)

    def _cast_ray(self, angle, max_range):
        """
        Cast a ray and return distance to nearest obstacle (road boundary or obstacle car)
        """
        rad = math.radians(angle)


        dx = math.cos(rad)
        dy = math.sin(rad)

        min_distance = max_range


        if dx < 0:
            t = -self.car_x / dx
            if 0 < t < min_distance:
                min_distance = t


        if dx > 0:
            t = (self.road_width - self.car_x) / dx
            if 0 < t < min_distance:
                min_distance = t


        for obstacle in self.obstacles:
            obstacle_left = obstacle["x"] - obstacle["width"] / 2
            obstacle_right = obstacle["x"] + obstacle["width"] / 2
            obstacle_bottom = obstacle["y"] - obstacle["height"] / 2
            obstacle_top = obstacle["y"] + obstacle["height"] / 2


            if abs(dx) > 0.001 or abs(dy) > 0.001:
                for t in range(1, int(min_distance) + 1):
                    ray_x = self.car_x + t * dx
                    ray_y = self.car_y + t * dy

                    if (
                        obstacle_left <= ray_x <= obstacle_right
                        and obstacle_bottom <= ray_y <= obstacle_top
                    ):
                        min_distance = t
                        break

        return min(min_distance, max_range)

    def _get_state(self):
        """Get current state observation"""
        sensors = self._get_sensor_readings()


        normalized_speed = (self.car_speed - self.min_speed) / (
            self.max_speed - self.min_speed
        )


        state = np.concatenate([sensors, [normalized_speed]])

        return state.astype(np.float32)

    def step(self, action, apply_steering=True):
        """
        Execute action and return new state, reward, done, info
        Actions (Update 1.9 - Extended with speed control):
          0=slow left, 1=slow straight, 2=slow right
          3=fast left, 4=fast straight, 5=fast right
        Speed transitions use decision-interval ramps:
          - fast action => +SPEED_UP km/h over one full decision interval
          - slow action => +SPEED_DOWN km/h over one full decision interval

        apply_steering: When True (decision step), update steering target cumulatively:
          - left  => target += TURNING_ANGLE
          - right => target -= TURNING_ANGLE
          - straight => keep target
          Then each frame (including non-decision) current steering moves toward
          target by animated rate so turning is smooth, not snappy.
        """
        self.steps += 1
        self.timeframe = self.steps
        self.last_action = action


        if apply_steering:
            if action in [0, 3]:
                self.steering_target_offset += self.steering_step
            elif action in [2, 5]:
                self.steering_target_offset -= self.steering_step


        steering_error = self.steering_target_offset - self.steering_offset
        if abs(steering_error) <= self.steering_anim_rate or self.steering_anim_rate <= 0:
            self.steering_offset = self.steering_target_offset
        else:
            if steering_error > 0:
                self.steering_offset += self.steering_anim_rate
            else:
                self.steering_offset -= self.steering_anim_rate


        self.car_angle = self.base_car_angle + self.steering_offset


        if apply_steering:
            decision_delta_kmh = (
                self.speed_down_kmh if action in [0, 1, 2] else self.speed_up_kmh
            )
            self._start_speed_interval(decision_delta_kmh)

        self._apply_interval_speed_dynamics()


        rad = math.radians(self.car_angle)
        dx = self.car_speed * math.cos(rad)
        dy = self.car_speed * math.sin(rad)
        self.car_x += dx
        self.car_y += dy
        self.world_distance += math.hypot(dx, dy)
        self.world_step = int(round(self.world_distance))


        moved_obstacle = False
        for obs in self.obstacles:
            obstacle_speed = self._resolve_obstacle_speed(
                obs.get("speed", getattr(self, "obstacle_speed", 0.0))
            )
            if obstacle_speed > 0:
                obs["speed"] = obstacle_speed
                obs["y"] = obs["y"] + obstacle_speed
                moved_obstacle = True

        if moved_obstacle and self.obstacles and not self.disable_finish:
            max_obstacle_y = max(obs["y"] for obs in self.obstacles)
            self.finish_line_y = max_obstacle_y + FINISH_DISTANCE
            self.road_length = self.finish_line_y


        for obs in self.obstacles:
            self._update_obstacle_lane_switch(obs)

        self._update_near_miss_state()

        reward = self._calculate_reward()


        self.done = self._check_done()

        ttc_details = calculate_env_ttc_details(self)
        info = {
            "car_x": self.car_x,
            "car_y": self.car_y,
            "car_angle": self.car_angle,
            "speed": self.car_speed,
            "target_speed": self.last_target_speed,
            "speed_delta": self.last_speed_delta,
            "timeframe": int(self.timeframe),
            "world_distance": float(self.world_distance),
            "world_step": int(self.world_step),
            "reached_finish": self.reached_finish,
            "warning_front": bool(self.warning_front),
            "warning_side_right": bool(self.warning_side_right),
            "warning_side_left": bool(self.warning_side_left),
            "warning_close_count": int(self.warning_close_count_step),
            "near_miss": int(self.near_miss_count_step),
            "near_miss_active": bool(self.near_miss_active),
            "nearmiss_distance": float(self.nearmiss_distance),
            "ttc_ms": ttc_details["ttc_ms"],
            "ttc_counting": bool(ttc_details["counting"]),
            "ttc_target_index": ttc_details["target_index"],
        }

        return self._get_state(), reward, self.done, info

    def _calculate_reward(self):
        """Calculate reward based on current state"""
        reward = 0.0
        self.warning_front = False
        self.warning_side_right = False
        self.warning_side_left = False
        self.warning_close_count_step = 0


        if self._check_collision_with_obstacles():
            reward += PENALTY_COLLISION
            return reward


        reward += REWARD_PROGRESS


        (
            _,
            distance_from_center,
            lane_center_reward_half_width,
            in_center_reward_zone,
        ) = self._lane_center_metrics()
        if in_center_reward_zone:
            reward += REWARD_LANE_CENTER_MAX
        else:
            reward += PENALTY_NOT_IN_CENTER

        self.in_center_reward_zone = in_center_reward_zone
        self.center_distance = distance_from_center
        self.center_half_width = lane_center_reward_half_width


        angle_diff = abs(self.car_angle - 90)
        if angle_diff < STRAIGHT_ANGLE_THRESHOLD:
            reward += REWARD_STRAIGHT_ANGLE


        front_sensor_distance = self._measure_sensor(2, apply_noise=True)["distance"]
        if front_sensor_distance < OBSTACLE_WARNING_DISTANCE_FRONT:
            reward += PENALTY_WARNING_DISTANCE_FRONT
            self.warning_front = True
            self.warning_close_count_step += 1

        side_right_distance = self._measure_sensor(5, apply_noise=True)["distance"]
        if side_right_distance < OBSTACLE_WARNING_DISTANCE_SIDES:
            reward += PENALTY_WARNING_DISTANCE_SIDES
            self.warning_side_right = True
            self.warning_close_count_step += 1

        side_left_distance = self._measure_sensor(6, apply_noise=True)["distance"]
        if side_left_distance < OBSTACLE_WARNING_DISTANCE_SIDES:
            reward += PENALTY_WARNING_DISTANCE_SIDES
            self.warning_side_left = True
            self.warning_close_count_step += 1


        front_is_clear = front_sensor_distance >= OBSTACLE_WARNING_DISTANCE_FRONT


        if front_is_clear:
            if self.last_action == 4:
                reward += REWARD_FAST_CLEAR
            elif self.last_action in [0, 1, 2]:

                reward += PENALTY_SLOW_WHEN_CLEAR

                self.slow_when_clear_count = (
                    getattr(self, "slow_when_clear_count", 0) + 1
                )


        car_left = self.car_x - self.car_width / 2
        car_right = self.car_x + self.car_width / 2
        if car_left < 0 or car_right > self.road_width:
            reward += PENALTY_COLLISION
            return reward


        if (
            not self.disable_finish
            and self.car_y >= self.finish_line_y
            and not self.reached_finish
        ):
            self.reached_finish = True
            reward += REWARD_FINISH


        if self.steps >= self.max_steps:
            reward += PENALTY_TIMEOUT

        return reward

    def get_car_box_corners(self, expand=0.0):
        """Return car outline corners in world units, optionally expanded from edges."""
        half_width = (float(self.car_width) / 2.0) + float(expand)
        half_height = (float(self.car_height) / 2.0) + float(expand)
        rad = math.radians(float(self.car_angle))
        forward_x = math.cos(rad)
        forward_y = math.sin(rad)
        right_x = math.sin(rad)
        right_y = -math.cos(rad)

        corners = []
        for width_sign, height_sign in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            x = (
                float(self.car_x)
                + (right_x * half_width * width_sign)
                + (forward_x * half_height * height_sign)
            )
            y = (
                float(self.car_y)
                + (right_y * half_width * width_sign)
                + (forward_y * half_height * height_sign)
            )
            corners.append((x, y))
        return corners

    @staticmethod
    def _axis_aligned_obstacle_corners(obstacle):
        half_width = float(obstacle["width"]) / 2.0
        half_height = float(obstacle["height"]) / 2.0
        x = float(obstacle["x"])
        y = float(obstacle["y"])
        return [
            (x - half_width, y - half_height),
            (x + half_width, y - half_height),
            (x + half_width, y + half_height),
            (x - half_width, y + half_height),
        ]

    @staticmethod
    def _project_polygon(corners, axis):
        axis_x, axis_y = axis
        projections = [(x * axis_x) + (y * axis_y) for x, y in corners]
        return min(projections), max(projections)

    @classmethod
    def _polygons_intersect(cls, corners_a, corners_b):
        axes = []
        for corners in (corners_a, corners_b):
            for idx in range(len(corners)):
                x1, y1 = corners[idx]
                x2, y2 = corners[(idx + 1) % len(corners)]
                edge_x = x2 - x1
                edge_y = y2 - y1
                axis_x = -edge_y
                axis_y = edge_x
                axis_len = math.hypot(axis_x, axis_y)
                if axis_len <= 1e-9:
                    continue
                axes.append((axis_x / axis_len, axis_y / axis_len))

        for axis in axes:
            min_a, max_a = cls._project_polygon(corners_a, axis)
            min_b, max_b = cls._project_polygon(corners_b, axis)
            if max_a < min_b or max_b < min_a:
                return False
        return True

    def _has_near_miss_obstacle(self):
        if not self.obstacles or self.nearmiss_distance <= 0:
            return False

        car_body_corners = self.get_car_box_corners(expand=0.0)
        near_miss_corners = self.get_car_box_corners(expand=self.nearmiss_distance)

        for obstacle in self.obstacles:
            obstacle_corners = self._axis_aligned_obstacle_corners(obstacle)
            if not self._polygons_intersect(near_miss_corners, obstacle_corners):
                continue
            if self._polygons_intersect(car_body_corners, obstacle_corners):
                continue
            return True
        return False

    def _update_near_miss_state(self):
        self.near_miss_active = bool(self._has_near_miss_obstacle())
        self.near_miss_count_step = 1 if self.near_miss_active else 0

    def _check_collision_with_obstacles(self):
        """Check if car collides with any obstacle"""

        car_left = self.car_x - self.car_width / 2
        car_right = self.car_x + self.car_width / 2
        car_bottom = self.car_y - self.car_height / 2
        car_top = self.car_y + self.car_height / 2

        for obstacle in self.obstacles:
            obstacle_left = obstacle["x"] - obstacle["width"] / 2
            obstacle_right = obstacle["x"] + obstacle["width"] / 2
            obstacle_bottom = obstacle["y"] - obstacle["height"] / 2
            obstacle_top = obstacle["y"] + obstacle["height"] / 2


            if (
                car_right > obstacle_left
                and car_left < obstacle_right
                and car_top > obstacle_bottom
                and car_bottom < obstacle_top
            ):
                return True
        return False

    def _check_done(self):
        """Check if episode should terminate"""

        if self._check_collision_with_obstacles():
            return True


        car_left = self.car_x - self.car_width / 2
        car_right = self.car_x + self.car_width / 2
        if car_left < 0 or car_right > self.road_width:
            return True


        if not self.disable_finish and self.car_y >= self.finish_line_y:
            return True


        if self.steps >= self.max_steps:
            return True


        if self.car_y < 0:
            return True

        return False

    def get_lane_positions(self):
        """Get x positions of lane dividers for rendering"""
        positions = []
        for i in range(1, self.lane_count):
            positions.append(i * self.lane_width)
        return positions

    def render_info(self):
        """Return info needed for rendering"""

        sensor_norms = self._get_sensor_readings().tolist()

        if self.max_speed - self.min_speed != 0:
            normalized_speed = (self.car_speed - self.min_speed) / (
                self.max_speed - self.min_speed
            )
        else:
            normalized_speed = 0.0
        normalized_speed = max(0.0, min(1.0, normalized_speed))

        nn_input = sensor_norms + [normalized_speed]
        (
            lane_center,
            distance_from_center,
            lane_center_reward_half_width,
            in_center_reward_zone,
        ) = self._lane_center_metrics()

        ttc_details = calculate_env_ttc_details(self)
        self._update_near_miss_state()
        return {
            "car_x": self.car_x,
            "car_y": self.car_y,
            "car_angle": self.car_angle,
            "car_width": self.car_width,
            "car_height": self.car_height,
            "speed": self.car_speed,
            "target_speed": self.last_target_speed,
            "speed_delta": self.last_speed_delta,
            "timeframe": int(self.steps),
            "world_distance": float(self.world_distance),
            "world_step": int(self.world_step),
            "road_width": self.road_width,
            "road_length": self.road_length,
            "lane_positions": self.get_lane_positions(),
            "finish_line_y": self.finish_line_y,
            "sensor_range": self.sensor_range,
            "sensor_ranges": self.sensor_ranges,
            "sensors": self._get_sensor_angles_and_distances(),
            "nn_input": nn_input,
            "lane_center_x": lane_center,
            "distance_from_lane_center": distance_from_center,
            "lane_center_half_width": lane_center_reward_half_width,
            "in_lane_center_zone": in_center_reward_zone,
            "obstacles": self.obstacles,
            "near_miss": int(self.near_miss_count_step),
            "near_miss_active": bool(self.near_miss_active),
            "nearmiss_distance": float(self.nearmiss_distance),
            "near_miss_corners": self.get_car_box_corners(
                expand=self.nearmiss_distance
            ),
            "ttc_ms": ttc_details["ttc_ms"],
            "ttc_counting": bool(ttc_details["counting"]),
            "ttc_target_index": ttc_details["target_index"],
        }

    def _get_sensor_angles_and_distances(self):
        """Get sensor angles and their distances for rendering"""

        sensors = []
        for sensor_index in range(len(self.sensor_ranges)):
            sensors.append(self._measure_sensor(sensor_index, apply_noise=True))
        return sensors
