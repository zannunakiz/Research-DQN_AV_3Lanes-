import unittest
from unittest.mock import patch

import numpy as np

from main_constant import (
    DECISION_INTERVAL,
    FINISH_DISTANCE,
    SENSOR_NOISE_RANGE,
    SPEED_DOWN,
    SPEED_UP,
)
from main_environment import CarEnvironment


class TestEnvironmentCore(unittest.TestCase):
    def _speed_to_kmh(self, env: CarEnvironment, speed_world: float) -> float:
        speed_span_world = float(env.max_speed) - float(env.min_speed)
        if abs(speed_span_world) < 1e-12:
            return 55.0
        normalized = (float(speed_world) - float(env.min_speed)) / speed_span_world
        return 55.0 + (normalized * 20.0)

    def _run_one_interval(self, env: CarEnvironment, action: int) -> None:
        for step_index in range(int(env.decision_interval)):
            env.step(action, apply_steering=(step_index == 0))

    def test_state_and_sensor_payload_shapes_are_consistent(self):
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        state = env.reset()

        self.assertEqual(state.shape, (env.state_size,))
        self.assertEqual(len(env.sensor_ranges), 7)

        readings = env._get_sensor_readings()
        sensors = env._get_sensor_angles_and_distances()

        self.assertEqual(readings.shape, (7,))
        self.assertEqual(len(sensors), 7)
        self.assertTrue(np.all(readings >= 0.0))
        self.assertTrue(np.all(readings <= 1.0))
        for sensor in sensors:
            self.assertIn("angle", sensor)
            self.assertIn("distance", sensor)
            self.assertIn("normalized", sensor)

    def test_speed_interval_profile_matches_constants(self):
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        env.reset()

        self.assertAlmostEqual(self._speed_to_kmh(env, env.car_speed), 55.0, places=9)

        self._run_one_interval(env, action=4)
        self.assertAlmostEqual(
            self._speed_to_kmh(env, env.car_speed),
            55.0 + float(SPEED_UP),
            places=9,
        )

        self._run_one_interval(env, action=4)
        self.assertAlmostEqual(
            self._speed_to_kmh(env, env.car_speed),
            55.0 + (2.0 * float(SPEED_UP)),
            places=9,
        )

        self._run_one_interval(env, action=1)
        self.assertAlmostEqual(
            self._speed_to_kmh(env, env.car_speed),
            55.0 + (2.0 * float(SPEED_UP)) + float(SPEED_DOWN),
            places=9,
        )

    def test_reaching_max_speed_requires_ten_fast_intervals(self):
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        env.reset()

        intervals = 0
        steps = 0
        while self._speed_to_kmh(env, env.car_speed) < 75.0 - 1e-12:
            intervals += 1
            self._run_one_interval(env, action=4)
            steps += int(env.decision_interval)
            self.assertLess(intervals, 1000)

        self.assertEqual(intervals, 10)
        self.assertEqual(steps, 10 * int(DECISION_INTERVAL))

    def test_finish_line_tracks_farthest_obstacle(self):
        env = CarEnvironment()
        env.reset()

        for _ in range(5):
            env.step(4, apply_steering=True)

        max_obstacle_y = max(float(obstacle["y"]) for obstacle in env.obstacles)
        expected_finish = max_obstacle_y + float(FINISH_DISTANCE)

        self.assertAlmostEqual(float(env.finish_line_y), expected_finish, places=6)
        self.assertAlmostEqual(
            float(env.render_info()["finish_line_y"]),
            float(env.finish_line_y),
            places=6,
        )

    def test_runtime_obstacle_management_updates_state(self):
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        env.reset()

        added = env.append_obstacles(
            [
                {"lane": 0, "y": 300},
                {"lane": 2, "y": 300},
                {"lane": 1, "y": 450},
            ]
        )

        self.assertEqual(added, 3)
        self.assertEqual(len(env.obstacles), 3)

        env.clear_obstacles()

        self.assertEqual(len(env.obstacles), 0)
        self.assertEqual(len(env.initial_obstacles), 0)

    def test_each_obstacle_speed_can_be_set_independently(self):
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        env.reset()

        env.append_obstacles(
            [
                {"lane": 0, "y": 300, "speed": 1.0},
                {"lane": 1, "y": 300, "speed": 2.5},
                {"lane": 2, "y": 300},
            ]
        )

        before = {int(obs["lane"]): float(obs["y"]) for obs in env.obstacles}
        env.step(4, apply_steering=True)
        after = {int(obs["lane"]): float(obs["y"]) for obs in env.obstacles}

        self.assertAlmostEqual(after[0] - before[0], 1.0, places=6)
        self.assertAlmostEqual(after[1] - before[1], 2.5, places=6)
        self.assertAlmostEqual(
            after[2] - before[2],
            float(env._resolve_obstacle_speed(None)),
            places=6,
        )

    def test_sensor_noise_options_include_zero_and_apply_to_measurement(self):
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        env.reset()

        expected_noise_values = set(float(value) for value in SENSOR_NOISE_RANGE)
        expected_noise_values.add(0.0)
        self.assertIn(0.0, env.sensor_noise_values)
        self.assertEqual(set(env.sensor_noise_values), expected_noise_values)
        self.assertEqual(len(env.sensor_noise_values), len(expected_noise_values))

        env._cast_ray = lambda _angle, _max_range: 50.0
        with patch("main_environment.random.choice", return_value=-0.1):
            sensors = env._get_sensor_angles_and_distances()

        self.assertEqual(len(sensors), 7)
        for sensor in sensors:
            self.assertAlmostEqual(sensor["base_distance"], 50.0, places=6)
            self.assertAlmostEqual(sensor["noise"], -0.1, places=6)
            expected_distance = min(49.9, float(env.sensor_ranges[sensor["index"]]))
            self.assertAlmostEqual(sensor["distance"], expected_distance, places=6)


if __name__ == "__main__":
    unittest.main()
