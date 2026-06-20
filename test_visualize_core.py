import os
import random
import tempfile
import unittest

from main_constant import (
    FINISH_DISTANCE,
    OBSTACLE_WARNING_DISTANCE_FRONT,
    OBSTACLE_WARNING_DISTANCE_SIDES,
)
from main_environment import CarEnvironment
from main_dqn_agent import DQNAgent
from main_visualize import (
    ExperimentObstaclePlanner,
    RandomObstacleGenerator,
    build_neuron_trace,
    build_visualize_episode_row,
    extract_model_number,
    get_evaluable_model_paths,
    get_evaluate_csv_path,
    get_next_visualize_csv_path,
    get_visualize_csv_headers,
)


class TestVisualizeCore(unittest.TestCase):
    def test_next_visualize_csv_path_increments_existing_index(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            open(os.path.join(tmp_dir, "visualize-1.csv"), "w", encoding="utf-8").close()
            open(os.path.join(tmp_dir, "visualize-3.csv"), "w", encoding="utf-8").close()
            open(os.path.join(tmp_dir, "notes.txt"), "w", encoding="utf-8").close()

            next_path = get_next_visualize_csv_path(tmp_dir)

            self.assertTrue(next_path.endswith("visualize-4.csv"))

    def test_evaluate_csv_path_uses_fixed_filename(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            evaluate_path = get_evaluate_csv_path(tmp_dir)

            self.assertEqual(os.path.basename(evaluate_path), "evaluate.csv")
            self.assertTrue(os.path.isdir(tmp_dir))

    def test_get_visualize_csv_headers_can_prefix_model(self):
        normal_headers = get_visualize_csv_headers()
        evaluate_headers = get_visualize_csv_headers(include_model=True)

        self.assertEqual(normal_headers[0], "episode")
        self.assertEqual(evaluate_headers[:2], ["model", "episode"])

    def test_extract_model_number_skips_checkpoints(self):
        self.assertEqual(extract_model_number("model_stage4-2995.pth"), 2995)
        self.assertIsNone(extract_model_number("model_stage4-3000_check.pth"))
        self.assertIsNone(extract_model_number("model_stage4.pth"))

    def test_get_evaluable_model_paths_sorts_descending_and_skips_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            for filename in [
                "model_stage4-2995.pth",
                "model_stage4-3000_check.pth",
                "model_stage4-2977.pth",
                "model_stage4.pth",
                "notes.txt",
            ]:
                open(os.path.join(tmp_dir, filename), "w", encoding="utf-8").close()

            entries = get_evaluable_model_paths(tmp_dir)

            self.assertEqual([number for number, _path in entries], [2995, 2977])

    def test_build_visualize_episode_row_format(self):
        row = build_visualize_episode_row(
            episode=7,
            close_distance=12,
            mse=0.0,
            reward=34.56789,
            avg_reward=20.1,
            time_ms=1234,
            timeframe=78,
            steps=56,
        )

        expected = {
            "episode": 7,
            "close distance": 12,
            "near_miss": 0,
            "MSE": "0.000",
            "Reward": "34.568",
            "Avg reward": "20.100",
            "time": 1234,
            "timeframe": 78,
            "steps": 56,
            "progress_pct": 0.0,
            "reached_finish": 0,
            "min_ttc": "None",
            "avg_ttc": "None",
        }
        for sensor_name in ["R2", "R1", "F", "L1", "L2", "SR", "SL"]:
            expected[f"min_{sensor_name}"] = "None"

        self.assertEqual(row, expected)

    def test_build_visualize_episode_row_can_prefix_model(self):
        row = build_visualize_episode_row(
            model=2995,
            episode=1,
            close_distance=0,
            mse=0.0,
            reward=1.0,
            avg_reward=1.0,
            time_ms=2,
            timeframe=3,
            steps=4,
        )

        self.assertEqual(list(row.keys())[:2], ["model", "episode"])
        self.assertEqual(row["model"], 2995)

    def test_experiment_planner_builds_cumulative_spawn_plan(self):
        planner = ExperimentObstaclePlanner()

        planner.toggle_lane("left")
        planner.distance = 150
        self.assertTrue(planner.add_current_selection())

        planner.toggle_lane("center")
        planner.distance = 125
        self.assertTrue(planner.add_current_selection())

        plan = planner.build_spawn_plan(current_car_y=1000)
        configs = planner.build_obstacle_configs(current_car_y=1000)

        self.assertEqual(len(plan), 2)
        self.assertAlmostEqual(plan[0]["spawn_y"], 1350.0)
        self.assertAlmostEqual(plan[1]["spawn_y"], 1475.0)
        self.assertEqual(len(configs), 3)
        self.assertEqual(configs[0]["lane"], 0)
        self.assertAlmostEqual(configs[0]["y"], 1350.0)
        self.assertEqual(configs[1]["lane"], 0)
        self.assertAlmostEqual(configs[1]["y"], 1475.0)
        self.assertEqual(configs[2]["lane"], 1)
        self.assertAlmostEqual(configs[2]["y"], 1475.0)

    def test_random_obstacle_generator_appends_rows_with_configured_gap(self):
        generator = RandomObstacleGenerator(
            start_y=400,
            gap_y=125,
            lookahead_y=500,
            rng=random.Random(7),
        )
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        env.reset()

        added_count = generator.append_due_obstacles(env)

        rows = sorted({obs["y"] for obs in env.obstacles})
        self.assertEqual(rows, [400.0, 525.0])
        self.assertEqual(generator.rows_spawned, 2)
        self.assertGreaterEqual(added_count, 2)
        self.assertLessEqual(added_count, 4)
        for y_value in rows:
            lanes_at_y = [obs["lane"] for obs in env.obstacles if obs["y"] == y_value]
            self.assertGreaterEqual(len(lanes_at_y), 1)
            self.assertLessEqual(len(lanes_at_y), 2)
            self.assertEqual(len(lanes_at_y), len(set(lanes_at_y)))

    def test_random_obstacle_generator_builds_fifty_row_stage(self):
        generator = RandomObstacleGenerator(
            start_y=400,
            gap_y=125,
            max_rows=50,
            rng=random.Random(11),
        )
        env = CarEnvironment(obstacles_config=[[]], disable_finish=False)
        env.reset()

        added_count = generator.append_all_obstacles(env)
        rows = sorted({obs["y"] for obs in env.obstacles})

        self.assertEqual(len(rows), 50)
        self.assertEqual(rows[0], 400.0)
        self.assertEqual(rows[-1], 400.0 + (125.0 * 49))
        self.assertGreaterEqual(added_count, 50)
        self.assertLessEqual(added_count, 100)
        self.assertEqual(env.finish_line_y, rows[-1] + FINISH_DISTANCE)

    def test_build_neuron_trace_reports_layer_shapes_and_q_values(self):
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        state = env.reset()
        agent = DQNAgent(env.state_size, env.action_size, memory_size=10)

        trace = build_neuron_trace(
            agent,
            state,
            max_neurons_per_hidden_layer=2,
            max_inputs_per_neuron=3,
        )

        self.assertEqual(len(trace["input"]), env.state_size)
        self.assertEqual(len(trace["layers"]), 4)
        self.assertEqual(len(trace["q_values"]), env.action_size)
        self.assertEqual(trace["layers"][0]["input_size"], env.state_size)
        self.assertEqual(trace["layers"][-1]["output_size"], env.action_size)
        self.assertGreater(trace["total_params"], 0)
        for layer in trace["layers"][:-1]:
            self.assertLessEqual(len(layer["neurons"]), 2)
            for neuron in layer["neurons"]:
                self.assertLessEqual(len(neuron["contributions"]), 3)

    def test_warning_close_count_reports_front_plus_sides(self):
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        env.reset()

        def fake_measure_sensor(sensor_index, apply_noise=True):
            distances = {
                2: OBSTACLE_WARNING_DISTANCE_FRONT - 1.0,
                5: OBSTACLE_WARNING_DISTANCE_SIDES - 1.0,
                6: OBSTACLE_WARNING_DISTANCE_SIDES + 1.0,
            }
            max_range = float(env.sensor_ranges[sensor_index])
            distance = float(distances.get(sensor_index, max_range))
            return {
                "index": int(sensor_index),
                "angle": 0.0,
                "base_distance": distance,
                "noise": 0.0,
                "distance": distance,
                "normalized": distance / max_range,
            }

        env._measure_sensor = fake_measure_sensor
        _, _, _, info = env.step(4, apply_steering=True)

        self.assertTrue(info["warning_front"])
        self.assertTrue(info["warning_side_right"])
        self.assertFalse(info["warning_side_left"])
        self.assertEqual(info["warning_close_count"], 2)


if __name__ == "__main__":
    unittest.main()
