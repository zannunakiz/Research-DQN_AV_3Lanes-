import csv
import os
import tempfile
import unittest

from main_constant import nearmiss_distance
from main_environment import CarEnvironment
from main_train import _train_dqn_run
from main_visualize import build_visualize_episode_row


class TestNearMissFunctionality(unittest.TestCase):
    def _set_obstacles(self, env, offsets):
        env.obstacles = [
            {
                "x": float(env.car_x) + float(dx),
                "y": float(env.car_y) + float(dy),
                "width": 18,
                "height": 39,
                "lane": 1,
                "speed": 0.0,
            }
            for dx, dy in offsets
        ]

    def test_near_miss_is_boolean_per_frame_for_multiple_obstacles(self):
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        env.reset()

        self._set_obstacles(
            env,
            [
                (env.car_width + (nearmiss_distance / 2.0), 0.0),
                (-(env.car_width + (nearmiss_distance / 2.0)), 0.0),
            ],
        )
        env._update_near_miss_state()

        self.assertTrue(env.near_miss_active)
        self.assertEqual(env.near_miss_count_step, 1)

    def test_near_miss_ignores_road_edges(self):
        env = CarEnvironment(obstacles_config=[[]], disable_finish=True)
        env.reset()
        env.car_x = 1.0
        env.obstacles = []

        env._update_near_miss_state()

        self.assertFalse(env.near_miss_active)
        self.assertEqual(env.near_miss_count_step, 0)

    def test_visualize_row_contains_new_near_miss_and_progress_fields(self):
        row = build_visualize_episode_row(
            episode=1,
            close_distance=2,
            near_miss=3,
            mse=0.0,
            reward=1.25,
            avg_reward=1.25,
            time_ms=10,
            timeframe=4,
            steps=5,
            progress_pct=65.0,
            reached_finish=1,
        )

        self.assertEqual(row["near_miss"], 3)
        self.assertEqual(row["progress_pct"], 65.0)
        self.assertEqual(row["reached_finish"], 1)

    def test_training_csv_contains_near_miss_and_close_distance_columns(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = os.path.join(tmp_dir, "TrainingLogs.csv")

            _train_dqn_run(
                num_episodes=1,
                max_steps_per_episode=1,
                save_interval=0,
                render_interval=999,
                verbose=False,
                visualize=False,
                seed=123,
                memory_size=128,
                save_dir=tmp_dir,
                save_models=False,
                csv_log_path=csv_path,
                renderer=None,
                close_renderer=True,
                plot_results=False,
            )

            with open(csv_path, "r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertIn("near_miss", rows[0])
        self.assertIn("close distance", rows[0])


if __name__ == "__main__":
    unittest.main()
