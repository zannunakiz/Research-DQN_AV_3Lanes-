import csv
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

from main_train import (
    _build_independent_model_path,
    _build_stage_model_path,
    _continue_csv_name,
    _checkpoint_training_state,
    _discounted_step_value,
    _ensure_tester_stage_csv,
    _fmt_reward4,
    _get_tester_stage_csv_path,
    _infer_episode_from_model_path,
    _infer_stage_from_model_path,
    _increment_tester_stage_fail_count,
    _resolve_resume_epsilon,
    _resolve_resume_stage,
    _resolve_resume_start_episode,
    _resolve_start_stage,
    _train_dqn_run,
    run_tester_validation,
    should_stop_on_episode_target_valid,
    train_dqn,
)


class _DummyEnv:
    def __init__(self, *args, **kwargs):
        pass


def _read_stage_counts(csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(row)
    return rows


class TestTrainingCore(unittest.TestCase):
    def test_episode_target_stop_rule(self):
        self.assertFalse(
            should_stop_on_episode_target_valid(
                episode=1999,
                num_episodes=2000,
                episode_had_final_valid_success=True,
                end_exact=False,
            )
        )
        self.assertFalse(
            should_stop_on_episode_target_valid(
                episode=2000,
                num_episodes=2000,
                episode_had_final_valid_success=False,
                end_exact=False,
            )
        )
        self.assertTrue(
            should_stop_on_episode_target_valid(
                episode=2000,
                num_episodes=2000,
                episode_had_final_valid_success=True,
                end_exact=False,
            )
        )
        self.assertTrue(
            should_stop_on_episode_target_valid(
                episode=2009,
                num_episodes=2000,
                episode_had_final_valid_success=True,
                end_exact=False,
            )
        )
        self.assertFalse(
            should_stop_on_episode_target_valid(
                episode=5000,
                num_episodes=sys.maxsize,
                episode_had_final_valid_success=True,
                end_exact=False,
            )
        )
        self.assertFalse(
            should_stop_on_episode_target_valid(
                episode=1999,
                num_episodes=2000,
                episode_had_final_valid_success=False,
                end_exact=True,
            )
        )
        self.assertTrue(
            should_stop_on_episode_target_valid(
                episode=2000,
                num_episodes=2000,
                episode_had_final_valid_success=False,
                end_exact=True,
            )
        )

    def test_discount_and_format_helpers(self):
        gamma = 0.99
        rewards = [-0.04, -0.03, -0.03, -5.0]
        expected = [-0.04, -0.0297, -0.029403, -4.851495]

        for power, (reward, expected_value) in enumerate(zip(rewards, expected)):
            self.assertAlmostEqual(
                _discounted_step_value(reward, gamma, power),
                expected_value,
                places=9,
            )

        self.assertEqual(_fmt_reward4(0), "0.0000")
        self.assertEqual(_fmt_reward4(1.2), "1.2000")
        self.assertEqual(_fmt_reward4(-0.0297), "-0.0297")

    def test_tester_stage_csv_create_and_increment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = _ensure_tester_stage_csv(tmp_dir, total_tester_stages=5)
            self.assertTrue(os.path.exists(csv_path))

            _increment_tester_stage_fail_count(
                tmp_dir,
                total_tester_stages=5,
                failed_stage=5,
            )
            _increment_tester_stage_fail_count(
                tmp_dir,
                total_tester_stages=5,
                failed_stage=5,
            )
            _increment_tester_stage_fail_count(
                tmp_dir,
                total_tester_stages=5,
                failed_stage=4,
            )

            rows = _read_stage_counts(csv_path)
            counts = {int(row["stage"]): int(row["fail_count"]) for row in rows}

            self.assertEqual(len(rows), 5)
            self.assertEqual(counts[5], 2)
            self.assertEqual(counts[4], 1)
            self.assertEqual(counts[1], 0)

    def test_continue_csv_and_stage_model_naming(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            self.assertEqual(
                os.path.basename(_build_stage_model_path(tmp_dir, 4)),
                "model_stage4.pth",
            )
            self.assertEqual(
                os.path.basename(_build_stage_model_path(tmp_dir, 4, 5688)),
                "model_stage4-5688.pth",
            )
            self.assertEqual(
                os.path.basename(
                    _build_stage_model_path(tmp_dir, 4, 5688, checkpoint=True)
                ),
                "model_stage4-5688_check.pth",
            )
            self.assertEqual(
                _continue_csv_name("model_stage4.csv", True),
                "con_model_stage4.csv",
            )
            self.assertEqual(
                os.path.basename(_get_tester_stage_csv_path(tmp_dir, True)),
                "con_tester_stage.csv",
            )
            self.assertEqual(
                _infer_stage_from_model_path("models/model_stage4-5688.pth"), 4
            )
            self.assertEqual(_resolve_start_stage(None, "models/model_stage4.pth", 5), 3)
            self.assertEqual(_resolve_start_stage(99, None, 5), 4)

    def test_continue_resume_defaults_from_checkpoint_state(self):
        checkpoint = {
            "epsilon": 0.5,
            "training_state": {
                "last_episode": 4030,
                "next_episode": 4031,
                "current_stage": 4,
                "epsilon_after_episode": 0.929,
            },
        }
        state = _checkpoint_training_state(checkpoint)

        self.assertEqual(
            _infer_episode_from_model_path("models/model_stage1-4030.pth"), 4030
        )
        self.assertEqual(
            _resolve_resume_start_episode(None, state, "models/model_stage1-4030.pth"),
            4031,
        )
        self.assertEqual(
            _resolve_resume_start_episode(40, state, "models/model_stage1-4030.pth"),
            40,
        )
        self.assertEqual(
            _resolve_resume_stage(None, state, "models/model_stage1-4030.pth", 5),
            3,
        )
        self.assertEqual(
            _resolve_resume_epsilon(None, state, checkpoint),
            0.929,
        )
        self.assertEqual(
            _resolve_resume_epsilon(0.25, state, checkpoint),
            0.25,
        )

    def test_independent_model_path_uses_independent_subdir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = _build_independent_model_path(
                save_dir=tmp_dir,
                stage_number=6,
                episode_number=5102,
            )

            self.assertEqual(
                os.path.relpath(path, tmp_dir),
                os.path.join("independent", "model_stage6-5102.pth"),
            )

    def test_train_dqn_forwards_continue_arguments(self):
        sentinel = (object(), [1.0], [0])
        with patch(
            "main_train._train_dqn_run", return_value=(*sentinel, None, True)
        ) as train_run:
            agent, rewards, success = train_dqn(
                num_episodes=10000,
                load_model_path="models/model_stage4.pth",
                continue_training=True,
                start_episode=40,
                start_epsilon=0.09,
                current_stage=4,
                algo="ddqn",
            )

        self.assertIs(agent, sentinel[0])
        self.assertEqual(rewards, sentinel[1])
        self.assertEqual(success, sentinel[2])
        kwargs = train_run.call_args.kwargs
        self.assertTrue(kwargs["continue_training"])
        self.assertEqual(kwargs["load_model_path"], "models/model_stage4.pth")
        self.assertEqual(kwargs["start_episode"], 40)
        self.assertEqual(kwargs["start_epsilon"], 0.09)
        self.assertEqual(kwargs["current_stage"], 4)
        self.assertEqual(kwargs["algo"], "ddqn")

    def test_end_exact_saves_target_checkpoint_and_csv(self):
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
                save_models=True,
                csv_log_path=csv_path,
                renderer=None,
                close_renderer=True,
                plot_results=False,
            )

            self.assertTrue(
                os.path.exists(os.path.join(tmp_dir, "model_stage1-1_check.pth"))
            )
            rows = _read_stage_counts(csv_path)
            self.assertEqual(len(rows), 1)
            self.assertEqual(int(rows[0]["episode"]), 1)

    def test_run_tester_validation_updates_failed_stage_counter(self):
        test_obstacles = [[], [], [], []]

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch("main_train.TEST_OBSTACLES", test_obstacles):
                with patch("main_train.CarEnvironment", _DummyEnv):
                    with patch(
                        "main_train.run_independent_test",
                        side_effect=[True, False],
                    ):
                        result = run_tester_validation(
                            agent=object(),
                            max_steps=10,
                            verbose=False,
                            save_dir=tmp_dir,
                        )
                        self.assertEqual(result, (False, 2, 4))

                    with patch(
                        "main_train.run_independent_test",
                        side_effect=[True, False],
                    ):
                        result = run_tester_validation(
                            agent=object(),
                            max_steps=10,
                            verbose=False,
                            save_dir=tmp_dir,
                        )
                        self.assertEqual(result, (False, 2, 4))

                    with patch(
                        "main_train.run_independent_test",
                        side_effect=[True, True, True, False],
                    ):
                        result = run_tester_validation(
                            agent=object(),
                            max_steps=10,
                            verbose=False,
                            save_dir=tmp_dir,
                        )
                        self.assertEqual(result, (False, 4, 4))

            rows = _read_stage_counts(os.path.join(tmp_dir, "tester_stage.csv"))
            counts = {int(row["stage"]): int(row["fail_count"]) for row in rows}

            self.assertEqual(counts[2], 2)
            self.assertEqual(counts[4], 1)
            self.assertEqual(counts[1], 0)
            self.assertEqual(counts[3], 0)


if __name__ == "__main__":
    unittest.main()
