import unittest
from unittest.mock import patch

from func_scale import to_kmh, to_meter
from func_ttc import (
    calculate_ttc_details,
    calculate_ttc_ms,
    format_ttc_ms,
    summarize_ttc_samples,
)
from main_constant import CAR_HEIGHT, OBSTACLE_HEIGHT, TTC_SCALE
from main_visualize import build_visualize_episode_row


class TestTimeToCollision(unittest.TestCase):
    def _obstacle(self, lane_x=49.5, y=100.0, speed=1.0):
        return {
            "x": lane_x,
            "y": y,
            "width": 18.0,
            "height": float(OBSTACLE_HEIGHT),
            "lane": 1,
            "speed": speed,
        }

    def test_same_lane_front_obstacle_returns_ttc_ms(self):
        car_x = 49.5
        car_y = 50.0
        car_speed = 3.0
        obstacle_speed = 1.0
        obstacle = self._obstacle(y=100.0, speed=obstacle_speed)

        ttc_ms = calculate_ttc_ms(
            car_x=car_x,
            car_y=car_y,
            car_height=CAR_HEIGHT,
            car_speed=car_speed,
            obstacles=[obstacle],
            lane_width=33,
            lane_count=3,
            sensor_ranges=[80, 100, 100, 100, 80, 40, 40],
        )

        gap_world = (100.0 - (OBSTACLE_HEIGHT / 2.0)) - (
            car_y + (CAR_HEIGHT / 2.0)
        )
        expected_ms = (
            to_meter(gap_world) / ((to_kmh(car_speed) - to_kmh(obstacle_speed)) / 3.6)
        ) * 1000.0 * TTC_SCALE
        self.assertAlmostEqual(ttc_ms, expected_ms, places=9)

    def test_ttc_scale_multiplies_reported_value(self):
        with patch("func_ttc.TTC_SCALE", 2.0):
            ttc_ms = calculate_ttc_ms(
                car_x=49.5,
                car_y=50.0,
                car_height=CAR_HEIGHT,
                car_speed=3.0,
                obstacles=[self._obstacle(y=100.0, speed=1.0)],
                lane_width=33,
                lane_count=3,
                sensor_ranges=[80, 100, 100, 100, 80, 40, 40],
            )

        gap_world = (100.0 - (OBSTACLE_HEIGHT / 2.0)) - (50.0 + (CAR_HEIGHT / 2.0))
        unscaled_ms = (
            to_meter(gap_world) / ((to_kmh(3.0) - to_kmh(1.0)) / 3.6)
        ) * 1000.0
        self.assertAlmostEqual(ttc_ms, unscaled_ms * 2.0, places=9)

    def test_ttc_details_reports_counting_and_target_index(self):
        details = calculate_ttc_details(
            car_x=49.5,
            car_y=50.0,
            car_height=CAR_HEIGHT,
            car_speed=3.0,
            obstacles=[
                self._obstacle(lane_x=16.5, y=80.0, speed=1.0),
                self._obstacle(y=100.0, speed=1.0),
            ],
            lane_width=33,
            lane_count=3,
            sensor_ranges=[80, 100, 100, 100, 80, 40, 40],
        )
        self.assertTrue(details["counting"])
        self.assertIsNotNone(details["ttc_ms"])
        self.assertEqual(details["target_index"], 1)

    def test_side_lane_obstacle_is_ignored(self):
        details = calculate_ttc_details(
            car_x=49.5,
            car_y=50.0,
            car_height=CAR_HEIGHT,
            car_speed=3.0,
            obstacles=[self._obstacle(lane_x=16.5, y=100.0, speed=1.0)],
            lane_width=33,
            lane_count=3,
            sensor_ranges=[80, 100, 100, 100, 80, 40, 40],
        )
        self.assertFalse(details["counting"])
        self.assertIsNone(details["ttc_ms"])
        self.assertIsNone(details["target_index"])

    def test_non_closing_obstacle_is_ignored(self):
        ttc_ms = calculate_ttc_ms(
            car_x=49.5,
            car_y=50.0,
            car_height=CAR_HEIGHT,
            car_speed=1.0,
            obstacles=[self._obstacle(y=100.0, speed=3.0)],
            lane_width=33,
            lane_count=3,
            sensor_ranges=[80, 100, 100, 100, 80, 40, 40],
        )
        self.assertIsNone(ttc_ms)

    def test_same_lane_obstacle_behind_agent_is_ignored(self):
        ttc_ms = calculate_ttc_ms(
            car_x=49.5,
            car_y=100.0,
            car_height=CAR_HEIGHT,
            car_speed=3.0,
            obstacles=[self._obstacle(y=40.0, speed=1.0)],
            lane_width=33,
            lane_count=3,
            sensor_ranges=[80, 100, 100, 100, 80, 40, 40],
        )
        self.assertIsNone(ttc_ms)

    def test_outside_extended_ttc_range_is_ignored(self):
        ttc_ms = calculate_ttc_ms(
            car_x=49.5,
            car_y=50.0,
            car_height=CAR_HEIGHT,
            car_speed=3.0,
            obstacles=[self._obstacle(y=400.0, speed=1.0)],
            lane_width=33,
            lane_count=3,
            sensor_ranges=[80, 100, 100, 100, 80, 40, 40],
        )
        self.assertIsNone(ttc_ms)

    def test_zero_ttc_offset_uses_front_sensor_range(self):
        with patch("func_ttc.TTC_OFFSET", 0):
            details = calculate_ttc_details(
                car_x=49.5,
                car_y=50.0,
                car_height=CAR_HEIGHT,
                car_speed=3.0,
                # gap = 191 - 19.5 - (50 + 19.5) = 102.0, outside the 100-unit F sensor range.
                obstacles=[self._obstacle(y=191.0, speed=1.0)],
                lane_width=33,
                lane_count=3,
                sensor_ranges=[80, 100, 100, 100, 80, 40, 40],
            )
        self.assertFalse(details["counting"])
        self.assertIsNone(details["ttc_ms"])
        self.assertIsNone(details["target_index"])

    def test_positive_ttc_offset_starts_counting_earlier(self):
        with patch("func_ttc.TTC_OFFSET", 5):
            details = calculate_ttc_details(
                car_x=49.5,
                car_y=50.0,
                car_height=CAR_HEIGHT,
                car_speed=3.0,
                obstacles=[self._obstacle(y=191.0, speed=1.0)],
                lane_width=33,
                lane_count=3,
                sensor_ranges=[80, 100, 100, 100, 80, 40, 40],
            )

        self.assertTrue(details["counting"])
        self.assertIsNotNone(details["ttc_ms"])
        self.assertEqual(details["target_index"], 0)

    def test_negative_ttc_offset_starts_counting_later(self):
        with patch("func_ttc.TTC_OFFSET", -20):
            details = calculate_ttc_details(
                car_x=49.5,
                car_y=50.0,
                car_height=CAR_HEIGHT,
                car_speed=3.0,
                obstacles=[self._obstacle(y=171.0, speed=1.0)],
                lane_width=33,
                lane_count=3,
                sensor_ranges=[80, 100, 100, 100, 80, 40, 40],
            )

        self.assertFalse(details["counting"])
        self.assertIsNone(details["ttc_ms"])
        self.assertIsNone(details["target_index"])

    def test_positive_ttc_offset_extends_past_front_sensor_range(self):
        with patch("func_ttc.TTC_OFFSET", 100):
            details = calculate_ttc_details(
                car_x=49.5,
                car_y=50.0,
                car_height=CAR_HEIGHT,
                car_speed=3.0,
                obstacles=[self._obstacle(y=191.0, speed=1.0)],
                lane_width=33,
                lane_count=3,
                sensor_ranges=[80, 100, 100, 100, 80, 40, 40],
            )

        self.assertTrue(details["counting"])
        self.assertIsNotNone(details["ttc_ms"])
        self.assertEqual(details["target_index"], 0)

    def test_summarize_and_format_ttc_samples(self):
        summary = summarize_ttc_samples([1000.0, None, 500.0, 1500.0])
        self.assertEqual(summary["count"], 3)
        self.assertEqual(summary["min_ttc"], 500.0)
        self.assertEqual(summary["avg_ttc"], 1000.0)
        self.assertEqual(format_ttc_ms(summary["min_ttc"]), "500.000")
        self.assertEqual(format_ttc_ms(None), "None")

    def test_visualize_episode_row_includes_ttc_fields(self):
        row = build_visualize_episode_row(
            episode=1,
            close_distance=2,
            mse=0.0,
            reward=1.25,
            avg_reward=1.25,
            time_ms=123,
            timeframe=10,
            steps=20,
            min_ttc=400.0,
            avg_ttc=800.0,
        )
        self.assertEqual(row["min_ttc"], "400.000")
        self.assertEqual(row["avg_ttc"], "800.000")


if __name__ == "__main__":
    unittest.main()
