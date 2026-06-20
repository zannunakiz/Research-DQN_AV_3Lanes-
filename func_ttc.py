from math import isfinite

from func_scale import to_kmh, to_meter
from main_constant import OBSTACLE_WARNING_DISTANCE_FRONT, TTC_OFFSET, TTC_SCALE


TTC_SENSOR_INDICES = (0, 1, 2, 3, 4)
TTC_FRONT_SENSOR_INDEX = 2


def _safe_float(value, default=0.0):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not isfinite(result):
        return float(default)
    return result


def _lane_from_x(x_value, lane_width, lane_count):
    lane_width = max(_safe_float(lane_width, 1.0), 1e-9)
    lane_count = max(int(lane_count), 1)
    lane = int(_safe_float(x_value, 0.0) // lane_width)
    return max(0, min(lane_count - 1, lane))


def format_ttc_ms(ttc_ms):
    if ttc_ms is None:
        return "None"
    return f"{float(ttc_ms):.3f}"


def summarize_ttc_samples(ttc_samples):
    values = [float(value) for value in ttc_samples if value is not None]
    if not values:
        return {"min_ttc": None, "avg_ttc": None, "count": 0}
    return {
        "min_ttc": min(values),
        "avg_ttc": sum(values) / len(values),
        "count": len(values),
    }


def _scale_ttc_ms(ttc_ms):
    if ttc_ms is None:
        return None
    return float(ttc_ms) * _safe_float(TTC_SCALE, 1.0)


def calculate_ttc_ms(
    car_x,
    car_y,
    car_height,
    car_speed,
    obstacles,
    lane_width,
    lane_count,
    sensor_ranges,
):
    details = calculate_ttc_details(
        car_x=car_x,
        car_y=car_y,
        car_height=car_height,
        car_speed=car_speed,
        obstacles=obstacles,
        lane_width=lane_width,
        lane_count=lane_count,
        sensor_ranges=sensor_ranges,
    )
    return details["ttc_ms"]


def calculate_ttc_details(
    car_x,
    car_y,
    car_height,
    car_speed,
    obstacles,
    lane_width,
    lane_count,
    sensor_ranges,
):
    """
    Return current TTC details for front-sensor vehicle targets.

    TTC is only valid when an obstacle vehicle is in the same lane, ahead of the
    agent, inside the configured F sensor range plus TTC_OFFSET, and the agent is
    closing the distance. Road boundaries and side sensors are excluded.
    """
    ranges = list(sensor_ranges or [])
    front_ranges = ranges[: len(TTC_SENSOR_INDICES)]
    if not front_ranges:
        return {"ttc_ms": None, "target_index": None, "counting": False}
    front_sensor_range = _safe_float(
        ranges[TTC_FRONT_SENSOR_INDEX] if len(ranges) > TTC_FRONT_SENSOR_INDEX else 0.0
    )
    configured_ttc_range = front_sensor_range + _safe_float(TTC_OFFSET)
    ttc_range = max(0.0, configured_ttc_range)
    if ttc_range <= 0:
        return {"ttc_ms": None, "target_index": None, "counting": False}

    agent_lane = _lane_from_x(car_x, lane_width, lane_count)
    car_y_value = _safe_float(car_y)
    car_half_height = _safe_float(car_height) / 2.0
    car_bottom = car_y_value - car_half_height
    car_top = car_y_value + car_half_height
    car_speed_kmh = to_kmh(_safe_float(car_speed))

    best_ttc_ms = None
    best_index = None
    for obstacle_index, obstacle in enumerate(obstacles or []):
        if not isinstance(obstacle, dict):
            continue

        obs_lane = _lane_from_x(obstacle.get("x", 0.0), lane_width, lane_count)
        if obs_lane != agent_lane:
            continue

        obs_height = _safe_float(obstacle.get("height", 0.0))
        obs_y = _safe_float(obstacle.get("y", 0.0))
        obs_bottom = obs_y - (obs_height / 2.0)
        obs_top = obs_y + (obs_height / 2.0)
        if obs_top < car_bottom:
            continue
        gap_world = obs_bottom - car_top
        if gap_world < 0:
            gap_world = 0.0
        if gap_world > ttc_range:
            continue

        obs_speed_kmh = to_kmh(_safe_float(obstacle.get("speed", 0.0)))
        relative_speed_kmh = car_speed_kmh - obs_speed_kmh
        if relative_speed_kmh <= 1e-9:
            continue

        distance_m = to_meter(gap_world)
        relative_speed_mps = relative_speed_kmh / 3.6
        ttc_ms = _scale_ttc_ms((distance_m / relative_speed_mps) * 1000.0)
        if best_ttc_ms is None or ttc_ms < best_ttc_ms:
            best_ttc_ms = ttc_ms
            best_index = obstacle_index

    return {
        "ttc_ms": best_ttc_ms,
        "target_index": best_index,
        "counting": best_ttc_ms is not None,
    }


def calculate_env_ttc_ms(env):
    return calculate_env_ttc_details(env)["ttc_ms"]


def calculate_env_ttc_details(env):
    return calculate_ttc_details(
        car_x=getattr(env, "car_x", 0.0),
        car_y=getattr(env, "car_y", 0.0),
        car_height=getattr(env, "car_height", 0.0),
        car_speed=getattr(env, "car_speed", 0.0),
        obstacles=getattr(env, "obstacles", []),
        lane_width=getattr(env, "lane_width", 1.0),
        lane_count=getattr(env, "lane_count", 1),
        sensor_ranges=getattr(env, "sensor_ranges", []),
    )
