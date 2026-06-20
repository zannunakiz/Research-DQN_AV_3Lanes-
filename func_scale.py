import argparse

from main_constant import DEFAULT_SCALE, FPS

PX_PER_METER = 14.0


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"


def to_pixel(pygame_value: float) -> float:
    return float(pygame_value) * float(DEFAULT_SCALE)


def to_meter(pygame_value: float) -> float:
    pixel_value = to_pixel(pygame_value)
    return pixel_value / PX_PER_METER


def to_pygame(meter_value: float) -> float:
    pixel_value = float(meter_value) * PX_PER_METER
    return pixel_value / float(DEFAULT_SCALE)


def to_pxs(pygame_speed: float) -> float:
    return float(pygame_speed) * float(DEFAULT_SCALE) * float(FPS)


def to_kmh(pygame_speed: float) -> float:
    px_per_second = to_pxs(pygame_speed)
    meter_per_second = px_per_second / PX_PER_METER
    return meter_per_second * 3.6


def to_pygame_speed(kmh_value: float) -> float:
    meter_per_second = float(kmh_value) / 3.6
    px_per_second = meter_per_second * PX_PER_METER
    return px_per_second / (float(DEFAULT_SCALE) * float(FPS))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale helper CLI for main_constant.py values using 14 px = 1 meter."
    )
    parser.add_argument(
        "--tometer",
        type=float,
        help="Convert pygame value (world unit) to real meter",
    )
    parser.add_argument(
        "--topixel",
        type=float,
        help="Convert pygame value (world unit) to pixel",
    )
    parser.add_argument(
        "--topygame",
        type=float,
        help="Convert real meter to pygame value (world unit)",
    )
    parser.add_argument(
        "--topxs",
        type=float,
        help="Convert pygame speed (world_units/step) to pixel/s",
    )
    parser.add_argument(
        "--tokmh",
        type=float,
        help="Convert pygame speed (world_units/step) to km/h",
    )
    parser.add_argument(
        "--topygamespeed",
        type=float,
        help="Convert km/h to pygame speed (world_units/step)",
    )

    args = parser.parse_args()

    conversion_checks = [
        ("tometer", args.tometer, to_meter),
        ("topixel", args.topixel, to_pixel),
        ("topygame", args.topygame, to_pygame),
        ("topxs", args.topxs, to_pxs),
        ("tokmh", args.tokmh, to_kmh),
        ("topygamespeed", args.topygamespeed, to_pygame_speed),
    ]

    selected = [
        (name, value, fn) for name, value, fn in conversion_checks if value is not None
    ]

    if len(selected) != 1:
        parser.error("Provide exactly one conversion flag at a time.")

    name, value, fn = selected[0]
    result = fn(value)

    units = {
        "tometer": "meter",
        "topixel": "px",
        "topygame": "world_unit",
        "topxs": "px/s",
        "tokmh": "km/h",
        "topygamespeed": "world_unit/step",
    }

    print(f"{_fmt(result)} {units[name]}")


if __name__ == "__main__":
    main()
