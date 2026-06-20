# =============================================================================
# ROAD AND WORLD GEOMETRY
# All fundamental dimension values in this file utilize world units.
# Quick conversion to pixels: pixel = world unit * DEFAULT_SCALE.
# =============================================================================
LANE_COUNT = 3
LANE_WIDTH = 33  # 33 world units ~= 49.5 px at 1.5 scale
ROAD_WIDTH = LANE_COUNT * LANE_WIDTH

FINISH_DISTANCE = (
    100  # The finish line is consistently placed 100 units beyond the furthest obstacle
)

# =============================================================================
# SCREEN AND CAMERA
# The vehicle remains at a fixed vertical screen position to simulate road scrolling.
# =============================================================================
SCREEN_HEIGHT = 800  # Main viewport height
CAR_STATIC_Y_POS = 150  # Fixed vertical position of the vehicle on screen

DEFAULT_SCALE = 1.5  # 1 world unit = 1.5 pixels
FPS = 60

# =============================================================================
# UI TYPOGRAPHY (Visualization)
# =============================================================================
FONT_TITLE = 24
FONT_SUBTITLE = 20
FONT_TEXT = 18

# =============================================================================
# VEHICLE DIMENSIONS
# 18 x 39 world units ~= 27 x 58.5 pixels at the default scale.
# =============================================================================
CAR_WIDTH = 18
CAR_HEIGHT = 39
nearmiss_distance = 10

OBSTACLE_WIDTH = 18
OBSTACLE_HEIGHT = 39

USE_PNG = True

# =============================================================================
# VEHICLE DYNAMICS
# All internal velocities are measured in world_units/step.
# Quick conversion: px/s = world_units/step * DEFAULT_SCALE * FPS.
# =============================================================================
# SPEEDS

# CAR_OBSTACLE_SPEED = 2.379259259259259
# Approx conversions (uses DEFAULT_SCALE=1.5 and FPS=60 -> px/s = world_units/step * 90)\

# CAR_OBSTACLE_SPEED = 1.3000000000000000  # ~= 117 px/s ~= 30 km/h
# CAR_OBSTACLE_SPEED = 1.5166666666666666  # ~= 136 px/s ~= 35 km/h
# CAR_OBSTACLE_SPEED = 1.7333333333333334  # ~= 156 px/s ~= 40 km/h
# CAR_OBSTACLE_SPEED = 1.9500000000000000  # ~= 175 px/s ~= 45 km/h (to train serious)

CAR_OBSTACLE_SPEED = 2.1555555555555554  # ~= 194 px/s ~= 50 km/h (default)
# CAR_OBSTACLE_SPEED = 2.379259259259259  # ~= 214.13 px/s ~= 55 km/h
# CAR_OBSTACLE_SPEED = 2.5955555555555554  # ~= 233.60 px/s ~= 60 km/h
# CAR_OBSTACLE_SPEED = 2.8129629629629627  # ~= 253.07 px/s ~= 65 km/h
# CAR_OBSTACLE_SPEED = 3.0281481481481482  # ~= 272.53 px/s ~= 70 km/h
# CAR_OBSTACLE_SPEED = 3.2444444444444445  # ~= 292.00 px/s ~= 75 km/h

OBSTACLE_SPEED = CAR_OBSTACLE_SPEED
# The current calibration maps 214 px/s ~= 55 km/h and 292 px/s ~= 75 km/h.
CAR_MAX_SPEED = 3.2444444444444445  # ~= 292 px/s ~= 75 km/h ~= 20.83 m/s
CAR_MIN_SPEED = 2.3777777777777778  # ~= 214 px/s ~= 55 km/h ~= 15.28 m/s

# Each action decision is sustained for a complete DECISION_INTERVAL.
# Actions 3, 4, and 5 increase velocity by SPEED_UP km/h per interval.
# Actions 0, 1, and 2 decrease velocity by SPEED_DOWN km/h per interval.
SPEED_UP = 2
SPEED_DOWN = -3

# SENSOR NOISES
# SENSOR_NOISE_RANGE = [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
SENSOR_NOISE_RANGE = [0]

# =============================================================================
# CURRICULUM AND TRAINING CONTROL
# =============================================================================
DECISION_INTERVAL = 10  # A single decision is maintained for 10 simulation steps
TURNING_ANGLE = 5  # Steering target adjustment per left/right decision

TRAIN_MULTIPLIER = 5
MEMORY_SIZE = 50000
# Disable automatic interval checkpointing by default. Set to 0 to disable.
# Previously this value saved a resumable *_check.pth checkpoint every N episodes.


# =============================================================================
# DQN TRAINING HYPERPARAMETERS
# =============================================================================
LEARNING_RATE = 0.001
GAMMA = 0.99
BATCH_SIZE = 512
TARGET_UPDATE_FREQ = 10
DQN_HIDDEN_SIZES = (128, 128, 64)
GRAD_CLIP_MAX_NORM = 1.0

KEYONE_MULTIPLIER = 50

CONSECUTIVE_SAVE_BEST = 3
CONSECUTIVE_STAGE_REQ = 1
ALLSTAGE_CONSECUTIVE_REQ = 1

INDEPENDENT_BASED = (
    False  # Independent evaluation condition to advance to the next stage
)
SUCCESS_BASED_REQ = 1
INDEPENDENT_COUNT_REQ = 1000000


# =============================================================================
# RANDOM VISUALIZATION OBSTACLES
# =============================================================================
startRandom = 400
gapRandom = 125
maxRandom = 50
START_RANDOM = startRandom
GAP_RANDOM = gapRandom
MAX_RANDOM = maxRandom

# =============================================================================
# EPSILON MANAGEMENT
# Non-final stages and the final stage may utilize distinct lower-bound epsilon limits.
# =============================================================================
TRAIN_MAX_EPSILON = 1.0  # Initial epsilon value for training
TRAIN_MIN_EPSILON = 0.05  # Lower bound epsilon for non-final stages
TRAIN_FINAL_MIN_EPSILON = 0.05  # Lower bound epsilon for the final stage when SSC is 0
TRAIN_FINAL_MIN_EPSILON_SSC = (
    0.05  # Lower bound epsilon for the final stage when SSC > 0
)


CONSECUTIVE_EPSILON_RECOVERY = (
    100000000  # Episodes at minimum epsilon before recovery is enforced
)
CONSECUTIVE_EPSILON_RECOVERY_SSC = (
    100000000  # Episodes at final-stage SSC minimum epsilon before recovery
)
AMOUNT_EPSILON_RECOVERY = 0  # Epsilon value when recovery triggers
ENABLE_EPSILON_RECOVERY = False  # False indicates standard DQN behavior

NEW_STAGE_EPSILON = (
    0  # 0 indicates maintaining the current epsilon upon stage transition
)
# =============================================================================
# SENSOR CONFIGURATION
# Sensor array sequence for environment and neural network integration:
# [R2, R1, F, L1, L2, SR, SL]
# =============================================================================
SENSOR_F = 100  # log -20
SENSOR_L1 = 100  # log -20
SENSOR_R1 = 100  # log -20
SENSOR_L2 = 80  # log -20
SENSOR_R2 = 80  # log -20
SENSOR_SL = 40  # log -10
SENSOR_SR = 40  # log -10

SENSOR_ANGLE_F = 0
SENSOR_ANGLE_L1 = 15
SENSOR_ANGLE_R1 = -15
SENSOR_ANGLE_L2 = 40
SENSOR_ANGLE_R2 = -40
SENSOR_ANGLE_SL = 110
SENSOR_ANGLE_SR = -110

SENSOR_ANGLES = [
    SENSOR_ANGLE_R2,
    SENSOR_ANGLE_R1,
    SENSOR_ANGLE_F,
    SENSOR_ANGLE_L1,
    SENSOR_ANGLE_L2,
    SENSOR_ANGLE_SR,
    SENSOR_ANGLE_SL,
]


# =============================================================================
# REWARD SYSTEM
# Rewards encourage maintaining the center lane, a straight heading, and fast-straight
# decisions when the path is clear. Major penalties stem from collisions and entering warning thresholds.
# =============================================================================
OBSTACLE_WARNING_DISTANCE_FRONT = 60
OBSTACLE_WARNING_DISTANCE_SIDES = 16.4  # Lateral warning sensor distance
# Manual TTC/yellow-indicator range adjustment in world units.
# 0 follows the configured F sensor range. Positive starts earlier, negative later.
TTC_OFFSET = 200  # TTC starts at SENSOR_F + 200 world units = 280 WORL UNIT = 30 METER.
TTC_SCALE = 1.5  # 1.5 is default, following applied scale on pygame for realistic value
STRAIGHT_ANGLE_THRESHOLD = 10
LANE_CENTER_REWARD_WIDTH = 8  # Width of the active reward zone near the lane center
SHOW_CENTERLANE_REWARD_INDICATOR = False
CENTERLANE_REWARD_INDICATOR_COLOR = (0, 0, 255, 50)

LEFT_LR_OFFSETX = -3
RIGHT_LR_OFFSETX = 4
CENTER_LR_OFFSETX = 0.5
LEFT_OBSTACLE_OFFSETX = 0.0
RIGHT_OBSTACLE_OFFSETX = 0
CENTER_OBSTACLE_OFFSETX = 0.5

REWARD_PROGRESS = 0
REWARD_LANE_CENTER_MAX = 0.020
REWARD_STRAIGHT_ANGLE = 0.020
REWARD_FAST_CLEAR = 0.020
REWARD_FINISH = 5.0


PENALTY_COLLISION = -5.0
PENALTY_TIMEOUT = 0.0
PENALTY_WARNING_DISTANCE_FRONT = -0.030
PENALTY_WARNING_DISTANCE_SIDES = -0.030
PENALTY_NOT_IN_CENTER = -0.020
PENALTY_SLOW_WHEN_CLEAR = -0.030

# =============================================================================
# EXPERIMENTAL AND DEBUG CONFIGURATIONS
# =============================================================================
ValidationTesterMode = False
END_EXACT = True  # True terminates execution precisely at the CLI --episodes target after saving
EPSILON_DECAY = 0.998  # Decay factor applied per episode
INDRUN_FINAL_STAGE = (
    True  # True retains final-stage independent and validation run behavior
)
SAVE_MODEL_CHECKPOINT = 50  # Periodically saves a resumable *_check.pth checkpoint every N episodes (0 disables this)
visualize_logs_sec = 2

# =============================================================================

# =============================================================================
# MAIN CURRICULUM STAGES
# OBSTACLES define the primary training progression stages.
# A higher Y-value positions the obstacle further ahead.
# Each obstacle can optionally be assigned a custom "speed" attribute. If omitted,
# the default OBSTACLE_SPEED applies.
# Quick distance guidelines between obstacles:
# - 150 = Loose
# - 130 = Dense
# - 50  = Extremely constrained
# =============================================================================
OBSTACLES = [
    # FULL TRAIN
    [
        {"lane": 1, "y": 200},
        {"lane": 0, "y": 350},
        {"lane": 2, "y": 350},  #
    ],
    [
        {"lane": 1, "y": 200},
        {"lane": 0, "y": 350},
        {"lane": 2, "y": 350},  #
        {"lane": 1, "y": 500},
        {"lane": 2, "y": 500},
        {"lane": 0, "y": 650},
        {"lane": 2, "y": 650},  #
    ],
    [
        {"lane": 1, "y": 200},
        {"lane": 0, "y": 350},
        {"lane": 2, "y": 350},  #
        {"lane": 1, "y": 500},
        {"lane": 2, "y": 500},
        {"lane": 0, "y": 650},
        {"lane": 2, "y": 650},  #
        {"lane": 1, "y": 800},
        {"lane": 0, "y": 800},
        {"lane": 1, "y": 950},
        {"lane": 2, "y": 950},  #
    ],
    [
        {"lane": 1, "y": 200},
        {"lane": 0, "y": 350},
        {"lane": 2, "y": 350},  #
        {"lane": 1, "y": 500},
        {"lane": 2, "y": 500},
        {"lane": 0, "y": 650},
        {"lane": 2, "y": 650},  #
        {"lane": 1, "y": 800},
        {"lane": 0, "y": 800},
        {"lane": 1, "y": 950},
        {"lane": 2, "y": 950},  #
        {"lane": 1, "y": 1100},
        {"lane": 0, "y": 1100},
        {"lane": 1, "y": 1250},
        {"lane": 2, "y": 1250},  #
    ],
    # DENSE-20
    # [
    #     {"lane": 1, "y": 200},
    #     {"lane": 0, "y": 330},
    #     {"lane": 2, "y": 330},
    #     {"lane": 1, "y": 460},
    #     {"lane": 2, "y": 460},
    #     {"lane": 0, "y": 590},
    #     {"lane": 2, "y": 590},
    #     {"lane": 1, "y": 720},
    #     {"lane": 0, "y": 720},
    #     {"lane": 1, "y": 850},
    #     {"lane": 2, "y": 850},
    #     {"lane": 1, "y": 980},
    #     {"lane": 0, "y": 980},
    #     {"lane": 1, "y": 1110},
    #     {"lane": 2, "y": 1110},
    # ],
    # DENSE -10
    # [
    #     {"lane": 1, "y": 200},
    #     {"lane": 0, "y": 340},
    #     {"lane": 2, "y": 340},
    #     {"lane": 1, "y": 480},
    #     {"lane": 2, "y": 480},
    #     {"lane": 0, "y": 620},
    #     {"lane": 2, "y": 620},
    #     {"lane": 1, "y": 760},
    #     {"lane": 0, "y": 760},
    #     {"lane": 1, "y": 900},
    #     {"lane": 2, "y": 900},
    #     {"lane": 1, "y": 1040},
    #     {"lane": 0, "y": 1040},
    #     {"lane": 1, "y": 1180},
    #     {"lane": 2, "y": 1180},
    # ],
    # CUSTOM TRAIN
    # [
    #     {"lane": 1, "y": 250},  #
    # ],
    # [
    #     {"lane": 1, "y": 250},  #
    #     {"lane": 0, "y": 380},
    #     {"lane": 2, "y": 380},  #
    # ],
    # [
    #     {"lane": 1, "y": 250},  #
    #     {"lane": 0, "y": 380},
    #     {"lane": 2, "y": 380},  #
    #     {"lane": 0, "y": 510},
    #     {"lane": 1, "y": 510},  #
    # ],
    # [
    #     {"lane": 1, "y": 250},  #
    #     {"lane": 0, "y": 380},
    #     {"lane": 2, "y": 380},  #
    #     {"lane": 0, "y": 510},
    #     {"lane": 1, "y": 510},  #
    #     {"lane": 2, "y": 640},
    #     {"lane": 1, "y": 640},  #
    # ],
    # BLIND TRAIN
    # [
    #     {"lane": 1, "y": 200},  #
    #     {"lane": 2, "y": 350},  #
    #     {"lane": 0, "y": 350},  #
    # ],
]


# =============================================================================
# TESTER STAGES
# TEST_OBSTACLES are utilized for model validation post-training.
# The trailing numerical comments serve as stage tester references.
# =============================================================================
TEST_OBSTACLES = [
    # Single obstacles and early obstacle pairs.
    [{"lane": 0, "y": 200}],  # 1
    [{"lane": 1, "y": 200}],  # 2
    [{"lane": 2, "y": 200}],  # 3
    [{"lane": 0, "y": 200}, {"lane": 2, "y": 200}],  # 4
    [{"lane": 1, "y": 200}, {"lane": 0, "y": 200}],  # 5
    [{"lane": 1, "y": 200}, {"lane": 2, "y": 200}],  # 6
    # Base obstacles initiating from the left lane.
    [{"lane": 0, "y": 200}, {"lane": 0, "y": 350}],  # 7
    [{"lane": 0, "y": 200}, {"lane": 1, "y": 350}],  # 8
    [{"lane": 0, "y": 200}, {"lane": 2, "y": 350}],  # 9
    [{"lane": 0, "y": 200}, {"lane": 0, "y": 350}, {"lane": 2, "y": 350}],  # 10
    [{"lane": 0, "y": 200}, {"lane": 1, "y": 350}, {"lane": 0, "y": 350}],  # 11
    [{"lane": 0, "y": 200}, {"lane": 1, "y": 350}, {"lane": 2, "y": 350}],  # 12
    # Base obstacles initiating from the center lane.
    [{"lane": 1, "y": 200}, {"lane": 0, "y": 350}],  # 13
    [{"lane": 1, "y": 200}, {"lane": 1, "y": 350}],  # 14
    [{"lane": 1, "y": 200}, {"lane": 2, "y": 350}],  # 15
    [{"lane": 1, "y": 200}, {"lane": 0, "y": 350}, {"lane": 2, "y": 350}],  # 16
    [{"lane": 1, "y": 200}, {"lane": 1, "y": 350}, {"lane": 0, "y": 350}],  # 17
    [{"lane": 1, "y": 200}, {"lane": 1, "y": 350}, {"lane": 2, "y": 350}],  # 18
    # Base obstacles initiating from the right lane.
    [{"lane": 2, "y": 200}, {"lane": 0, "y": 350}],  # 19
    [{"lane": 2, "y": 200}, {"lane": 1, "y": 350}],  # 20
    [{"lane": 2, "y": 200}, {"lane": 2, "y": 350}],  # 21
    [{"lane": 2, "y": 200}, {"lane": 0, "y": 350}, {"lane": 2, "y": 350}],  # 22
    [{"lane": 2, "y": 200}, {"lane": 1, "y": 350}, {"lane": 0, "y": 350}],  # 23
    [{"lane": 2, "y": 200}, {"lane": 1, "y": 350}, {"lane": 2, "y": 350}],  # 24
    # Base obstacles initiating with dual parallel placements.
    [{"lane": 0, "y": 200}, {"lane": 2, "y": 200}, {"lane": 0, "y": 350}],  # 25
    [{"lane": 0, "y": 200}, {"lane": 2, "y": 200}, {"lane": 1, "y": 350}],  # 26
    [{"lane": 0, "y": 200}, {"lane": 2, "y": 200}, {"lane": 2, "y": 350}],  # 27
    [
        {"lane": 0, "y": 200},
        {"lane": 2, "y": 200},
        {"lane": 0, "y": 350},
        {"lane": 2, "y": 350},
    ],  # 28
    [
        {"lane": 0, "y": 200},
        {"lane": 2, "y": 200},
        {"lane": 1, "y": 350},
        {"lane": 0, "y": 350},
    ],  # 29
    [
        {"lane": 0, "y": 200},
        {"lane": 2, "y": 200},
        {"lane": 1, "y": 350},
        {"lane": 2, "y": 350},
    ],  # 30
    # Base obstacles exploring center-left combinations.
    [{"lane": 1, "y": 200}, {"lane": 0, "y": 200}, {"lane": 0, "y": 350}],  # 31
    [{"lane": 1, "y": 200}, {"lane": 0, "y": 200}, {"lane": 1, "y": 350}],  # 32
    [{"lane": 1, "y": 200}, {"lane": 0, "y": 200}, {"lane": 2, "y": 350}],  # 33
    [
        {"lane": 1, "y": 200},
        {"lane": 0, "y": 200},
        {"lane": 0, "y": 350},
        {"lane": 2, "y": 350},
    ],  # 34
    [
        {"lane": 1, "y": 200},
        {"lane": 0, "y": 200},
        {"lane": 1, "y": 350},
        {"lane": 0, "y": 350},
    ],  # 35
    [
        {"lane": 1, "y": 200},
        {"lane": 0, "y": 200},
        {"lane": 1, "y": 350},
        {"lane": 2, "y": 350},
    ],  # 36
    # Base obstacles exploring center-right combinations.
    [{"lane": 1, "y": 200}, {"lane": 2, "y": 200}, {"lane": 0, "y": 350}],  # 37
    [{"lane": 1, "y": 200}, {"lane": 2, "y": 200}, {"lane": 1, "y": 350}],  # 38
    [{"lane": 1, "y": 200}, {"lane": 2, "y": 200}, {"lane": 2, "y": 350}],  # 39
    [
        {"lane": 1, "y": 200},
        {"lane": 2, "y": 200},
        {"lane": 0, "y": 350},
        {"lane": 2, "y": 350},
    ],  # 40
    [
        {"lane": 1, "y": 200},
        {"lane": 2, "y": 200},
        {"lane": 1, "y": 350},
        {"lane": 0, "y": 350},
    ],  # 41
    [
        {"lane": 1, "y": 200},
        {"lane": 2, "y": 200},
        {"lane": 1, "y": 350},
        {"lane": 2, "y": 350},
    ],  # 42
    # # Dense patterns evaluating constrained maneuvering space.
    # [
    #     {"lane": 1, "y": 250},
    #     {"lane": 2, "y": 250},
    #     {"lane": 1, "y": 300},
    #     {"lane": 2, "y": 300},
    #     {"lane": 1, "y": 350},
    #     {"lane": 2, "y": 350},  #
    #     {"lane": 1, "y": 500},
    #     {"lane": 0, "y": 500},
    #     {"lane": 1, "y": 550},
    #     {"lane": 0, "y": 550},
    #     {"lane": 1, "y": 600},
    #     {"lane": 0, "y": 600},  #
    #     {"lane": 2, "y": 750},
    #     {"lane": 0, "y": 750},
    #     {"lane": 2, "y": 800},
    #     {"lane": 0, "y": 800},
    #     {"lane": 2, "y": 850},
    #     {"lane": 0, "y": 850},  #
    # ],  # 43
    # # Zig-zag patterns forcing consecutive lane changes.
    # [
    #     {"lane": 1, "y": 200},
    #     {"lane": 0, "y": 200},
    #     {"lane": 1, "y": 350},
    #     {"lane": 2, "y": 350},
    #     {"lane": 1, "y": 500},
    #     {"lane": 0, "y": 500},
    #     {"lane": 1, "y": 650},
    #     {"lane": 2, "y": 650},
    # ],  # 44
]
