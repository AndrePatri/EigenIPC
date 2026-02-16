import numpy as np

NAMESPACE = "RosBridgeTest"
REMAP_NS = "RosBridgeTestRemapped"

NUMERIC_BASENAME = "NumericState"
STRING_BASENAME = "StringState"

NUMERIC_DATA = np.array(
    [
        [1.5, -2.0, 3.25],
        [4.75, 5.0, -6.125],
    ],
    dtype=np.float32,
)

STRING_DATA = [
    "base_link",
    "front_left_foot",
    "front_right_foot",
    "rear_left_foot",
    "rear_right_foot",
]

PUBLISH_DT_S = 0.02
CHECK_DT_S = 0.02
WAIT_TIMEOUT_S = 10.0
