import numpy as np

NAMESPACE = "ZmqBridgeTest"
REMAP_NS = "ZmqBridgeTestRemap"

NUMERIC_BASENAME = "NumericData"
STRING_BASENAME = "StringData"

NUMERIC_DATA = np.array(
    [
        [1.0, -2.0, 3.5, 4.25],
        [5.5, 6.0, -7.125, 8.75],
        [9.0, 10.5, 11.25, -12.5],
    ],
    dtype=np.float32,
)

STRING_DATA = [
    "front_left",
    "front_right",
    "rear_left",
    "rear_right",
    "bridge_ok",
]

PUBLISH_DT_S = 0.02
CHECK_DT_S = 0.02
WAIT_TIMEOUT_S = 15.0
