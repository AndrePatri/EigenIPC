import argparse
import time

import numpy as np

from EigenIPC.PyEigenIPC import (
    ClientFactory,
    StringTensorClient,
    VLevel,
    dtype,
)
from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.from_ros import FromRos

from common_test_data import (
    NAMESPACE,
    REMAP_NS,
    NUMERIC_BASENAME,
    STRING_BASENAME,
    NUMERIC_DATA,
    STRING_DATA,
    CHECK_DT_S,
    WAIT_TIMEOUT_S,
)


def _parse_args():

    parser = argparse.ArgumentParser(description="ROS receiver + shared-memory validation client")
    parser.add_argument("--namespace", type=str, default=NAMESPACE)
    parser.add_argument("--remap-ns", type=str, default=REMAP_NS)
    parser.add_argument("--numeric-basename", type=str, default=NUMERIC_BASENAME)
    parser.add_argument("--string-basename", type=str, default=STRING_BASENAME)
    parser.add_argument("--ros-backend", type=str, choices=["ros1", "ros2"], default="ros2")
    parser.add_argument("--queue-size", type=int, default=1)
    parser.add_argument("--wait-timeout", type=float, default=WAIT_TIMEOUT_S)
    parser.add_argument("--check-dt", type=float, default=CHECK_DT_S)

    return parser.parse_args()


def _init_ros(backend: str):

    if backend == "ros1":
        import rospy

        rospy.init_node("test_from_ros_client", anonymous=True)
        return None

    import rclpy

    rclpy.init()
    return rclpy.create_node("test_from_ros_client")


def _spin_ros_once(backend: str, node):

    if backend == "ros2":
        import rclpy

        rclpy.spin_once(node, timeout_sec=0.0)


def _shutdown_ros(backend: str, node):

    if backend == "ros1":
        import rospy

        if not rospy.is_shutdown():
            rospy.signal_shutdown("test_from_ros_client exit")
        return

    import rclpy

    if node is not None:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def _wait_bridge_ready(bridge, backend: str, node, timeout_s: float):

    start = time.time()
    while (time.time() - start) < timeout_s:
        _spin_ros_once(backend, node)
        if bridge.run():
            return True
        time.sleep(0.05)

    return False


def _decode_string_tensor(np_data):

    decoded = []

    for col_idx in range(np_data.shape[1]):
        raw = bytearray()
        terminated = False

        for row_idx in range(np_data.shape[0]):
            value = int(np_data[row_idx, col_idx])
            for byte_idx in range(4):
                byte = (value >> (8 * byte_idx)) & 0xFF
                if byte == 0:
                    terminated = True
                    break
                raw.append(byte)
            if terminated:
                break

        decoded.append(raw.decode("utf-8", errors="ignore"))

    return decoded


def main():

    args = _parse_args()

    node = _init_ros(args.ros_backend)

    bridge_kwargs = dict(
        queue_size=args.queue_size,
        ros_backend=args.ros_backend,
        verbose=True,
        vlevel=VLevel.V2,
        force_reconnection=True,
    )
    if args.ros_backend == "ros2":
        bridge_kwargs["node"] = node

    numeric_bridge = FromRos(
        basename=args.numeric_basename,
        namespace=args.namespace,
        remap_ns=args.remap_ns,
        **bridge_kwargs,
    )
    string_bridge = FromRos(
        basename=args.string_basename,
        namespace=args.namespace,
        remap_ns=args.remap_ns,
        **bridge_kwargs,
    )

    if not _wait_bridge_ready(numeric_bridge, args.ros_backend, node, args.wait_timeout):
        raise RuntimeError("Timeout while waiting for numeric FromRos bridge metadata")

    if not _wait_bridge_ready(string_bridge, args.ros_backend, node, args.wait_timeout):
        raise RuntimeError("Timeout while waiting for string FromRos bridge metadata")

    numeric_client = ClientFactory(
        basename=args.numeric_basename,
        namespace=args.remap_ns,
        verbose=True,
        vlevel=VLevel.V2,
        dtype=dtype.Float,
        safe=True,
    )
    numeric_client.attach()

    string_client = StringTensorClient(
        basename=args.string_basename,
        name_space=args.remap_ns,
        verbose=True,
        vlevel=VLevel.V2,
        safe=True,
    )
    string_client.run()

    raw_string_client = ClientFactory(
        basename=args.string_basename,
        namespace=args.remap_ns,
        verbose=True,
        vlevel=VLevel.V2,
        dtype=dtype.Int,
        safe=True,
    )
    raw_string_client.attach()

    numeric_buffer = np.empty_like(NUMERIC_DATA)
    string_buffer = [""] * len(STRING_DATA)
    raw_string_buffer = np.zeros(
        (raw_string_client.getNRows(), raw_string_client.getNCols()),
        dtype=np.int32,
    )

    print(
        "Receiver running checks. "
        f"backend={args.ros_backend}, namespace={args.namespace}, remap_ns={args.remap_ns}."
    )

    start = time.time()
    numeric_ok = False
    string_ok = False

    try:
        while (time.time() - start) < args.wait_timeout:
            _spin_ros_once(args.ros_backend, node)

            numeric_bridge.update()
            string_bridge.update()

            numeric_read = numeric_client.read(numeric_buffer, 0, 0)
            string_read = string_client.read_vec(string_buffer, 0)
            raw_string_read = raw_string_client.read(raw_string_buffer, 0, 0)

            if numeric_read:
                numeric_ok = np.allclose(numeric_buffer, NUMERIC_DATA, atol=1e-6, rtol=0.0)

            if string_read:
                string_ok = string_buffer == STRING_DATA
            elif raw_string_read:
                decoded = _decode_string_tensor(raw_string_buffer)
                string_ok = decoded[:len(STRING_DATA)] == STRING_DATA

            if numeric_ok and string_ok:
                print("SUCCESS: numeric and string tensor data match expected values.")
                return

            time.sleep(args.check_dt)

        raise RuntimeError(
            "Validation timeout. "
            f"numeric_ok={numeric_ok}, string_ok={string_ok}, "
            f"numeric_last={numeric_buffer}, string_last={string_buffer}"
        )

    finally:
        try:
            numeric_bridge.close()
        except Exception:
            pass
        try:
            string_bridge.close()
        except Exception:
            pass
        try:
            numeric_client.close()
        except Exception:
            pass
        try:
            string_client.close()
        except Exception:
            pass
        try:
            raw_string_client.close()
        except Exception:
            pass

        _shutdown_ros(args.ros_backend, node)


if __name__ == "__main__":
    main()
