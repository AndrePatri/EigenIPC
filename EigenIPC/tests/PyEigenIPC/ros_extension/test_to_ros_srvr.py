import argparse
import time

from EigenIPC.PyEigenIPC import (
    ServerFactory,
    ClientFactory,
    StringTensorServer,
    StringTensorClient,
    VLevel,
    dtype,
)
from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.to_ros import ToRos

from common_test_data import (
    NAMESPACE,
    NUMERIC_BASENAME,
    STRING_BASENAME,
    NUMERIC_DATA,
    STRING_DATA,
    PUBLISH_DT_S,
)


def _parse_args():

    parser = argparse.ArgumentParser(description="ROS sender for numeric + string shared-memory data")
    parser.add_argument("--namespace", type=str, default=NAMESPACE)
    parser.add_argument("--numeric-basename", type=str, default=NUMERIC_BASENAME)
    parser.add_argument("--string-basename", type=str, default=STRING_BASENAME)
    parser.add_argument("--ros-backend", type=str, choices=["ros1", "ros2"], default="ros2")
    parser.add_argument("--queue-size", type=int, default=1)
    parser.add_argument("--publish-dt", type=float, default=PUBLISH_DT_S)

    return parser.parse_args()


def _init_ros(backend: str):

    if backend == "ros1":
        import rospy

        rospy.init_node("test_to_ros_srvr", anonymous=True)
        return None

    import rclpy

    rclpy.init()
    return rclpy.create_node("test_to_ros_srvr")


def _spin_ros_once(backend: str, node):

    if backend == "ros2":
        import rclpy

        rclpy.spin_once(node, timeout_sec=0.0)


def _shutdown_ros(backend: str, node):

    if backend == "ros1":
        import rospy

        if not rospy.is_shutdown():
            rospy.signal_shutdown("test_to_ros_srvr exit")
        return

    import rclpy

    if node is not None:
        node.destroy_node()
    if rclpy.ok():
        rclpy.shutdown()


def main():

    args = _parse_args()

    node = _init_ros(args.ros_backend)

    numeric_server = ServerFactory(
        n_rows=NUMERIC_DATA.shape[0],
        n_cols=NUMERIC_DATA.shape[1],
        basename=args.numeric_basename,
        namespace=args.namespace,
        verbose=True,
        vlevel=VLevel.V2,
        force_reconnection=True,
        dtype=dtype.Float,
        safe=True,
    )
    numeric_server.run()

    numeric_client = ClientFactory(
        basename=args.numeric_basename,
        namespace=args.namespace,
        verbose=True,
        vlevel=VLevel.V2,
        dtype=dtype.Float,
        safe=True,
    )
    numeric_client.attach()

    string_server = StringTensorServer(
        length=len(STRING_DATA),
        basename=args.string_basename,
        name_space=args.namespace,
        verbose=True,
        vlevel=VLevel.V2,
        force_reconnection=True,
        safe=True,
    )
    string_server.run()

    string_client = StringTensorClient(
        basename=args.string_basename,
        name_space=args.namespace,
        verbose=True,
        vlevel=VLevel.V2,
        safe=True,
    )
    string_client.run()

    bridge_kwargs = dict(
        queue_size=args.queue_size,
        ros_backend=args.ros_backend,
    )
    if args.ros_backend == "ros2":
        bridge_kwargs["node"] = node

    numeric_bridge = ToRos(client=numeric_client, **bridge_kwargs)
    string_bridge = ToRos(client=string_client, **bridge_kwargs)

    numeric_bridge.run()
    string_bridge.run()

    print(
        "ROS sender running. "
        f"backend={args.ros_backend}, namespace={args.namespace}, "
        f"numeric={args.numeric_basename}, string={args.string_basename}. "
        "Press Ctrl+C to stop."
    )

    try:
        while True:
            numeric_server.write(NUMERIC_DATA, 0, 0)
            string_server.write_vec(STRING_DATA, 0)

            numeric_bridge.update()
            string_bridge.update()

            _spin_ros_once(args.ros_backend, node)
            time.sleep(args.publish_dt)

    except KeyboardInterrupt:
        print("Stopped sender.")

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
            numeric_server.close()
        except Exception:
            pass
        try:
            string_server.close()
        except Exception:
            pass

        _shutdown_ros(args.ros_backend, node)


if __name__ == "__main__":
    main()
