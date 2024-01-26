from SharsorIPCpp.PySharsor.extensions.ros_bridge.abstractions import RosPublisher
from SharsorIPCpp.PySharsor.extensions.ros_bridge.abstractions import RosSubscriber
from SharsorIPCpp.PySharsor.extensions.ros_bridge.abstractions import toRosDType

import rclpy
from rclpy.qos import QoSProfile
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.subscription import Subscription
from rclpy.qos import ReliabilityPolicy, DurabilityPolicy, HistoryPolicy, LivelinessPolicy


from std_msgs.msg import Bool, Int32, Float32, Float64
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Float64MultiArray

import numpy as np

class Ros2Publisher(RosPublisher):

    def __init__(self,
                node: rclpy.node.Node,
                n_rows: int, 
                n_cols: int,
                basename: str,
                namespace: str = "",
                queue_size: int = 1, # by default only read latest msg
                dtype = np.float32):
        
        self._node = node

        self._qos_settings = QoSProfile(
                    reliability=ReliabilityPolicy.RELIABLE, # BEST_EFFORT
                    durability=DurabilityPolicy.TRANSIENT_LOCAL, # VOLATILE
                    history=HistoryPolicy.KEEP_LAST, # KEEP_ALL
                    depth=queue_size,  # Number of samples to keep if KEEP_LAST is used
                    liveliness=LivelinessPolicy.AUTOMATIC,
                    # deadline=1000000000,  # [ns]
                    # partition='my_partition' # useful to isolate communications
                    )

        super().__init__(n_rows=n_rows, 
                n_cols=n_cols,
                basename=basename,
                namespace=namespace,
                queue_size=queue_size, # by default only read latest msg
                dtype=dtype)

    def _create_publisher(self,
                    name: str, 
                    dtype, 
                    queue_size: int,
                    is_array = False,
                    latch = True):

        publisher = self._node.create_publisher(msg_type=toRosDType(dtype, is_array),
                                    topic=name,
                                    qos_profile=self._qos_settings)
        
        return publisher

    def _close(self):

        # called in the close()

        if self.publisher is not None:

            self.publisher.destroy()

class Ros2Subscriber(RosSubscriber):

    def __init__(self,
                node: rclpy.node.Node,
                basename: str,
                namespace: str = "",
                queue_size: int = 1):

        self._node = node
        
        super().__init__(basename=basename,
                namespace=namespace,
                queue_size=queue_size)
        
    def _create_subscriber(self,
                    name: str, 
                    dtype, 
                    callback, 
                    callback_args = None,
                    queue_size: int = None,
                    is_array = False):

        qos_settings = QoSProfile(depth=queue_size)

        subscriber = self._node.create_subscription(msg_type = toRosDType(dtype, is_array),
                        topic=name,
                        callback=callback,
                        qos_profile=qos_settings,
                        raw=False
                        )

    def _close(self):

        # called in the close()
        for i in range(len(self._ros_subscribers)):

            if self._ros_subscribers[i] is not None:

                self._ros_subscribers[i].destroy()
