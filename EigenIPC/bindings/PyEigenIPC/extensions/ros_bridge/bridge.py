from EigenIPC.PySharsor.extensions.ros_bridge.to_ros import ToRos
from EigenIPC.PySharsor.extensions.ros_bridge.from_ros import FromRos

from typing import Union, List

class SharsorBridge():

    # high level class for instantiating a bridge between EigenIPC/PyEigenIPC and ROS.
    # It allows to create a bidirectional bridge, meaning it can concurrently forward local shared memory 
    # over topic and topics back to shared memory. 
    # Useful for implementing distributed systems in general and, in particular, remote debugging tools

    def __init__(self, 
                rate: float,
                traffic_list: List[Union[ToRos, FromRos]]):
        
        self._rate = rate # data from Sharsor to Ros will be approximately published at this rate

        self._traffic_list = traffic_list


