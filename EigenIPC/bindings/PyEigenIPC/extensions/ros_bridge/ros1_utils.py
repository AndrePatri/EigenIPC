from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.abstractions import RosPublisher
from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.abstractions import RosSubscriber
from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.abstractions import toRosDType

import rospy

import numpy as np
        
class Ros1Publisher(RosPublisher):

    def __init__(self,
            n_rows: int, 
            n_cols: int,
            basename: str,
            namespace: str = "",
            queue_size: int = 1, # by default only read latest msg
            dtype = np.float32,
            latch: bool = True):

        self._latch = latch

        super().__init__(n_rows=n_rows, 
                n_cols=n_cols,
                basename=basename,
                namespace=namespace,
                queue_size=queue_size, # by default only read latest msg
                dtype=dtype)

        self._use_np_directly = True

    def _create_publisher(self,
                    name: str, 
                    dtype, 
                    queue_size: int,
                    is_array = False):

        # use numpy msg for efficiency
        publisher = rospy.Publisher(name =name, 
                        data_class=toRosDType(dtype, is_array), 
                        queue_size=queue_size,
                        latch=self._latch)
        
        return publisher

    def _close(self):
        
        # called in the close()

        for i in range(len(self._ros_publishers)):

                if self._ros_publishers[i] is not None:

                    self._ros_publishers[i].unregister()
            
class Ros1Subscriber(RosSubscriber):

    def __init__(self,
                basename: str,
                namespace: str = "",
                queue_size: int = 1,
                tcp_nodelay = False):

        self._tcp_nodelay = tcp_nodelay
        
        super().__init__(basename=basename,
                namespace=namespace,
                queue_size=queue_size)

    def _create_subscriber(self,
                    name: str, 
                    dtype, 
                    callback, 
                    queue_size: int = None,
                    is_array = False):

        # use numpy msg for efficiency
        subscriber = rospy.Subscriber(name =name, 
                        data_class=toRosDType(dtype, is_array), 
                        callback=callback,
                        queue_size=queue_size, 
                        tcp_nodelay=self._tcp_nodelay
                        )
        
        return subscriber

    def _close(self):
    
        # called in the close()
        for i in range(len(self._ros_subscribers)):

            if self._ros_subscribers[i] is not None:

                self._ros_subscribers[i].unregister()
                    