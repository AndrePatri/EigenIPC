from EigenIPC.PyEigenIPC import ServerFactory
from EigenIPC.PyEigenIPC import StringTensorServer

from typing import Union

from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.defs import NamingConventions

from EigenIPC.PyEigenIPC import Journal, VLevel, LogType, dtype
from EigenIPC.PyEigenIPC import toNumpyDType

import numpy as np

class FromRos():

    # Atomic bridge element to forward data from conventional Ros Topics 
    # on shared memory thorugh PyEigenIPC. Can be useful when developing
    # distributed architectures or when one needs to implement remote debugging features 

    # Given a basename and namespace, this object creates a suitable EigenIPC server
    # which is created reading metadata on the conventional topics and updated with data 
    # streaming on topics in real time
    
    def __init__(self,
                basename: str,
                namespace: str = "",
                queue_size: int = 1, 
                ros_backend = "ros1",
                vlevel = VLevel.V3,
                verbose: bool = True,
                force_reconnection: bool = False,
                node = None):
        
        self._queue_size = queue_size

        self._basename = basename
        self._namespace = namespace

        self._vlevel = vlevel
        self._verbose = verbose
        self._force_reconnection = force_reconnection

        self._subscriber = None

        self._node = node # only used when ros2

        self._server = None

        self._is_running = False

        self._ros_backend = ros_backend

        self._check_backend()

        self._init_subscriber()

    def _check_backend(self):

        if not (self._ros_backend == "ros1" or \
                self._ros_backend == "ros2"):
            
            exception = f"Unsupported ROS backend {self._ros_backend}. Supported are \"ros1\" and \"ros2\""

            Journal.log(self.__class__.__name__,
                        "_check_backend",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
    
    def _init_subscriber(self):
        
        if self._ros_backend == "ros1":
            
            if self._node is not None:

                warn = f"A node argument was provided to constructor!" + \
                    f"but when using ros2 backend, that's not necessary!"

                Journal.log(self.__class__.__name__,
                            "_init_subscriber",
                            warn,
                            LogType.WARN,
                            throw_when_excep = True)

            from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.ros1_utils import Ros1Subscriber

            self._subscriber = Ros1Subscriber(basename = self._basename,
                                namespace = self._namespace,
                                queue_size = self._queue_size)

        elif self._ros_backend == "ros2":
            
            if self._node is None:

                exception = f"No node argument provided to constructor! " + \
                    f"When using ros2 backend, you should provide it!"

                Journal.log(self.__class__.__name__,
                            "_init_subscriber",
                            exception,
                            LogType.EXCEP,
                            throw_when_excep = True)
                            
            from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.ros2_utils import Ros2Subscriber

            self._subscriber = Ros2Subscriber(basename = self._basename,
                                namespace = self._namespace,
                                queue_size = self._queue_size,
                                node = self._node)

        else:
            
            exception = f"backend {self._ros_backend} not supported. Please use either" + \
                    "\"ros1\" or \"ros2\"!"

            Journal.log(self.__class__.__name__,
                        "_init_subscriber",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)

    def _write_to_shared(self,
                    retry: bool = True):
        
        if retry:
            
            while not self._server.write(self._subscriber.np_data[:, :], 0, 0):
                
                continue

            return True
        
        else:

            return self._server.write(self._subscriber.np_data[:, :], 0, 0)
        
    def _synch_from_topic(self):
        
        self._subscriber.synch()
    
    def _to_sharsor_dtype(self,
                    np_dtype):

        if np_dtype == np.bool_:

            return dtype.Bool 
        
        if np_dtype == np.int32:
            
            return dtype.Int 
        
        if np_dtype == np.float32:
            
            return dtype.Float 
        
        if np_dtype == np.float64:

            return dtype.Double 
    
    def run(self):

        if not self._is_running:

            sub_success = self._subscriber.run() # tried to get metadata and initialize
            # data subscription

            if sub_success:

                # also initialize and run shared mem server

                # creating a shared mem server
                self._server = ServerFactory(n_rows = self._subscriber.n_rows(), 
                            n_cols = self._subscriber.n_cols(),
                            basename = self._basename + "AAAAAA",
                            namespace = self._namespace, 
                            verbose = self._verbose, 
                            vlevel = self._vlevel, 
                            force_reconnection = self._force_reconnection, 
                            dtype = self._to_sharsor_dtype(self._subscriber.dtype()),
                            safe = True)
            
                self._server.run() # run server

                self._is_running = True # ready
        
        return sub_success and self._is_running

    def close(self):

        self._server.close()

    def update(self):
        
        if self._is_running:

            self._subscriber.acquire_data() # blocking

            # updates shared mem with latest read data on topic
                    
            return self._write_to_shared()

        else:

            return False