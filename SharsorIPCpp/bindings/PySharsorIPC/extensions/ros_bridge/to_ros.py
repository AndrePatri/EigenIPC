from SharsorIPCpp.PySharsorIPC import Client
from SharsorIPCpp.PySharsorIPC import StringTensorClient

from typing import Union

from SharsorIPCpp.PySharsor.extensions.ros_bridge.defs import NamingConventions

from SharsorIPCpp.PySharsorIPC import Journal, VLevel, LogType
from SharsorIPCpp.PySharsorIPC import toNumpyDType, dtype

class ToRos():

    # Atomic bridge element to forward data from SharsorIPCpp or PySharsorIPCpp
    # over topic. Can be useful when developing distributed architectures or when 
    # one needs to implement remote debugging features 

    # Given a client object (either a normal client or a string client), 
    # this object creates a publisher to a number of conventional topics to
    # stream both data and meta data associated with Sharsor.

    def __init__(self,
                client: Union[Client, 
                            StringTensorClient],
                queue_size: int = 1, # by default only read latest msg
                ros_backend = "ros1",
                node = None):

        self._check_client(client)

        self._client = client

        self._queue_size = queue_size

        self._publisher = None

        self._ros_backend = ros_backend
        
        self._check_backend()

        self._node = node # only used when ros2

        self._is_string_tensor = False
        if isinstance(client, StringTensorClient):
            self._is_string_tensor = True

        self._strigtensor_length = None
        self._stringtensor_data = None
        self._stringtensor_raw_buffer = None

    def _check_client(self,
                client: Union[Client, 
                            StringTensorClient]):

        if not isinstance(client, 
                    (Client, StringTensorClient)):

            exception = f"Provided client should be either an instance of ClientFactory" + \
                " or of StringTensorClient classes!"

            Journal.log(self.__class__.__name__,
                        "_check_client",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
    
    def _check_backend(self):

        if not (self._ros_backend == "ros1" or \
                self._ros_backend == "ros2"):
            
            exception = f"Unsupported ROS backend {self._ros_backend}. Supported are \"ros1\" and \"ros2\""

            Journal.log(self.__class__.__name__,
                        "_check_backend",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)
    
    def _synch_from_shared_mem(self,
                    retry: bool = True):

        if retry:
            
            if not self._is_string_tensor:

                while not self._client.read(self._publisher.np_data[:, :], 0, 0):
                    
                    continue

            else:

                while True:
                    
                    read = self._client.read_vec(self._stringtensor_data, 0)
                    
                    if not read:

                        continue
                    
                    else:
                        
                        self._publisher.np_data[:, :] = self._client.get_raw_buffer()

                        break

            return True
        
        else:
            
            if not self._is_string_tensor:

                return self._client.read(self._publisher.np_data[:, :], 0, 0)

            else:
                
                read = self._client.read_vec(self._stringtensor_data, 0)
                
                if not read:

                    return False
                
                else:

                    self._publisher.np_data[:, :] = self._client.get_raw_buffer()
            
    def _init_publisher(self):

        if self._ros_backend == "ros1":
            
            if self._node is not None:

                warn = f"A node argument was provided to constructor!" + \
                    f"but when using ros2 backend, that's not necessary!"

                Journal.log(self.__class__.__name__,
                            "_init_publisher",
                            warn,
                            LogType.WARN,
                            throw_when_excep = True)

            from SharsorIPCpp.PySharsor.extensions.ros_bridge.ros1_utils import Ros1Publisher

            if not self._is_string_tensor:

                self._publisher = Ros1Publisher(n_rows = self._client.getNRows(), 
                            n_cols = self._client.getNCols(),
                            basename = self._client.getBasename(),
                            namespace = self._client.getNamespace(),
                            queue_size = self._queue_size,
                            dtype = toNumpyDType(self._client.getScalarType()))
            
            else:
                
                # we publish the encoded string tensor

                self._strigtensor_length = self._client.length()
                self._stringtensor_data = [""] * self._strigtensor_length

                self._stringtensor_raw_buffer = self._client.get_raw_buffer()

                self._publisher = Ros1Publisher(n_rows = self._stringtensor_raw_buffer.shape[0], 
                            n_cols = self._stringtensor_raw_buffer.shape[1],
                            basename = self._client.getBasename(),
                            namespace = self._client.getNamespace(),
                            queue_size = self._queue_size,
                            dtype = toNumpyDType(dtype.Int))

        elif self._ros_backend == "ros2":
            
            if self._node is None:

                exception = f"No node argument provided to constructor! " + \
                    f"When using ros2 backend, you should provide it!"

                Journal.log(self.__class__.__name__,
                            "_init_publisher",
                            exception,
                            LogType.EXCEP,
                            throw_when_excep = True)
                            
            from SharsorIPCpp.PySharsor.extensions.ros_bridge.ros2_utils import Ros2Publisher

            if not self._is_string_tensor:

                self._publisher = Ros2Publisher(node=self._node,
                            n_rows = self._client.getNRows(), 
                            n_cols = self._client.getNCols(),
                            basename = self._client.getBasename(),
                            namespace = self._client.getNamespace(),
                            queue_size = self._queue_size,
                            dtype = toNumpyDType(self._client.getScalarType()))
            else:
                
                # we publish the encoded string tensor
                self._strigtensor_length = self._client.length()
                self._stringtensor_data = [""] * self._strigtensor_length

                string_tensor_raw_buffer = self._client.get_raw_buffer()

                string_tensor_raw_buffer.shape[0]

                self._publisher = Ros2Publisher(node=self._node,
                            n_rows = string_tensor_raw_buffer.shape[0], 
                            n_cols = string_tensor_raw_buffer.shape[1],
                            basename = self._client.getBasename(),
                            namespace = self._client.getNamespace(),
                            queue_size = self._queue_size,
                            dtype = toNumpyDType(dtype.Int))
                
        else:
            
            exception = f"backend {self._ros_backend} not supported. Please use either" + \
                    "\"ros1\" or \"ros2\"!"

            Journal.log(self.__class__.__name__,
                        "_init_publisher",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep = True)

        self._publisher.run() # initialized topics and writes initializations

    def run(self):

        if not self._client.isRunning():
                        
            # manually run client if not running

            self._client.run()
        
        self._init_publisher()
    
    def close(self):

        self._client.close()

    def update(self):
        
        success = self._synch_from_shared_mem() # updated publisher np view with shared memory

        if self._ros_backend == "ros2":
            
            self._publisher.pub_data(copy=True)

        if self._ros_backend == "ros1":

            self._publisher.pub_data()
       
        return success