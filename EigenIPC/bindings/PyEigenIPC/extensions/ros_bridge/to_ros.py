from EigenIPC.PyEigenIPC import Client
from EigenIPC.PyEigenIPC import StringTensorClient

from typing import Union
import re

from EigenIPC.PyEigenIPC import Journal, LogType
from EigenIPC.PyEigenIPC import toNumpyDType, dtype


class ToRos:

    # Atomic bridge element to forward data from EigenIPC or PyEigenIPC over ROS topics.

    def __init__(self,
                client: Union[Client, StringTensorClient],
                queue_size: int = 1,
                ros_backend="ros1",
                node=None,
                source_row_index: int = None,
                source_n_rows: int = 1):

        self._check_client(client)

        self._client = client

        self._queue_size = queue_size

        self._publisher = None

        self._ros_backend = ros_backend

        self._check_backend()

        self._node = node  # only used when ros2

        self._source_row_index = source_row_index
        self._source_n_rows = source_n_rows

        self._is_string_tensor = self._is_string_tensor_client(client)

        self._stringtensor_length = None
        self._stringtensor_data = None

        self._read_row_index = 0
        self._read_n_rows = None

        self._check_slice_config()

    def _client_type_fqn(self, client) -> str:

        module = type(client).__module__
        name = type(client).__name__
        return f"{module}.{name}"

    def _is_string_tensor_client(self, client) -> bool:

        if isinstance(client, StringTensorClient):
            return True

        fqn = self._client_type_fqn(client)

        match = re.match(r"^(EigenIPC\.)?PyEigenIPC\.StringTensorClient$", fqn)

        return match is not None

    def _is_client(self, client) -> bool:

        if isinstance(client, Client):
            return True

        fqn = self._client_type_fqn(client)

        match = re.match(r"^(EigenIPC\.)?PyEigenIPC\.PyClient[A-Za-z0-9_]*$", fqn)

        return match is not None

    def _check_client(self,
                client: Union[Client, StringTensorClient]):

        is_numeric_client = self._is_client(client)
        is_string_client = self._is_string_tensor_client(client)

        if not (is_numeric_client or is_string_client):

            fqn = self._client_type_fqn(client)
            exception = "Provided client has unsupported type. Got type: " + \
                f"{fqn}. Expected types matching (EigenIPC.)?PyEigenIPC.PyClient* " + \
                "or (EigenIPC.)?PyEigenIPC.StringTensorClient."

            Journal.log(self.__class__.__name__,
                        "_check_client",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep=True)

    def _check_backend(self):

        if not (self._ros_backend == "ros1" or self._ros_backend == "ros2"):

            exception = f"Unsupported ROS backend {self._ros_backend}. Supported are \"ros1\" and \"ros2\""

            Journal.log(self.__class__.__name__,
                        "_check_backend",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep=True)

    def _check_slice_config(self):

        if self._source_row_index is not None and self._source_row_index < 0:
            Journal.log(self.__class__.__name__,
                "_check_slice_config",
                f"source_row_index {self._source_row_index} must be >= 0",
                LogType.EXCEP,
                throw_when_excep=True)

        if self._source_n_rows < 1:
            Journal.log(self.__class__.__name__,
                "_check_slice_config",
                f"source_n_rows {self._source_n_rows} must be >= 1",
                LogType.EXCEP,
                throw_when_excep=True)

    def _resolve_numeric_slice(self):

        n_rows_full = self._client.getNRows()

        if self._source_row_index is None:
            self._read_row_index = 0
            self._read_n_rows = n_rows_full
            return

        if self._source_row_index >= n_rows_full:
            Journal.log(self.__class__.__name__,
                "_resolve_numeric_slice",
                f"source_row_index {self._source_row_index} out of bounds for n_rows={n_rows_full}",
                LogType.EXCEP,
                throw_when_excep=True)

        end_row = self._source_row_index + self._source_n_rows
        if end_row > n_rows_full:
            Journal.log(self.__class__.__name__,
                "_resolve_numeric_slice",
                f"Requested rows [{self._source_row_index}, {end_row}) exceed n_rows={n_rows_full}",
                LogType.EXCEP,
                throw_when_excep=True)

        self._read_row_index = self._source_row_index
        self._read_n_rows = self._source_n_rows

    def _synch_from_shared_mem(self,
                    retry: bool = True):

        if retry:

            if not self._is_string_tensor:

                while not self._client.read(self._publisher.np_data[:, :], self._read_row_index, 0):

                    continue

            else:

                while True:

                    read = self._client.read_vec(self._stringtensor_data, 0)

                    if not read:

                        continue

                    self._publisher.np_data[:, :] = self._client.get_raw_buffer()

                    break

            return True

        if not self._is_string_tensor:

            return self._client.read(self._publisher.np_data[:, :], self._read_row_index, 0)

        read = self._client.read_vec(self._stringtensor_data, 0)

        if not read:

            return False

        self._publisher.np_data[:, :] = self._client.get_raw_buffer()

        return True

    def _init_publisher(self):

        if self._ros_backend == "ros1":

            if self._node is not None:

                warn = "A node argument was provided to constructor but when using ros1 backend, that's not necessary!"

                Journal.log(self.__class__.__name__,
                            "_init_publisher",
                            warn,
                            LogType.WARN,
                            throw_when_excep=True)

            from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.ros1_utils import Ros1Publisher

            if not self._is_string_tensor:

                self._resolve_numeric_slice()

                self._publisher = Ros1Publisher(n_rows=self._read_n_rows,
                            n_cols=self._client.getNCols(),
                            basename=self._client.getBasename(),
                            namespace=self._client.getNamespace(),
                            queue_size=self._queue_size,
                            dtype=toNumpyDType(self._client.getScalarType()))

            else:

                # we publish the encoded string tensor
                self._stringtensor_length = self._client.length()
                self._stringtensor_data = [""] * self._stringtensor_length

                stringtensor_raw_buffer = self._client.get_raw_buffer()

                self._publisher = Ros1Publisher(n_rows=stringtensor_raw_buffer.shape[0],
                            n_cols=stringtensor_raw_buffer.shape[1],
                            basename=self._client.getBasename(),
                            namespace=self._client.getNamespace(),
                            queue_size=self._queue_size,
                            dtype=toNumpyDType(dtype.Int))

        elif self._ros_backend == "ros2":

            if self._node is None:

                exception = "No node argument provided to constructor! When using ros2 backend, you should provide it!"

                Journal.log(self.__class__.__name__,
                            "_init_publisher",
                            exception,
                            LogType.EXCEP,
                            throw_when_excep=True)

            from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.ros2_utils import Ros2Publisher

            if not self._is_string_tensor:

                self._resolve_numeric_slice()

                self._publisher = Ros2Publisher(node=self._node,
                            n_rows=self._read_n_rows,
                            n_cols=self._client.getNCols(),
                            basename=self._client.getBasename(),
                            namespace=self._client.getNamespace(),
                            queue_size=self._queue_size,
                            dtype=toNumpyDType(self._client.getScalarType()))
            else:

                # we publish the encoded string tensor
                self._stringtensor_length = self._client.length()
                self._stringtensor_data = [""] * self._stringtensor_length

                stringtensor_raw_buffer = self._client.get_raw_buffer()

                self._publisher = Ros2Publisher(node=self._node,
                            n_rows=stringtensor_raw_buffer.shape[0],
                            n_cols=stringtensor_raw_buffer.shape[1],
                            basename=self._client.getBasename(),
                            namespace=self._client.getNamespace(),
                            queue_size=self._queue_size,
                            dtype=toNumpyDType(dtype.Int))

        else:

            exception = f"backend {self._ros_backend} not supported. Please use either \"ros1\" or \"ros2\"!"

            Journal.log(self.__class__.__name__,
                        "_init_publisher",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep=True)

        self._publisher.run()  # initializes topics and writes metadata

    def run(self):

        if not self._client.isRunning():

            # manually run client if not running
            self._client.run()

        self._init_publisher()

    def close(self):

        try:
            self._client.close()
        except Exception:
            pass

    def update(self):

        success = self._synch_from_shared_mem()  # update publisher view with shared memory data

        if self._ros_backend == "ros2":

            self._publisher.pub_data(copy=True)

        if self._ros_backend == "ros1":

            self._publisher.pub_data()

        return success
