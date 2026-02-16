from EigenIPC.PyEigenIPC import ServerFactory

from EigenIPC.PyEigenIPC import Journal, VLevel, LogType, dtype

import numpy as np


class FromRos:

    # Atomic bridge element to forward data from ROS topics
    # into shared memory through PyEigenIPC.

    def __init__(self,
                basename: str,
                namespace: str = "",
                queue_size: int = 1,
                ros_backend="ros1",
                vlevel=VLevel.V3,
                verbose: bool = True,
                force_reconnection: bool = False,
                node=None,
                remap_ns: str = None):

        self._queue_size = queue_size

        self._basename = basename
        self._namespace = namespace
        self._remap_ns = self._namespace if remap_ns is None else remap_ns

        self._vlevel = vlevel
        self._verbose = verbose
        self._force_reconnection = force_reconnection

        self._subscriber = None

        self._node = node  # only used when ros2

        self._server = None

        self._is_running = False

        self._ros_backend = ros_backend

        self._check_backend()
        self._init_subscriber()

    def _check_backend(self):

        if not (self._ros_backend == "ros1" or self._ros_backend == "ros2"):

            exception = f"Unsupported ROS backend {self._ros_backend}. Supported are \"ros1\" and \"ros2\""

            Journal.log(self.__class__.__name__,
                        "_check_backend",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep=True)

    def _init_subscriber(self):

        if self._ros_backend == "ros1":

            if self._node is not None:

                warn = "A node argument was provided to constructor but when using ros1 backend, that's not necessary!"

                Journal.log(self.__class__.__name__,
                            "_init_subscriber",
                            warn,
                            LogType.WARN,
                            throw_when_excep=True)

            from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.ros1_utils import Ros1Subscriber

            self._subscriber = Ros1Subscriber(
                basename=self._basename,
                namespace=self._namespace,
                queue_size=self._queue_size,
            )

        elif self._ros_backend == "ros2":

            if self._node is None:

                exception = "No node argument provided to constructor! When using ros2 backend, you should provide it!"

                Journal.log(self.__class__.__name__,
                            "_init_subscriber",
                            exception,
                            LogType.EXCEP,
                            throw_when_excep=True)

            from EigenIPC.PyEigenIPCExt.extensions.ros_bridge.ros2_utils import Ros2Subscriber

            self._subscriber = Ros2Subscriber(
                basename=self._basename,
                namespace=self._namespace,
                queue_size=self._queue_size,
                node=self._node,
            )

        else:

            exception = f"backend {self._ros_backend} not supported. Please use either \"ros1\" or \"ros2\"!"

            Journal.log(self.__class__.__name__,
                        "_init_subscriber",
                        exception,
                        LogType.EXCEP,
                        throw_when_excep=True)

    def _write_to_shared(self,
                    retry: bool = True):

        if retry:

            while not self._server.write(self._subscriber.np_data[:, :], 0, 0):
                continue

            return True

        return self._server.write(self._subscriber.np_data[:, :], 0, 0)

    def _to_eigenipc_dtype(self,
                    np_dtype):

        if np_dtype == np.bool_:
            return dtype.Bool

        if np_dtype == np.int32:
            return dtype.Int

        if np_dtype == np.float32:
            return dtype.Float

        if np_dtype == np.float64:
            return dtype.Double

        exception = f"Unsupported numpy dtype {np_dtype}"
        Journal.log(self.__class__.__name__,
                    "_to_eigenipc_dtype",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep=True)

    def run(self):

        if self._is_running:
            return True

        sub_success = self._subscriber.run()  # initializes metadata/data subscriptions

        if not sub_success:
            return False

        self._server = ServerFactory(
            n_rows=self._subscriber.n_rows(),
            n_cols=self._subscriber.n_cols(),
            basename=self._basename,
            namespace=self._remap_ns,
            verbose=self._verbose,
            vlevel=self._vlevel,
            force_reconnection=self._force_reconnection,
            dtype=self._to_eigenipc_dtype(self._subscriber.dtype()),
            safe=True,
        )

        self._server.run()

        self._is_running = True

        return True

    def close(self):

        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None

        self._is_running = False

    def update(self):

        if not self._is_running:
            return False

        self._subscriber.acquire_data()  # blocking until callback write is done

        return self._write_to_shared()
