from EigenIPC.PyEigenIPC import ServerFactory, VLevel

from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.abstractions import ZmqSubscriber, default_endpoint
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.defs import MSG_DATA, decode_dtype, to_eigenipc_dtype


class FromZmq:

    def __init__(self,
        basename: str,
        namespace: str = "",
        endpoint: str = None,
        ip: str = None,
        port: int = None,
        connect: bool = True,
        queue_size: int = 1,
        conflate: bool = True,
        timeout_ms: int = 0,
        vlevel: VLevel = VLevel.V3,
        verbose: bool = True,
        force_reconnection: bool = False,
        remap_ns: str = None):

        self._basename = basename
        self._namespace = namespace
        self._remap_ns = self._namespace if remap_ns is None else remap_ns

        self._vlevel = vlevel
        self._verbose = verbose
        self._force_reconnection = force_reconnection

        self._endpoint = endpoint
        if self._endpoint is None:
            self._endpoint = default_endpoint(
                namespace=self._namespace,
                basename=self._basename,
                ip=ip,
                port=port,
            )

        self._subscriber = ZmqSubscriber(
            endpoint=self._endpoint,
            connect=connect,
            queue_size=queue_size,
            conflate=conflate,
            timeout_ms=timeout_ms,
        )

        self._server = None
        self._is_running = False

    def _write_to_shared(self,
        np_data,
        retry: bool = True):

        if retry:
            while not self._server.write(np_data[:, :], 0, 0):
                continue
            return True

        return self._server.write(np_data[:, :], 0, 0)

    def _create_server(self,
        header):

        np_dtype = decode_dtype(header.dtype_code)

        self._server = ServerFactory(
            n_rows=header.n_rows,
            n_cols=header.n_cols,
            basename=self._basename,
            namespace=self._remap_ns,
            verbose=self._verbose,
            vlevel=self._vlevel,
            force_reconnection=self._force_reconnection,
            dtype=to_eigenipc_dtype(np_dtype),
            safe=True,
        )

        self._server.run()

    def run(self):

        if self._is_running:
            return True

        self._subscriber.run()

        header, payload = self._subscriber.recv_latest()
        if header is None:
            return False

        if header.msg_type != MSG_DATA:
            return False

        data = self._subscriber.payload_to_numpy(header, payload, copy=True)

        self._create_server(header)
        self._write_to_shared(data, retry=True)

        self._is_running = True

        return True

    def close(self):

        self._subscriber.close()
    
        if self._server is not None:
            self._server.close()
            self._server = None

    def update(self,
        retry_write: bool = True):

        if not self._is_running:
            return False

        header, payload = self._subscriber.recv_latest()
        if header is None:
            return False

        if header.msg_type != MSG_DATA:
            return False

        data = self._subscriber.payload_to_numpy(header, payload, copy=False)

        return self._write_to_shared(data, retry=retry_write)
