from typing import Optional

import numpy as np
import zmq

from EigenIPC.PyEigenIPC import Journal, LogType

from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.defs import (
    HEADER_SIZE,
    MSG_DATA,
    NamingConventions,
    decode_dtype,
    encode_dtype,
    pack_header,
    payload_nbytes,
    unpack_header,
)


class ZmqPublisher:

    def __init__(self,
        endpoint: str,
        bind: bool = True,
        queue_size: int = 1,
        linger_ms: int = 0,
        conflate: bool = True,
        context: Optional[zmq.Context] = None):

        self._endpoint = endpoint
        self._bind = bind
        self._queue_size = max(1, int(queue_size))
        self._linger_ms = int(linger_ms)
        self._conflate = bool(conflate)

        self._owned_context = context is None
        self._context = context if context is not None else zmq.Context.instance()
        self._socket = None

        self._seq = 0
        self._running = False

    def __del__(self):

        self.close()

    def run(self):

        if self._running:
            return

        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.LINGER, self._linger_ms)
        self._socket.setsockopt(zmq.SNDHWM, self._queue_size)

        if hasattr(zmq, "CONFLATE"):
            self._socket.setsockopt(zmq.CONFLATE, 1 if self._conflate else 0)

        if self._bind:
            self._socket.bind(self._endpoint)
        else:
            self._socket.connect(self._endpoint)

        self._running = True

    def publish_numpy(self,
        np_data: np.ndarray,
        flags: int = 0,
        msg_type: int = MSG_DATA):

        if not self._running:
            exception = "Publisher is not running. Call run() first."
            Journal.log(self.__class__.__name__,
                "publish_numpy",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)

        if np_data.ndim != 2:
            exception = f"Expected 2D array. Got shape {np_data.shape}."
            Journal.log(self.__class__.__name__,
                "publish_numpy",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)

        tx_data = np_data
        if not tx_data.flags.c_contiguous:
            tx_data = np.ascontiguousarray(tx_data)

        dtype_code = encode_dtype(tx_data.dtype.type)

        header = pack_header(
            msg_type=msg_type,
            dtype_code=dtype_code,
            flags=flags,
            n_rows=tx_data.shape[0],
            n_cols=tx_data.shape[1],
            seq=self._seq,
            payload_nbytes=tx_data.nbytes,
        )

        self._socket.send(header, flags=zmq.SNDMORE)
        self._socket.send(memoryview(tx_data), copy=False)

        self._seq += 1

    def close(self):

        if self._socket is not None:
            try:
                self._socket.close(linger=self._linger_ms)
            except Exception:
                pass
            self._socket = None

        if self._owned_context and self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass

        self._running = False


class ZmqSubscriber:

    def __init__(self,
        endpoint: str,
        connect: bool = True,
        queue_size: int = 1,
        linger_ms: int = 0,
        conflate: bool = True,
        timeout_ms: int = 0,
        context: Optional[zmq.Context] = None):

        self._endpoint = endpoint
        self._connect = connect
        self._queue_size = max(1, int(queue_size))
        self._linger_ms = int(linger_ms)
        self._conflate = bool(conflate)
        self._timeout_ms = int(timeout_ms)

        self._owned_context = context is None
        self._context = context if context is not None else zmq.Context.instance()
        self._socket = None
        self._poller = None

        self._running = False

    def __del__(self):

        self.close()

    def run(self):

        if self._running:
            return

        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.LINGER, self._linger_ms)
        self._socket.setsockopt(zmq.RCVHWM, self._queue_size)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")

        if hasattr(zmq, "CONFLATE"):
            self._socket.setsockopt(zmq.CONFLATE, 1 if self._conflate else 0)

        if self._connect:
            self._socket.connect(self._endpoint)
        else:
            self._socket.bind(self._endpoint)

        self._poller = zmq.Poller()
        self._poller.register(self._socket, zmq.POLLIN)

        self._running = True

    def recv_latest(self):

        if not self._running:
            exception = "Subscriber is not running. Call run() first."
            Journal.log(self.__class__.__name__,
                "recv_latest",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)

        timeout = self._timeout_ms
        events = dict(self._poller.poll(timeout=timeout))
        if self._socket not in events:
            return None, None

        frames = self._socket.recv_multipart(copy=False)

        if not self._conflate:
            while True:
                events = dict(self._poller.poll(timeout=0))
                if self._socket not in events:
                    break
                frames = self._socket.recv_multipart(copy=False)

        if len(frames) != 2:
            exception = f"Malformed ZMQ message: expected 2 frames, got {len(frames)}"
            Journal.log(self.__class__.__name__,
                "recv_latest",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)

        header_bytes = bytes(memoryview(frames[0]))
        if len(header_bytes) != HEADER_SIZE:
            exception = (
                f"Malformed ZMQ header frame size {len(header_bytes)}. "
                f"Expected {HEADER_SIZE}"
            )
            Journal.log(self.__class__.__name__,
                "recv_latest",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)

        header = unpack_header(header_bytes)
        payload = memoryview(frames[1])

        if len(payload) != header.payload_nbytes:
            exception = (
                f"Payload size mismatch. Got {len(payload)} bytes, "
                f"header declares {header.payload_nbytes}"
            )
            Journal.log(self.__class__.__name__,
                "recv_latest",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)

        return header, payload

    def payload_to_numpy(self,
        header,
        payload,
        copy: bool = False):

        np_dtype = decode_dtype(header.dtype_code)

        expected_nbytes = payload_nbytes(
            n_rows=header.n_rows,
            n_cols=header.n_cols,
            np_dtype=np_dtype,
        )

        if expected_nbytes != len(payload):
            exception = (
                f"Invalid payload length {len(payload)} for shape "
                f"({header.n_rows}, {header.n_cols}) and dtype {np_dtype}. "
                f"Expected {expected_nbytes}."
            )
            Journal.log(self.__class__.__name__,
                "payload_to_numpy",
                exception,
                LogType.EXCEP,
                throw_when_excep=True)

        array_1d = np.frombuffer(payload, dtype=np_dtype, count=header.n_rows * header.n_cols)
        array_2d = array_1d.reshape((header.n_rows, header.n_cols))

        if copy:
            return array_2d.copy()

        return array_2d

    def close(self):

        if self._socket is not None:
            try:
                self._socket.close(linger=self._linger_ms)
            except Exception:
                pass
            self._socket = None

        self._poller = None

        if self._owned_context and self._context is not None:
            try:
                self._context.term()
            except Exception:
                pass

        self._running = False


def default_endpoint(namespace: str,
    basename: str,
    ip: str = None,
    port: int = None,
    port_base: int = None,
    port_span: int = None):

    naming = NamingConventions()

    return naming.endpoint(
        namespace=namespace,
        basename=basename,
        ip=ip,
        port=port,
        port_base=port_base,
        port_span=port_span,
    )
