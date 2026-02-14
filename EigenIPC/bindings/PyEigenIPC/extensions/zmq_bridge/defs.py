import struct
import zlib

import numpy as np

from EigenIPC.PyEigenIPC import dtype as eigenipc_dtype


MAGIC = b"EIZM"
PROTOCOL_VERSION = 1

MSG_DATA = 1

DTYPE_BOOL = 0
DTYPE_INT32 = 1
DTYPE_FLOAT32 = 2
DTYPE_FLOAT64 = 3

FLAG_NONE = 0
FLAG_STRING_TENSOR = 1 << 0

# magic(4s), version(B), msg_type(B), dtype_code(B), flags(B), rows(I), cols(I), seq(Q), payload_nbytes(I)
HEADER_FORMAT = "<4sBBBBIIQI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class Header:

    def __init__(self,
        msg_type: int,
        dtype_code: int,
        flags: int,
        n_rows: int,
        n_cols: int,
        seq: int,
        payload_nbytes: int):

        self.msg_type = msg_type
        self.dtype_code = dtype_code
        self.flags = flags
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.seq = seq
        self.payload_nbytes = payload_nbytes


class NamingConventions:

    def __init__(self,
        default_ip: str = "127.0.0.1",
        port_base: int = 20000,
        port_span: int = 40000):

        self.default_ip = default_ip
        self.port_base = port_base
        self.port_span = port_span

    def stream_name(self,
        namespace: str,
        basename: str):

        if namespace is None or namespace == "":
            return basename

        return f"{namespace}/{basename}"

    def stream_port(self,
        namespace: str,
        basename: str,
        port_base: int = None,
        port_span: int = None):

        if port_base is None:
            port_base = self.port_base

        if port_span is None:
            port_span = self.port_span

        stream_id = self.stream_name(namespace=namespace, basename=basename)

        checksum = zlib.crc32(stream_id.encode("utf-8")) & 0xFFFFFFFF

        return port_base + (checksum % port_span)

    def endpoint(self,
        namespace: str,
        basename: str,
        ip: str = None,
        port: int = None,
        port_base: int = None,
        port_span: int = None):

        if ip is None:
            ip = self.default_ip

        if port is None:
            port = self.stream_port(
                namespace=namespace,
                basename=basename,
                port_base=port_base,
                port_span=port_span,
            )

        return f"tcp://{ip}:{int(port)}"


def encode_dtype(np_dtype):

    if np_dtype == np.bool_:
        return DTYPE_BOOL

    if np_dtype == np.int32:
        return DTYPE_INT32

    if np_dtype == np.float32:
        return DTYPE_FLOAT32

    if np_dtype == np.float64:
        return DTYPE_FLOAT64

    raise ValueError(f"Unsupported NumPy dtype: {np_dtype}")


def decode_dtype(dtype_code: int):

    if dtype_code == DTYPE_BOOL:
        return np.bool_

    if dtype_code == DTYPE_INT32:
        return np.int32

    if dtype_code == DTYPE_FLOAT32:
        return np.float32

    if dtype_code == DTYPE_FLOAT64:
        return np.float64

    raise ValueError(f"Unsupported encoded dtype: {dtype_code}")


def to_eigenipc_dtype(np_dtype):

    if np_dtype == np.bool_:
        return eigenipc_dtype.Bool

    if np_dtype == np.int32:
        return eigenipc_dtype.Int

    if np_dtype == np.float32:
        return eigenipc_dtype.Float

    if np_dtype == np.float64:
        return eigenipc_dtype.Double

    raise ValueError(f"Unsupported NumPy dtype: {np_dtype}")


def is_string_tensor(flags: int):

    return (int(flags) & FLAG_STRING_TENSOR) != 0


def payload_nbytes(n_rows: int,
    n_cols: int,
    np_dtype):

    return int(n_rows) * int(n_cols) * np.dtype(np_dtype).itemsize


def pack_header(msg_type: int,
    dtype_code: int,
    flags: int,
    n_rows: int,
    n_cols: int,
    seq: int,
    payload_nbytes: int):

    return struct.pack(
        HEADER_FORMAT,
        MAGIC,
        PROTOCOL_VERSION,
        int(msg_type),
        int(dtype_code),
        int(flags),
        int(n_rows),
        int(n_cols),
        int(seq),
        int(payload_nbytes),
    )


def unpack_header(header_bytes: bytes):

    if len(header_bytes) != HEADER_SIZE:
        raise ValueError(f"Invalid header size {len(header_bytes)}. Expected {HEADER_SIZE}")

    magic, version, msg_type, dtype_code, flags, n_rows, n_cols, seq, n_payload = struct.unpack(
        HEADER_FORMAT,
        header_bytes,
    )

    if magic != MAGIC:
        raise ValueError(f"Invalid magic in ZMQ frame. Got {magic}")

    if version != PROTOCOL_VERSION:
        raise ValueError(
            f"Unsupported protocol version {version}. Expected {PROTOCOL_VERSION}"
        )

    return Header(
        msg_type=msg_type,
        dtype_code=dtype_code,
        flags=flags,
        n_rows=n_rows,
        n_cols=n_cols,
        seq=seq,
        payload_nbytes=n_payload,
    )
