import argparse
import time

import numpy as np

from EigenIPC.PyEigenIPC import (
    ClientFactory,
    StringTensorClient,
    VLevel,
    dtype,
)
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.from_zmq import FromZmq

from common_test_data import (
    NAMESPACE,
    REMAP_NS,
    NUMERIC_BASENAME,
    STRING_BASENAME,
    NUMERIC_DATA,
    STRING_DATA,
    CHECK_DT_S,
    WAIT_TIMEOUT_S,
)


def _parse_args():

    parser = argparse.ArgumentParser(description="ZMQ receiver + shared-memory validation client")
    parser.add_argument("--namespace", type=str, default=NAMESPACE)
    parser.add_argument("--remap-ns", type=str, default=REMAP_NS)
    parser.add_argument("--numeric-basename", type=str, default=NUMERIC_BASENAME)
    parser.add_argument("--string-basename", type=str, default=STRING_BASENAME)

    parser.add_argument("--sender-ip", type=str, default="127.0.0.1")
    parser.add_argument("--numeric-port", type=int, default=23001)
    parser.add_argument("--string-port", type=int, default=23002)
    parser.add_argument("--numeric-endpoint", type=str, default=None)
    parser.add_argument("--string-endpoint", type=str, default=None)

    parser.add_argument("--queue-size", type=int, default=1)
    parser.add_argument("--no-conflate", action="store_true")
    parser.add_argument("--timeout-ms", type=int, default=50)

    parser.add_argument("--wait-timeout", type=float, default=WAIT_TIMEOUT_S)
    parser.add_argument("--check-dt", type=float, default=CHECK_DT_S)

    return parser.parse_args()


def _endpoint(explicit, ip, port):

    if explicit is not None:
        return explicit

    return f"tcp://{ip}:{port}"


def _wait_bridge(bridge, timeout_s):

    start = time.time()
    while (time.time() - start) < timeout_s:
        if bridge.run():
            return True
        time.sleep(0.05)

    return False


def main():

    args = _parse_args()

    numeric_endpoint = _endpoint(args.numeric_endpoint, args.sender_ip, args.numeric_port)
    string_endpoint = _endpoint(args.string_endpoint, args.sender_ip, args.string_port)
    use_conflate = not args.no_conflate

    numeric_bridge = FromZmq(
        basename=args.numeric_basename,
        namespace=args.namespace,
        endpoint=numeric_endpoint,
        remap_ns=args.remap_ns,
        queue_size=args.queue_size,
        conflate=use_conflate,
        timeout_ms=args.timeout_ms,
        verbose=True,
        vlevel=VLevel.V2,
        force_reconnection=True,
    )

    string_bridge = FromZmq(
        basename=args.string_basename,
        namespace=args.namespace,
        endpoint=string_endpoint,
        remap_ns=args.remap_ns,
        queue_size=args.queue_size,
        conflate=use_conflate,
        timeout_ms=args.timeout_ms,
        verbose=True,
        vlevel=VLevel.V2,
        force_reconnection=True,
    )

    if not _wait_bridge(numeric_bridge, args.wait_timeout):
        raise RuntimeError(
            "Timeout while waiting for numeric FromZmq bridge to initialize. "
            f"endpoint={numeric_endpoint}"
        )

    if not _wait_bridge(string_bridge, args.wait_timeout):
        raise RuntimeError(
            "Timeout while waiting for string FromZmq bridge to initialize. "
            f"endpoint={string_endpoint}"
        )

    numeric_client = ClientFactory(
        basename=args.numeric_basename,
        namespace=args.remap_ns,
        verbose=True,
        vlevel=VLevel.V2,
        dtype=dtype.Float,
        safe=True,
    )
    numeric_client.attach()

    string_client = StringTensorClient(
        basename=args.string_basename,
        name_space=args.remap_ns,
        verbose=True,
        vlevel=VLevel.V2,
        safe=True,
    )
    string_client.run()

    numeric_buffer = np.empty_like(NUMERIC_DATA)
    string_buffer = [""] * len(STRING_DATA)

    print(
        "Receiver running checks. "
        f"numeric={numeric_endpoint}, string={string_endpoint}, conflate={use_conflate}."
    )

    start = time.time()
    numeric_ok = False
    string_ok = False

    try:
        while (time.time() - start) < args.wait_timeout:

            numeric_bridge.update(retry_write=True)
            string_bridge.update(retry_write=True)

            numeric_read = numeric_client.read(numeric_buffer, 0, 0)
            string_read = string_client.read_vec(string_buffer, 0)

            if numeric_read:
                numeric_ok = np.allclose(numeric_buffer, NUMERIC_DATA, atol=1e-6, rtol=0.0)

            if string_read:
                string_ok = string_buffer == STRING_DATA

            if numeric_ok and string_ok:
                print("SUCCESS: numeric and string tensor data match expected values.")
                return

            time.sleep(args.check_dt)

        raise RuntimeError(
            "Validation timeout. "
            f"numeric_ok={numeric_ok}, string_ok={string_ok}, "
            f"numeric_last={numeric_buffer}, string_last={string_buffer}"
        )

    finally:
        try:
            numeric_bridge.close()
        except Exception:
            pass
        try:
            string_bridge.close()
        except Exception:
            pass
        try:
            numeric_client.close()
        except Exception:
            pass
        try:
            string_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
