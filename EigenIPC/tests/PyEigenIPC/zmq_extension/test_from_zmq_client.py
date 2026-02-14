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


def _wait_bridge(bridge, timeout_s):

    start = time.time()
    while (time.time() - start) < timeout_s:
        if bridge.run():
            return True
        time.sleep(0.05)

    return False


def main():

    numeric_bridge = FromZmq(
        basename=NUMERIC_BASENAME,
        namespace=NAMESPACE,
        remap_ns=REMAP_NS,
        queue_size=1,
        conflate=True,
        timeout_ms=50,
        verbose=True,
        vlevel=VLevel.V2,
        force_reconnection=True,
    )

    string_bridge = FromZmq(
        basename=STRING_BASENAME,
        namespace=NAMESPACE,
        remap_ns=REMAP_NS,
        queue_size=1,
        conflate=True,
        timeout_ms=50,
        verbose=True,
        vlevel=VLevel.V2,
        force_reconnection=True,
    )

    if not _wait_bridge(numeric_bridge, WAIT_TIMEOUT_S):
        raise RuntimeError("Timeout while waiting for numeric FromZmq bridge to initialize.")

    if not _wait_bridge(string_bridge, WAIT_TIMEOUT_S):
        raise RuntimeError("Timeout while waiting for string FromZmq bridge to initialize.")

    numeric_client = ClientFactory(
        basename=NUMERIC_BASENAME,
        namespace=REMAP_NS,
        verbose=True,
        vlevel=VLevel.V2,
        dtype=dtype.Float,
        safe=True,
    )
    numeric_client.attach()

    string_client = StringTensorClient(
        basename=STRING_BASENAME,
        name_space=REMAP_NS,
        verbose=True,
        vlevel=VLevel.V2,
        safe=True,
    )
    string_client.run()

    numeric_buffer = np.empty_like(NUMERIC_DATA)
    string_buffer = [""] * len(STRING_DATA)

    print("Receiver running checks...")

    start = time.time()
    numeric_ok = False
    string_ok = False

    try:
        while (time.time() - start) < WAIT_TIMEOUT_S:

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

            time.sleep(CHECK_DT_S)

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
