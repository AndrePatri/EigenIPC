import time

from EigenIPC.PyEigenIPC import (
    ServerFactory,
    ClientFactory,
    StringTensorServer,
    StringTensorClient,
    VLevel,
    dtype,
)
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.to_zmq import ToZmq

from common_test_data import (
    NAMESPACE,
    NUMERIC_BASENAME,
    STRING_BASENAME,
    NUMERIC_DATA,
    STRING_DATA,
    PUBLISH_DT_S,
)


def main():

    numeric_server = ServerFactory(
        n_rows=NUMERIC_DATA.shape[0],
        n_cols=NUMERIC_DATA.shape[1],
        basename=NUMERIC_BASENAME,
        namespace=NAMESPACE,
        verbose=True,
        vlevel=VLevel.V2,
        force_reconnection=True,
        dtype=dtype.Float,
        safe=True,
    )
    numeric_server.run()

    numeric_client = ClientFactory(
        basename=NUMERIC_BASENAME,
        namespace=NAMESPACE,
        verbose=True,
        vlevel=VLevel.V2,
        dtype=dtype.Float,
        safe=True,
    )
    numeric_client.attach()

    string_server = StringTensorServer(
        length=len(STRING_DATA),
        basename=STRING_BASENAME,
        name_space=NAMESPACE,
        verbose=True,
        vlevel=VLevel.V2,
        force_reconnection=True,
        safe=True,
    )
    string_server.run()

    string_client = StringTensorClient(
        basename=STRING_BASENAME,
        name_space=NAMESPACE,
        verbose=True,
        vlevel=VLevel.V2,
        safe=True,
    )
    string_client.run()

    numeric_bridge = ToZmq(client=numeric_client, queue_size=1, conflate=True)
    string_bridge = ToZmq(client=string_client, queue_size=1, conflate=True)

    numeric_bridge.run()
    string_bridge.run()

    print("ZMQ sender running. Press Ctrl+C to stop.")

    try:
        while True:
            numeric_server.write(NUMERIC_DATA, 0, 0)
            string_server.write_vec(STRING_DATA, 0)

            numeric_bridge.update(retry=True)
            string_bridge.update(retry=True)

            time.sleep(PUBLISH_DT_S)

    except KeyboardInterrupt:
        print("Stopped sender.")

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
            numeric_server.close()
        except Exception:
            pass
        try:
            string_server.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
