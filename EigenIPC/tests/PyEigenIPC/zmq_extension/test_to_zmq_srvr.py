import argparse
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


def _parse_args():

    parser = argparse.ArgumentParser(description="ZMQ sender for numeric + string shared-memory data")
    parser.add_argument("--namespace", type=str, default=NAMESPACE)
    parser.add_argument("--numeric-basename", type=str, default=NUMERIC_BASENAME)
    parser.add_argument("--string-basename", type=str, default=STRING_BASENAME)

    parser.add_argument("--bind-ip", type=str, default="0.0.0.0")
    parser.add_argument("--numeric-port", type=int, default=23001)
    parser.add_argument("--string-port", type=int, default=23002)
    parser.add_argument("--numeric-endpoint", type=str, default=None)
    parser.add_argument("--string-endpoint", type=str, default=None)

    parser.add_argument("--queue-size", type=int, default=1)
    parser.add_argument("--no-conflate", action="store_true")
    parser.add_argument("--publish-dt", type=float, default=PUBLISH_DT_S)

    return parser.parse_args()


def _endpoint(explicit, ip, port):

    if explicit is not None:
        return explicit

    return f"tcp://{ip}:{port}"


def main():

    args = _parse_args()

    numeric_endpoint = _endpoint(args.numeric_endpoint, args.bind_ip, args.numeric_port)
    string_endpoint = _endpoint(args.string_endpoint, args.bind_ip, args.string_port)
    use_conflate = not args.no_conflate

    numeric_server = ServerFactory(
        n_rows=NUMERIC_DATA.shape[0],
        n_cols=NUMERIC_DATA.shape[1],
        basename=args.numeric_basename,
        namespace=args.namespace,
        verbose=True,
        vlevel=VLevel.V2,
        force_reconnection=True,
        dtype=dtype.Float,
        safe=True,
    )
    numeric_server.run()

    numeric_client = ClientFactory(
        basename=args.numeric_basename,
        namespace=args.namespace,
        verbose=True,
        vlevel=VLevel.V2,
        dtype=dtype.Float,
        safe=True,
    )
    numeric_client.attach()

    string_server = StringTensorServer(
        length=len(STRING_DATA),
        basename=args.string_basename,
        name_space=args.namespace,
        verbose=True,
        vlevel=VLevel.V2,
        force_reconnection=True,
        safe=True,
    )
    string_server.run()

    string_client = StringTensorClient(
        basename=args.string_basename,
        name_space=args.namespace,
        verbose=True,
        vlevel=VLevel.V2,
        safe=True,
    )
    string_client.run()

    numeric_bridge = ToZmq(
        client=numeric_client,
        endpoint=numeric_endpoint,
        queue_size=args.queue_size,
        conflate=use_conflate,
    )
    string_bridge = ToZmq(
        client=string_client,
        endpoint=string_endpoint,
        queue_size=args.queue_size,
        conflate=use_conflate,
    )

    numeric_bridge.run()
    string_bridge.run()

    print(
        "ZMQ sender running. "
        f"numeric={numeric_endpoint}, string={string_endpoint}, "
        f"conflate={use_conflate}. Press Ctrl+C to stop."
    )

    try:
        while True:
            numeric_server.write(NUMERIC_DATA, 0, 0)
            string_server.write_vec(STRING_DATA, 0)

            numeric_bridge.update(retry=True)
            string_bridge.update(retry=True)

            time.sleep(args.publish_dt)

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
