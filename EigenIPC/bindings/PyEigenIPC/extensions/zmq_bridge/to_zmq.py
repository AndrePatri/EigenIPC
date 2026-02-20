from typing import Union
import re

import numpy as np

from EigenIPC.PyEigenIPC import Client
from EigenIPC.PyEigenIPC import StringTensorClient
from EigenIPC.PyEigenIPC import Journal, LogType, toNumpyDType

from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.abstractions import ZmqPublisher, default_endpoint
from EigenIPC.PyEigenIPCExt.extensions.zmq_bridge.defs import FLAG_NONE, FLAG_STRING_TENSOR


class ToZmq:

    def __init__(self,
        client: Union[Client, StringTensorClient],
        endpoint: str = None,
        ip: str = None,
        port: int = None,
        bind: bool = True,
        queue_size: int = 1,
        conflate: bool = True,
        drop_if_busy: bool = False,
        source_row_index: int = None,
        source_n_rows: int = 1):

        self._check_client(client)

        self._client = client
        self._bind = bind
        self._queue_size = queue_size
        self._conflate = conflate

        self._source_row_index = source_row_index
        self._source_n_rows = source_n_rows

        self._is_string_tensor = self._is_string_tensor_client(client)

        self._endpoint = endpoint
        if self._endpoint is None:
            self._endpoint = default_endpoint(
                namespace=self._client.getNamespace(),
                basename=self._client.getBasename(),
                ip=ip,
                port=port,
            )

        self._publisher = ZmqPublisher(
            endpoint=self._endpoint,
            bind=self._bind,
            queue_size=self._queue_size,
            conflate=self._conflate,
            drop_if_busy=drop_if_busy,
        )

        self._tx_data = None
        self._stringtensor_data = None

        self._read_row_index = 0
        self._read_n_rows = None

        self._check_slice_config()

    def _client_type_fqn(self, client) -> str:

        return f"{type(client).__module__}.{type(client).__name__}"

    def _is_string_tensor_client(self, client) -> bool:

        if isinstance(client, StringTensorClient):
            return True

        fqn = self._client_type_fqn(client)
        return re.match(r"^(EigenIPC\.)?PyEigenIPC\.StringTensorClient$", fqn) is not None

    def _is_numeric_client(self, client) -> bool:

        if isinstance(client, Client):
            return True

        fqn = self._client_type_fqn(client)
        return re.match(r"^(EigenIPC\.)?PyEigenIPC\.PyClient[A-Za-z0-9_]*$", fqn) is not None

    def _check_client(self,
        client: Union[Client, StringTensorClient]):

        if self._is_numeric_client(client) or self._is_string_tensor_client(client):
            return

        fqn = self._client_type_fqn(client)
        exception = (
            "Provided client has unsupported type. Got type: "
            f"{fqn}. Expected (EigenIPC.)?PyEigenIPC.PyClient* "
            "or (EigenIPC.)?PyEigenIPC.StringTensorClient."
        )

        Journal.log(self.__class__.__name__,
            "_check_client",
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

    def _init_tx_buffer(self):

        if self._is_string_tensor:
            length = self._client.length()
            self._stringtensor_data = [""] * length
            self._tx_data = None
            return

        self._resolve_numeric_slice()

        n_cols = self._client.getNCols()
        np_dtype = toNumpyDType(self._client.getScalarType())

        self._tx_data = np.empty((self._read_n_rows, n_cols), dtype=np_dtype)

    def _synch_from_shared_mem(self,
        retry: bool = True):

        if retry:
            if not self._is_string_tensor:
                while not self._client.read(self._tx_data[:, :], self._read_row_index, 0):
                    continue
                return True

            while True:
                read = self._client.read_vec(self._stringtensor_data, 0)
                if not read:
                    continue
                self._tx_data = self._client.get_raw_buffer()
                return True

        if not self._is_string_tensor:
            return self._client.read(self._tx_data[:, :], self._read_row_index, 0)

        read = self._client.read_vec(self._stringtensor_data, 0)
        if not read:
            return False

        self._tx_data = self._client.get_raw_buffer()
        return True

    def run(self):

        if not self._client.isRunning():
            self._client.run()

        self._init_tx_buffer()
        self._publisher.run()

    def close(self):

        try:
            self._publisher.close()
        except Exception:
            pass

        try:
            self._client.close()
        except Exception:
            pass

    def update(self,
        retry: bool = True):

        success = self._synch_from_shared_mem(retry=retry)

        if not success:
            return False

        flags = FLAG_STRING_TENSOR if self._is_string_tensor else FLAG_NONE
        published = self._publisher.publish_numpy(self._tx_data, flags=flags)

        return bool(published)
