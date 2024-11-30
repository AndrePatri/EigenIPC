import unittest
import numpy as np

import timeit

from EigenIPC.PyEigenIPC import *

import psutil

process = psutil.Process()

def get_memory_usage():
    
    return process.memory_info().rss / (1024**3)

namespace = "RAMTests"

N_ITERATIONS = 10000

N_ROWS = 10000
N_COLS = 10000

STR_TENSOR_LENGTH = 100

class TestPerfBenchBase(unittest.TestCase):

    # we only benchmark the Server write and
    # read method since the Client's versions
    # perform the same operations

    def setUp(self):

        self.rows = N_ROWS
        self.cols = N_COLS
        self.iterations = N_ITERATIONS

        self.is_release = isRelease()

        self.server_write = ServerFactory(self.rows,
                                    self.cols,
                                    basename="PySharsor_write" + \
                                    str(self.layout)  +
                                    str(self.data_type),
                                    namespace=namespace,
                                    verbose=True,
                                    vlevel=VLevel.V3,
                                    force_reconnection = True,
                                    dtype=self.data_type,
                                    layout=self.layout)

        self.client_read = ClientFactory(basename="PySharsor_write" + \
                                    str(self.layout)  +
                                    str(self.data_type),
                                    namespace=namespace,
                                    verbose=True,
                                    vlevel=VLevel.V3,
                                    dtype=self.data_type,
                                    layout=self.layout)
        self.server_write.run()
        self.client_read.attach()

        if self.layout == RowMajor:

            self.order = 'C' # 'C'

        if self.layout == ColMajor:

            self.order = 'F' # 'F'

        self.tensor_written = np.zeros((3 * self.server_write.getNRows(), 3 * self.server_write.getNCols()),
                                dtype=toNumpyDType(self.server_write.getScalarType()),
                                order=self.order)
        self.tensor_buffer = np.zeros((3 * self.server_write.getNRows(), 3 * self.server_write.getNCols()),
                                dtype=toNumpyDType(self.server_write.getScalarType()),
                                order=self.order)
        self.tensor_read = np.zeros((3 * self.client_read.getNRows(), 3 * self.client_read.getNCols()),
                                dtype=toNumpyDType(self.client_read.getScalarType()),
                                order=self.order)

        self.consistency_checks = [] # True -> OK; False -> failed

    def tearDown(self):

        self.server_write.close()
        self.client_read.close()

    def RamCheck(self):

        if self.is_release:

            message = "EigenIPC was compiled in Release mode. " + \
                "This is good and will ensure meaningful benchmarking results.\n"

            Journal.log(self.__class__.__name__,
                        "test_write_read",
                        message,
                        LogType.STAT)

        if not self.is_release:

            message = "EigenIPC was not compiled in Release mode. " + \
                "Please compile it in Release mode to get meaningful results.\n"

            Journal.log(self.__class__.__name__,
                        "test_write_read",
                        message,
                        LogType.WARN)

        row = self.server_write.getNRows()
        col = self.server_write.getNCols()

        ram_usage = [0.0] * N_ITERATIONS

        # randomize central block
        self.randomize(self.tensor_written,
                        row, col,
                        row, col)

        for i in range(0, N_ITERATIONS):

            self.server_write.read(self.tensor_buffer[row:2*row, col:2*col], 0, 0) # we read the tensor

            self.client_read.write(self.tensor_buffer[row:2*row, col:2*col], 0, 0) # we write it on the other memory

            a = 1.23232323
            ram_usage[i] = get_memory_usage()

        post_proc_message = f"Initial RAM usage: {ram_usage[0]} GB. Final RAM usage: {ram_usage[1]} GB."

        Journal.log(self.__class__.__name__,
                    "RamCheck",
                    post_proc_message,
                    LogType.STAT)

    def randomize(self, arr, i=0, j=0, n_rows=None, n_cols=None):

            # Set default values for n_rows and n_cols if they are None
            n_rows = n_rows or (arr.shape[0] - i)
            n_cols = n_cols or (arr.shape[1] - j)

            # Check if the indices and dimensions are valid
            if i < 0 or j < 0 or n_rows < 1 or n_cols < 1:

                raise ValueError("Invalid indices or dimensions")

            if i + n_rows > arr.shape[0] or j + n_cols > arr.shape[1]:

                raise ValueError("Submatrix dimensions exceed array bounds")

            # Define the submatrix bounds
            rows = slice(i, i + n_rows)
            cols = slice(j, j + n_cols)

            # Randomize the submatrix
            if np.issubdtype(arr.dtype, np.integer):

                arr[rows, cols] = np.random.randint(np.iinfo(arr.dtype).min,
                                                    np.iinfo(arr.dtype).max,
                                                    (n_rows, n_cols))

            elif arr.dtype == np.float32:

                arr[rows, cols] = np.random.uniform(-1, 1, (n_rows, n_cols))

            elif arr.dtype == np.float64:

                arr[rows, cols] = np.random.rand(n_rows, n_cols)

            elif arr.dtype == np.bool_:

                arr[rows, cols] = np.random.choice([True, False], (n_rows, n_cols))

            else:

                raise ValueError("Unsupported dtype for randomization")

class TestPerfBenchDoubleRowMaj(TestPerfBenchBase):

    data_type = dtype.Double
    layout = RowMajor

    def test_write_read(self):

        self.RamCheck()

if __name__ == "__main__":

    unittest.main()

    
