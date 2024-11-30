import numpy as np

from EigenIPC.PyEigenIPC import ServerFactory, ClientFactory
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import RowMajor, ColMajor
from EigenIPC.PyEigenIPC import toNumpyDType
from EigenIPC.PyEigenIPC import dtype as eigenipc_dtype 
from EigenIPC.PyEigenIPC import Journal as Logger
from EigenIPC.PyEigenIPC import LogType
    
class SharedTWrapper:

    def __init__(self, 
            namespace = "",
            basename = "",
            is_server = False, 
            n_rows: int = None, 
            n_cols: int = None, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            dtype: eigenipc_dtype = eigenipc_dtype.Float,
            with_torch_view: bool = False,
            with_gpu_mirror: bool = False,
            fill_value = None,
            safe = True,
            force_reconnection = False):

        self.basename = basename
        self.namespace = namespace

        self.fill_value = fill_value
        
        self.safe = safe

        self.verbose = verbose
        self.vlevel = vlevel

        self.is_server = is_server 

        self._shared_mem = None
        self._numpy_view = None
        self._torch_view = None

        self._with_torch_view = with_torch_view
        self._with_gpu_mirror = with_gpu_mirror
        if not self._with_torch_view: # no torch view -> cannot use gpu mirror
            self._with_gpu_mirror = False

        self._gpu_mirror = None 

        self.dtype = dtype

        self.layout = RowMajor
        if self.layout == RowMajor:

            self.order = 'C' # 'C'

        if self.layout == ColMajor:

            self.order = 'F' # 'F'

        if self.is_server:
            self._shared_mem = ServerFactory(n_rows = n_rows, 
                    n_cols = n_cols,
                    basename = self.basename,
                    namespace = self.namespace, 
                    verbose = self.verbose, 
                    vlevel = self.vlevel, 
                    force_reconnection = force_reconnection, 
                    dtype = self.dtype,
                    layout = self.layout,
                    safe = self.safe)
        else:
            self._shared_mem = ClientFactory(
                    basename = self.basename,
                    namespace = self.namespace, 
                    verbose = self.verbose, 
                    vlevel = self.vlevel,
                    dtype = self.dtype,
                    safe = self.safe)
        
        self._is_running = False
    
    def __del__(self):

        self.close()

    def _ensure2D(self, 
                data):
        
        if data.ndim == 2:
            return True
        else:
            return False

    def _fits_into(self, 
                data1, 
                data2, 
                row_index: int, 
                col_index: int):

        # Check if array1 fits into array2 at specific indexes, 
        # assuming both are 2D arrays or tensors

        # Calculate ending indices
        end_row_index = row_index + data1.shape[0]
        end_col_index = col_index + data1.shape[1]

        # Check if data1 fits within the bounds of data2
        return end_row_index <= data2.shape[0] and end_col_index <= data2.shape[1]

    def _init_gpu_mirror(self):
        
        if self._with_gpu_mirror:
            
            import torch
            if torch.cuda.is_available():
                # we copy torch view init. and dtype
                # (of course the mirror has to be updated manually)
                self._gpu_mirror = self._torch_view.to('cuda').requires_grad_(False)
            else:
                exception = f"GPU mirror cannot be initialized! No cuda device detected"
                Logger.log(self.__class__.__name__,
                    "_init_gpu_mirror",
                    exception,
                    LogType.EXCEP,
                    throw_when_excep = True)

    def get_n_clients(self):

        if self.is_server:
            return self._shared_mem.getNClients()
        else:
            message = "Number of clients can be retrieved only if is_server = True!!"
            Logger.log(self.__class__.__name__,
                "get_n_clients",
                message,
                LogType.EXCEP,
                throw_when_excep = True)
            return -1
    
    def is_running(self):

        return self._is_running
    
    def run(self):
    
        if self.is_server:

            self._shared_mem.run()
        
        else:
            
            self._shared_mem.attach()

        self.n_rows = self._shared_mem.getNRows()
        self.n_cols = self._shared_mem.getNCols()

        # we create views as big as the underlying shared memory
        # in case only a portion of it is needed, this is not optimal
        # memory-wise. However, this way we gain in simplicity
                
        if self.fill_value is not None:
            self._numpy_view = np.full((self.n_rows, self.n_cols),
                            self.fill_value,
                            dtype=toNumpyDType(self._shared_mem.getScalarType()),
                            order=self.order
                            )
        else:
            self._numpy_view = np.zeros((self.n_rows, self.n_cols),
                                dtype=toNumpyDType(self._shared_mem.getScalarType()),
                                order=self.order 
                                )
        
        if self._with_torch_view:
            import torch
            self._torch_view = torch.from_numpy(self._numpy_view).requires_grad_(False) # changes in either the 
        # numpy or torch view will be reflected into the other one

        # also write fill value to shared memory

        if self.fill_value is not None and self.is_server:
            # view is initialized with NaN -> 
            # we write initialization
            self.synch_all(read = False, 
                    retry=True)
        
        self._init_gpu_mirror() # does nothing if not _with_gpu_mirror

        self._is_running = True
                
    def write(self, 
            data, 
            row_index: int, 
            col_index: int):
        
        if isinstance(data, (int, float, bool,
                            np.float32, np.float64)):  
            
            # write scalar into numpy view
            self._numpy_view[row_index, col_index] = data
            # write to shared memory
            return self._shared_mem.write(self._numpy_view[row_index:row_index + 1, 
                                                        col_index:col_index + 1], 
                                        row_index, col_index)
        
        if isinstance(data, np.ndarray):  # Check if data is a numpy array
            if not self._ensure2D(data):
                message = "Provided data should be 2D!!"
                Logger.log(self.__class__.__name__,
                    "write",
                    message,
                    LogType.EXCEP,
                    throw_when_excep = True)
            if not self._fits_into(data, self._numpy_view, 
                                row_index, col_index):
                message = "Provided data does not fit in numpy view!!"
                Logger.log(self.__class__.__name__,
                    "write",
                    message,
                    LogType.EXCEP,
                    throw_when_excep = True)
            
            input_rows, input_cols = data.shape
            # insert data into part of numpy view
            self._numpy_view[row_index:row_index + input_rows, 
                    col_index:col_index + input_cols] = data
            # write corresponding part of numpy view to memory
            return self._shared_mem.write(self._numpy_view[row_index:row_index + input_rows, 
                        col_index:col_index + input_cols], row_index, col_index)                

        # here reached only if not got one of the valid dtypes
        message = "Unsupported data type provided. " + \
            "Supported types are: bool, int, float, double, " + \
            "numpy.ndarray"
        Logger.log(self.__class__.__name__,
            "write",
            message,
            LogType.EXCEP,
            throw_when_excep = True)

    def write_retry(self,
            data, 
            row_index: int, 
            col_index: int):

        # tries writing until success
        while self._shared_mem.isRunning() and not self.write(data=data,
                    row_index=row_index,
                    col_index=col_index):
            
            continue

    def read(self, 
            row_index: int, 
            col_index: int, 
            data = None):
            
        if data is None:
            # we return a scalar reading of the underlying shared memory
            success = self._shared_mem.read(self._numpy_view[row_index:row_index + 1, 
                col_index:col_index + 1], row_index, col_index)
            return self._numpy_view[row_index, 
                col_index].item(), success
        
        else:
            
            if isinstance(data, np.ndarray):  
                if not self._ensure2D(data):
                    message = "Provided data should be 2D!!"
                    Logger.log(self.__class__.__name__,
                        "write",
                        message,
                        LogType.EXCEP,
                        throw_when_excep = True)
                if not self._fits_into(data, self._torch_view, 
                                    row_index, col_index):
                    message = "Provided data does not fit in data view!!"
                    Logger.log(self.__class__.__name__,
                        "write",
                        message,
                        LogType.EXCEP,
                        throw_when_excep = True)
                
                input_rows, input_cols = data.shape

                # update block of numpy view from shared memory
                success = self._shared_mem.read(self._numpy_view[row_index:row_index + input_rows, 
                        col_index:col_index + input_cols], row_index, col_index)
                if not success:
                    return None, False
                
                # copy data into part of numpy view
                data[:, :] = self._numpy_view[row_index:row_index + input_rows, 
                        col_index:col_index + input_cols]
                
                return None, True
        
            # data is provided, but it's neither a ndarray nor a tensor nor None -> throw exception
            message = "Provided data has to be either " + \
            f"numpy.ndarray or None! Got {type(data)}"
            Logger.log(self.__class__.__name__,
                "write",
                message,
                LogType.EXCEP,
                throw_when_excep = True)
            return None, False

    def read_retry(self, 
            row_index: int, 
            col_index: int, 
            data = None):
        
        # tries reading until success
        read_done = False
        while self._shared_mem.isRunning() and not read_done: 
            data_read, read_done = self.read(
                                row_index=row_index,
                                col_index=col_index,
                                data=data)
            if read_done:
                return data_read, read_done
            else:
                continue

    def synch(self, 
        row_index: int, 
        col_index: int, 
        n_rows: int,
        n_cols: int,
        read: bool = True):

        # synchs a block of the internal views from shared memory
        # or vice-versa
        
        # Calculate ending indices
        end_row_index = row_index + n_rows
        end_col_index = col_index + n_cols

        # Check if we are withing bounds
        fits = (end_row_index <= self._numpy_view.shape[0] \
            and end_col_index <= self._numpy_view.shape[1])
        if not fits:
            return False
        
        if read:
            success = self._shared_mem.read(self._numpy_view[row_index:row_index + n_rows, 
                    col_index:col_index + n_cols], row_index, col_index)
            
            return success
        else:
            success = self._shared_mem.write(self._numpy_view[row_index:row_index + n_rows, 
                    col_index:col_index + n_cols], row_index, col_index)
            
            return success
    
    def synch_retry(self, 
        row_index: int, 
        col_index: int, 
        n_rows: int,
        n_cols: int,
        read: bool = True):

        # tries synching until success
        while not self.synch(row_index=row_index,
                        col_index=col_index,
                        n_rows=n_rows,
                        n_cols=n_cols,
                        read=read):
            continue
        
    def synch_all(self, 
            read: bool = True, 
            retry = False):
        
        # synch whole view from shared memory
        if retry:

            self.synch_retry(row_index=0, col_index=0, 
                        n_rows=self.n_rows, n_cols=self.n_cols, 
                        read=read)
            return True

        else:
            
            return self.synch(row_index=0, col_index=0, 
                        n_rows=self.n_rows, n_cols=self.n_cols, 
                        read=read)

    def gpu_mirror_exists(self):
        return self._gpu_mirror is not None
    
    def torch_view_exists(self):
        return self._with_torch_view
    
    def get_torch_mirror(self, 
                gpu: bool = False):
        
        if self._with_torch_view:

            if not gpu:
                return self._torch_view
        
            else:
                return self._gpu_mirror
            
        else:
            exception = f"Torch view is not available!"+\
                "Did you set the with_torch_view flag in the constructor?"
            Logger.log(self.__class__.__name__,
                "get_torch_view",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)
            
    def get_numpy_mirror(self):

        return self._numpy_view
    
    def synch_mirror(self,
        from_gpu: bool,
        non_blocking: bool = False):
        
        if self._gpu_mirror is None:
            exception = f"Cannot be called since no GPU mirror is available! " + \
                "Did you set the with_gpu_mirror in the contructor to True?"
            Logger.log(self.__class__.__name__,
                "synch_mirror",
                exception,
                LogType.EXCEP,
                throw_when_excep = True)

        # from torch docs
        # While cpu_tensor.to("cuda", non_blocking=True).mean() executes correctly, 
        # attempting cuda_tensor.to("cpu", non_blocking=True).mean() will result in erroneous outputs.
        # when copying from GPU to CPU non_blocking is therefore overridden to be always False
        if from_gpu:
            # synch cpu torch view from latest gpu mirror data (GPU -> CPU)
            self._torch_view[:, :] = self._gpu_mirror.cpu()
            self._torch_view[:, :] = self._gpu_mirror.to("cpu",non_blocking=False)

        else:
            # synch gpu torch data from torch view on cpu (CPU -> GPU)
            self._gpu_mirror[:, :] = self._torch_view.to('cuda',non_blocking=non_blocking)

        # torch.cuda.synchronize() # ensuring that all the streams on the GPU are completed \
        # before the CPU continues execution

    def data_sem_acquire(self,
                    timeout: float = None):

        if timeout is None:
            self._shared_mem.dataSemAcquire()
        else:
            self._shared_mem.dataSemAcquireDt(timeout)

    def data_sem_release(self):

        self._shared_mem.dataSemRelease()
    
    def to_zero(self):

        if self._gpu_mirror is not None:
            self._gpu_mirror.zero_() # reset gpu view
        self._numpy_view[:, :] = 0  # reset numpy view on CPU            
        self.synch_all(read=False, retry=True) # writes to shared mem

    def get_shared_mem(self): # return low-level sharedmem client/server
        return self._shared_mem
    
    def close(self):

        if self._shared_mem is not None:

            self._shared_mem.close()