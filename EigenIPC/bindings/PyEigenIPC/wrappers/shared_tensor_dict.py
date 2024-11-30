from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import SharedTWrapper
from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import dtype as eigenipc_dtype, toNumpyDType

from EigenIPC.PyEigenIPC import StringTensorServer, StringTensorClient

from EigenIPC.PyEigenIPC import Journal, LogType

import numpy as np

from typing import List

class SharedTensorDict():

    # A basic implementation of a shared dictionary. 
    # key -> numpy.ndarray

    class Names():
        
        def __init__(self,
            names: List[str] = None,
            namespace = "",
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            force_reconnection: bool = False):

            basename = "debug_data_names"
            
            self.is_server = is_server

            self.names = names

            if self.is_server:

                self.shared_names = StringTensorServer(length = len(names), 
                                            basename = basename, 
                                            name_space = namespace,
                                            verbose = verbose, 
                                            vlevel = vlevel, 
                                            force_reconnection = force_reconnection)

            else:

                self.shared_names = StringTensorClient(
                                            basename = basename, 
                                            name_space = namespace,
                                            verbose = verbose, 
                                            vlevel = vlevel)
                
        def run(self):
            
            self.shared_names.run()
            
            if self.is_server:
                
                jnt_names_written = self.shared_names.write_vec(self.names, 0)

                if not jnt_names_written:

                    raise Exception("Could not write joint names on shared memory!")
            
            else:
                
                self.names = [""] * self.shared_names.length()

                while not self.shared_names.read_vec(self.names, 0):

                    Journal.log(self.__class__.__name__,
                            "run",
                            "Could not read names on shared memory. Retrying...",
                            LogType.WARN,
                            throw_when_excep = True)

        def close(self):

            self.shared_names.close()
            
    class DataDims(SharedTWrapper):
        
        def __init__(self,
            dims: List[int] = None,
            namespace = "",
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False):
        
            basename = "debug_data_dims"

            self.dims = dims

            if is_server:
                
                super().__init__(namespace = namespace,
                    basename = basename,
                    is_server = is_server, 
                    n_rows = len(self.dims), 
                    n_cols = 1, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    dtype=eigenipc_dtype.Int,
                    fill_value=-1,
                    safe = safe,
                    force_reconnection = force_reconnection)

            else:

                super().__init__(namespace = namespace,
                    basename = basename,
                    is_server = is_server, 
                    verbose = verbose, 
                    vlevel = vlevel,
                    dtype=eigenipc_dtype.Int,
                    fill_value=-1,
                    safe = safe)
                
        def run(self):

            super().run()

            if self.is_server:

                dims = np.array(self.dims, dtype=toNumpyDType(self.get_shared_mem().getScalarType())).reshape((len(self.dims), 1))

                self.write_retry(data = dims, row_index= 0,
                        col_index=0)
            
            else:

                # updates shared dims
                self.synch_all(read=True, retry=True)
                
                self.dims = self.get_numpy_mirror()[:, :].copy()

                self.n_dims = self.n_rows
                self.n_nodes = self.n_cols
                
    class Data(SharedTWrapper):
        
        def __init__(self,
            namespace = "",
            is_server = False, 
            n_dims: int = -1, 
            n_nodes: int = -1, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False):
        
            basename = "debug_data" 

            super().__init__(namespace = namespace,
                basename = basename,
                is_server = is_server, 
                n_rows = n_dims, 
                n_cols = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel,
                fill_value=np.nan,
                safe = safe,
                force_reconnection = force_reconnection)

    def __init__(self,
            names: List[str] = None, # not needed if client
            dimensions: List[int] = None, # not needed if client
            n_nodes: int = -1, # not needed if client 
            namespace = "",
            is_server = False, 
            verbose: bool = False, 
            vlevel: VLevel = VLevel.V0,
            safe: bool = True,
            force_reconnection: bool = False):
        
        self.names = names
        self.dimensions = dimensions

        self.n_dims = None
        self.n_nodes = n_nodes

        self.is_server = is_server

        if self.is_server:
            
            n_dims = 0

            for i in range(0, len(dimensions)):

                n_dims = n_dims + dimensions[i]

            self.n_dims = n_dims

        # actual data
        self.data = self.Data(namespace = namespace,
                is_server = is_server, 
                n_dims= self.n_dims, 
                n_nodes = n_nodes, 
                verbose = verbose, 
                vlevel = vlevel,
                safe = safe,
                force_reconnection = force_reconnection)
        
        # names of each block of data
        self.shared_names = self.Names(namespace = namespace,
                is_server = is_server, 
                names = self.names,
                verbose = verbose, 
                vlevel = vlevel,
                force_reconnection = force_reconnection)

        # dimenions of each block of data
        self.shared_dims = self.DataDims(namespace = namespace,
                is_server = is_server, 
                dims = dimensions,
                verbose = verbose, 
                vlevel = vlevel,
                safe = safe,
                force_reconnection = force_reconnection)

    def run(self):
        
        self.data.run()

        self.shared_names.run()

        self.shared_dims.run()

        if not self.is_server:

            self.names = self.shared_names.names
            
            # updates shared dims
            self.shared_dims.synch_all(read=True, retry=True)

            self.dimensions = self.shared_dims.dims.flatten().tolist()

            self.n_dims = self.data.n_rows
            self.n_nodes = self.data.n_cols

    def write(self,
        data: np.ndarray,
        name: str,
        retry = True):

        # we assume names does not contain
        # duplicates
        
        data_idx = self.names.index(name)

        # we sum dimensions up until the data we 
        # need to write to get the starting index
        # of the data block
        starting_idx = 0
        for index in range(data_idx):

            starting_idx += self.dimensions[index]

        data_2D = np.atleast_2d(data)

        if retry: 

            self.data.write_retry(np.atleast_2d(data_2D), starting_idx, 0) # blocking
            
            return True
        
        else:

            return self.data.write(np.atleast_2d(data_2D), starting_idx, 0) # non-blocking

    def synch(self,
            retry = True):

        # to be called before using get() on one or more data 
        # blocks

        # updates the whole view with shared data
        return self.data.synch_all(read = True, retry = retry)

    def get(self,
        name: str):

        # we assume names does not contain
        # duplicates
        data_idx = self.names.index(name)

        # we sum dimensions up until the data we 
        # need to write to get the starting index
        # of the data block
        starting_idx = 0
        for index in range(data_idx):
            starting_idx += self.dimensions[index]

        view = self.data.get_numpy_mirror()[starting_idx:starting_idx + self.dimensions[data_idx], :]

        view_copy = view.copy()

        return view_copy
    
    def close(self):

        self.data.close()

        self.shared_names.close()

        self.shared_dims.close()
      