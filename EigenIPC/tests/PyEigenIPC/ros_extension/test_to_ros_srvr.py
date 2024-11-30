from EigenIPC.PyEigenIPCExt.extensions.ros_bridge import *
from EigenIPC.PyEigenIPCExt.wrappers.shared_data_view import *

from EigenIPC.PyEigenIPC import *

import numpy as np

import time
from perf_sleep.pyperfsleep import PerfSleep

order = 'C'

import os

def set_affinity(cores):
    try:
        os.sched_setaffinity(0, cores)
        print(f"Set CPU affinity to cores: {cores}")
    except Exception as e:
        print(f"Error setting CPU affinity: {e}")

set_affinity([8])

server = SharedTWrapper(namespace = "Prova",
            basename = "ToRosTest",
            is_server = True, 
            n_rows = 100, 
            n_cols = 200, 
            verbose = True, 
            vlevel = VLevel.V3,
            dtype= dtype.Float,
            fill_value = np.nan,
            safe = True,
            force_reconnection = False)

server.run()

server.get_numpy_mirror()[:, :] = np.random.rand(server.n_rows, server.n_cols)

update_dt = 0.001
start_time = time.perf_counter() 
start_time = 0.0
elapsed_time = 0.0
actual_loop_dt = 0.0

time_to_sleep_ns = 0
debug = False

perf_timer = PerfSleep()

try:

    while True:

        start_time = time.perf_counter() 

        server.get_numpy_mirror()[:, :] = np.random.rand(server.n_rows, server.n_cols)

        server.synch_all(read=False, 
                    retry=True)
        
        elapsed_time = time.perf_counter() - start_time

        time_to_sleep_ns = int((update_dt - elapsed_time) * 1e+9) # [ns]

        if time_to_sleep_ns < 0:

            warning = f"Could not match desired update dt of {update_dt} s. " + \
                f"Elapsed time to update {elapsed_time}."

            Journal.log("test_to_ros_srvr.py",
                        "",
                        warning,
                        LogType.WARN,
                        throw_when_excep = True)

        perf_timer.thread_sleep(time_to_sleep_ns) 

        actual_loop_dt = time.perf_counter() - start_time

        if debug:

            print(f"Actual loop dt {actual_loop_dt} s.")

except KeyboardInterrupt:
    print("\nCtrl+C pressed. Exiting the loop.")


