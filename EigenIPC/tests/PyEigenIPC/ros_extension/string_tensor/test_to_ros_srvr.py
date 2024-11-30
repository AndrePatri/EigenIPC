from EigenIPC.PySharsor.extensions.ros_bridge import *
from EigenIPC.PySharsor.wrappers.shared_data_view import *

from EigenIPC.PyEigenIPC import *

import numpy as np

import time
from perf_sleep.pyperfsleep import PerfSleep

order = 'C'

import os

import random
import string

def generate_random_string_list(n):
    # Set the length of each string
    string_length = 10

    # Generate a list of n random strings
    random_string_list = [''.join(random.choices(string.ascii_letters + string.digits, k=string_length)) for _ in range(n)]

    return random_string_list

def set_affinity(cores):
    try:
        os.sched_setaffinity(0, cores)
        print(f"Set CPU affinity to cores: {cores}")
    except Exception as e:
        print(f"Error setting CPU affinity: {e}")

set_affinity([8])

list_string_prova = ["pippo", "sbarufug", "sadcsd9c9)8//&$", "dscv=?ç°DS)(CUDB/())"]

string_list_size = 10
server = StringTensorServer(length=string_list_size,
            name_space = "StringProva",
            basename = "ToRosStringTest",
            verbose = True, 
            vlevel = VLevel.V3)

server.run()

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

        random_stringlist = generate_random_string_list(string_list_size)

        print("Writing ")
        print(random_stringlist)

        jnt_names_written = server.write_vec(random_stringlist, 0)
        
        if not jnt_names_written:
            
            Journal.log("test_to_ros_srvr.py",
                        "",
                        "could not write list of strings to shared mem",
                        LogType.EXCEP,
                        throw_when_excep = False)

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


