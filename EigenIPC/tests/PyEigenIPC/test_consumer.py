
from EigenIPC.PyEigenIPC import *
import sys

# Get the total number of arguments
num_args = len(sys.argv)

# Check if there are enough arguments
if num_args < 1:
    print("Usage: python script.py [namespace]...")
    sys.exit(1)

# Access individual arguments
namespace = sys.argv[1]

consumer = Consumer(namespace=namespace,
            basename="ProducerConsumerTests",
            verbose=True,
            vlevel=VLevel.V2)

consumer.run()

timeout = 10000
terminated=False
while not terminated:
    
    try:

        if not consumer.wait(timeout):

            terminated = True
            
            break

        print("Dong stuff...")

        if not consumer.ack():

            terminated = True
            
            break

    except KeyboardInterrupt:

        terminated = True
        
        break

consumer.close()