
from EigenIPC.PyEigenIPC import *
import sys

# Get the total number of arguments
num_args = len(sys.argv)

# Check if there are enough arguments
if num_args < 2:
    print("Usage: python script.py [namespace] [n_consumers] ...")
    sys.exit(1)

# Access individual arguments
namespace = sys.argv[1]
n_consumers = int(sys.argv[2])

producer = Producer(namespace=namespace,
            basename="ProducerConsumerTests",
            verbose=True,
            vlevel=VLevel.V2,
            force_reconnection=False)

producer.run()

timeout = 10000
terminated=False
while not terminated:
    
    try:

        producer.trigger()

        print("Triggering...")

        if not producer.wait_ack_from(n_consumers, 
                            timeout):

            terminated = True
            
            break

    except KeyboardInterrupt:

        terminated = True
        
        break

producer.close()