#include <iostream>
#include <EigenIPC/Producer.hpp>
#include <string>
#include <EigenIPC/Server.hpp>
#include <EigenIPC/Journal.hpp>

#include <csignal>
#include <cstdlib>

using namespace EigenIPC;
using LogType = Journal::LogType;
using VLevel = Journal::VLevel;
using Producer = EigenIPC::Producer;

bool terminated = false;
unsigned int timeout = 10000;

// Signal handler function
void interruptHandler(int signal) {
    std::cout << "Interrupt signal received (Ctrl+C pressed)." << std::endl;
    terminated = true;
}

int main(int argc, char *argv[]) {

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <namespace, n_consumers>" << std::endl;
        return 1;
        
    }
    
    std::string name_space = argv[1];

    std::string n_consumers_arg = argv[2];
    int n_consumers = std::stoi(n_consumers_arg);

    std::signal(SIGINT, interruptHandler);

    Producer producer = Producer("ProducerConsumerTests", 
                            name_space,
                            true,
                            VLevel::V2,
                            false);

    producer.run();
    
    while(!terminated) {

        producer.trigger();

        std::cout << "Triggering..." << std::endl;

        if (!producer.wait_ack_from(n_consumers, timeout)) {
            
            std::cout << "Wait failed" << std::endl;

            break;
        }

    }

    producer.close();

    return 0;
}

