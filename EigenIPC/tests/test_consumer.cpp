#include <iostream>
#include <EigenIPC/Consumer.hpp>
#include <string>
#include <EigenIPC/Client.hpp>
#include <EigenIPC/Journal.hpp>

#include <csignal>
#include <cstdlib>

using namespace EigenIPC;
using LogType = Journal::LogType;
using VLevel = Journal::VLevel;
using Consumer = EigenIPC::Consumer;

std::string name_space = "rthytrtkjdhsxdhn";

int counter = 0;
int listener_idx = 0;
bool terminated = false;
unsigned int timeout = 10000;

// Signal handler function
void interruptHandler(int signal) {
    std::cout << "Interrupt signal received (Ctrl+C pressed)." << std::endl;
    terminated = true;
}

int main(int argc, char *argv[]) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <namespace>" << std::endl;
        return 1;
        
    }
    // Get the command line argument for idx
    std::string name_space = argv[1];

    std::signal(SIGINT, interruptHandler);

    Consumer consumer = Consumer("ProducerConsumerTests", 
                            name_space,
                            true,
                            VLevel::V2);
    consumer.run();

    while(!terminated) {

        if (!consumer.wait(timeout)) {

            std::cout << "Wait failed" << std::endl;

            break;
        }

        std::cout << "Doing stuff..." << std::endl;

        if (!consumer.ack()) {

            std::cout << "Acknowledgement failed!!" << std::endl;

            break;
        }

    }

    consumer.close();

    return 0;
}

