// Copyright (C) 2023  Andrea Patrizi (AndrePatri)
// 
// This file is part of EigenIPC and distributed under the General Public License version 2 license.
// 
// EigenIPC is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
// 
// EigenIPC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with EigenIPC.  If not, see <http://www.gnu.org/licenses/>.
// 
#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <limits>
#include <Eigen/Dense>
#include <csignal>

#include <EigenIPC/Server.hpp>
#include <EigenIPC/StringTensor.hpp>
#include <EigenIPC/Helpers.hpp>
#include <EigenIPC/Journal.hpp>

#include <test_utils.hpp>

int N_ITERATIONS = 100000;
int N_ITERATIONS_STR = 100000;

int N_ROWS = 100;
int N_COLS = 60;
int STR_TENSOR_LENGTH = 1000;

using namespace EigenIPC;

using VLevel = Journal::VLevel;

static std::string name_space = "ConsistencyTests";

static Journal journal("ConsistencyTests");

// Define a structure to hold both the scalar type and the memory layout
template<typename T, int Layout>
struct TypeWithLayout {
    using type = T;
    static const int layout = Layout;
};

// List of types to test against
using MyTypes = ::testing::Types<
    TypeWithLayout<bool, ColMajor>,
    TypeWithLayout<bool, RowMajor>,
    TypeWithLayout<int, ColMajor>,
    TypeWithLayout<int, RowMajor>,
    TypeWithLayout<float, ColMajor>,
    TypeWithLayout<float, RowMajor>,
    TypeWithLayout<double, ColMajor>,
    TypeWithLayout<double, RowMajor>
>;

// normal tensor API
template <typename P>
class ConsistencyChecks : public ::testing::Test {
protected:

    using ScalarType = typename P::type;
    static const int layout = P::layout;

    ConsistencyChecks() : rows(N_ROWS),
                   cols(N_COLS),
                   iterations(N_ITERATIONS),
                   server_ping_ptr(new Server<ScalarType, layout>(rows, cols,
                                     "EigenIPC_ping",
                                     name_space + getTypeAsString<ScalarType>() + std::to_string(layout),
                                     false,
                                     VLevel::V3,
                                     true)),
                   server_pong_ptr(new Server<ScalarType, layout>(rows, cols,
                                     "EigenIPC_pong",
                                     name_space + getTypeAsString<ScalarType>() + std::to_string(layout),
                                     false,
                                     VLevel::V3,
                                     true)),
                   server_terminate_ptr(new Server<bool>(1, 1,
                                     "EigenIPC_terminate",
                                     name_space + getTypeAsString<ScalarType>() + std::to_string(layout),
                                     false,
                                     VLevel::V3,
                                     true)),
                   server_flag_ptr(new Server<bool>(1, 1,
                                     "EigenIPC_flag",
                                     name_space + getTypeAsString<ScalarType>() + std::to_string(layout),
                                     false,
                                     VLevel::V3,
                                     true)),
                   data_written(rows, cols),
                   data_read(rows, cols){

        // run all servers
        server_ping_ptr->run();
        server_pong_ptr->run();
        server_terminate_ptr->run();
        server_flag_ptr->run();

        terminate(0, 0) = false;
        server_terminate_ptr->write(terminate);

    }

    void Run() {

        for (int i = 0; i < this->iterations; ++i) {

            data_written.setRandom(); // randomize data

            // writes down the whole random matrix on ping memory
            while (! server_ping_ptr->write(data_written, 0, 0)) {

                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }

            flag(0, 0) = true;
            while(!server_flag_ptr->write(flag)) {
                // try again
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
             // signals the client the write operation was completed

            while (flag(0, 0)){

                server_flag_ptr->read(flag); // continue reading the flag

                std::this_thread::sleep_for(std::chrono::microseconds(1)); // avoid busy wait loop

            }
            // client sets flag to false -> it has read the data and copied to the pong data

            // reads data from pong memory
            while(!server_pong_ptr->read(data_read, 0, 0)) {


                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }

            checks.push_back(areEqual<ScalarType>(data_written, data_read));

        }

        terminate(0, 0) = true;
        while(!server_terminate_ptr->write(terminate)) {

            // try again
            std::this_thread::sleep_for(std::chrono::microseconds(1));

        }

    }

    void SetUp() override {

        // Initialization code (if needed)
    }

    void TearDown() override {

        // close all servers
        server_ping_ptr->close();
        server_pong_ptr->close();
        server_terminate_ptr->close();
        server_flag_ptr->close();

    }

    int rows;
    int cols;
    int iterations;
    typename Server<ScalarType, layout>::UniquePtr server_ping_ptr,
                                                   server_pong_ptr;


    typename Server<bool>::UniquePtr server_terminate_ptr,
                                server_flag_ptr;

    Tensor<ScalarType, layout> data_read, data_written;

    Tensor<bool> terminate = Tensor<bool>::Zero(1, 1); // flags to false
    Tensor<bool> flag = Tensor<bool>::Zero(1, 1);

    std::vector<bool> checks;

};

// view API (this server will use the view API for reading
// and writing, the client side will use the normal tensor API)
template <typename P>
class ViewsConsistencyChecks : public ::testing::Test {
protected:

    using ScalarType = typename P::type;
    static const int layout = P::layout;

    ViewsConsistencyChecks() : rows(N_ROWS),
                   cols(N_COLS),
                   iterations(N_ITERATIONS),
                   server_ping_ptr(new Server<ScalarType, layout>(rows, cols,
                                     "EigenIPC_view_ping",
                                     name_space + getTypeAsString<ScalarType>() + std::to_string(layout),
                                     false,
                                     VLevel::V3,
                                     true)),
                   server_pong_ptr(new Server<ScalarType, layout>(rows, cols,
                                     "EigenIPC_view_pong",
                                     name_space + getTypeAsString<ScalarType>() + std::to_string(layout),
                                     false,
                                     VLevel::V3,
                                     true)),
                   server_terminate_ptr(new Server<bool>(1, 1,
                                     "EigenIPC_view_terminate",
                                     name_space + getTypeAsString<ScalarType>() + std::to_string(layout),
                                     false,
                                     VLevel::V3,
                                     true)),
                   server_flag_ptr(new Server<bool>(1, 1,
                                     "EigenIPC_view_flag",
                                     name_space + getTypeAsString<ScalarType>() + std::to_string(layout),
                                     false,
                                     VLevel::V3,
                                     true)),
                   data_written(rows, cols),
                   data_read(rows, cols),
                   data_written_view(helpers::createViewFrom<ScalarType, layout>(
                                         data_written,
                                         0, 0, // indeces
                                         rows, cols)),
                   data_read_view(helpers::createViewFrom<ScalarType, layout>(
                                      data_read,
                                      0, 0, // indeces
                                      rows, cols)){

        // run all servers
        server_ping_ptr->run();
        server_pong_ptr->run();
        server_terminate_ptr->run();
        server_flag_ptr->run();

        terminate(0, 0) = false;
        server_terminate_ptr->write(terminate);

    }

    void Run() {

        for (int i = 0; i < this->iterations; ++i) {

            // here we write and read using views, to test the view API
            // is working properly. The client side will use the normal
            // tensor API to write and read

            data_written.setRandom(); // randomize data

            // writes down the whole random matrix on ping memory (view)
            while (! server_ping_ptr->write(data_written_view, 0, 0)) {

                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }

            flag(0, 0) = true;
            while(!server_flag_ptr->write(flag)) {
                // try again
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
             // signals the client the write operation was completed

            while (flag(0, 0)){

                server_flag_ptr->read(flag); // continue reading the flag

                std::this_thread::sleep_for(std::chrono::microseconds(1)); // avoid busy wait loop

            }
            // client sets flag to false -> it has read the data and copied to the pong data

            // reads data from pong memory (view)
            while(!server_pong_ptr->read(data_read_view, 0, 0)) {


                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }

            // being the views updated, also the original tensors should be
            checks.push_back(areEqual<ScalarType>(data_written, data_read));

        }

        terminate(0, 0) = true;
        while(!server_terminate_ptr->write(terminate)) {

            // try again
            std::this_thread::sleep_for(std::chrono::microseconds(1));

        }

    }

    void SetUp() override {

        // Initialization code (if needed)
    }

    void TearDown() override {

        // close all servers
        server_ping_ptr->close();
        server_pong_ptr->close();
        server_terminate_ptr->close();
        server_flag_ptr->close();

    }

    int rows;
    int cols;
    int iterations;
    typename Server<ScalarType, layout>::UniquePtr server_ping_ptr,
                                                   server_pong_ptr;


    typename Server<bool>::UniquePtr server_terminate_ptr,
                                server_flag_ptr;

    Tensor<ScalarType, layout> data_read, data_written;
    TensorView<ScalarType, layout> data_read_view, data_written_view;

    Tensor<bool> terminate = Tensor<bool>::Zero(1, 1); // flags to false
    Tensor<bool> flag = Tensor<bool>::Zero(1, 1);

    std::vector<bool> checks;

};

class StringTensorCheck : public ::testing::Test {
protected:

    StringTensorCheck() :

                    string_t_ping_ptr(new StringTensor<StrServer>(
                                     STR_TENSOR_LENGTH,
                                     "SharedStrTensor_ping",
                                     name_space,
                                     false,
                                     VLevel::V3,
                                     true)),
                    string_t_pong_ptr(new StringTensor<StrServer>(
                                     STR_TENSOR_LENGTH,
                                     "SharedStrTensor_pong",
                                     name_space,
                                     false,
                                     VLevel::V3,
                                     true)),
                    server_terminate_ptr(new Server<bool>(1, 1,
                                   "SharedStrTensor_terminate",
                                   name_space,
                                   false,
                                   VLevel::V3,
                                   true)),
                    server_flag_ptr(new Server<bool>(1, 1,
                                   "SharedStrTensor_flag",
                                   name_space,
                                   false,
                                   VLevel::V3,
                                   true)),
                    length(STR_TENSOR_LENGTH),
                    iterations(N_ITERATIONS_STR)
    {

        data_read.resize(length);
        data_written.resize(length);

        string_t_ping_ptr->run();
        string_t_pong_ptr->run();
        server_terminate_ptr->run();
        server_flag_ptr->run();

        terminate(0, 0) = false;
        server_terminate_ptr->write(terminate);

    }

    std::string collapseStrVec(const std::vector<std::string>& input) {

        std::string delimiter = ", ";

        return std::accumulate(input.begin(), input.end(),
                             std::string(),
                             [&delimiter](const std::string& a, const std::string& b) {
                                 return a.empty() ? b : a + delimiter + b;
                             });

    }

    void SetUp() override {

    }

    void TearDown() override {

        string_t_ping_ptr->close();
        string_t_pong_ptr->close();
        server_terminate_ptr->close();
        server_flag_ptr->close();

    }

    void Randomize() {

        for (int i = 0; i < length; ++i) {

            data_written[i] = random_string(13); // randomize vector

        }
    }

    void Run() {

        for (int i = 0; i < this->iterations; ++i) {

            Randomize(); // randomize data

            // writes down the whole random matrix on ping memory
            while (!string_t_ping_ptr->write(data_written, 0)) {

                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }

//            std::string written = collapseStrVec(data_written);
//            std::cout << "Written: " << written << std::endl;

            flag(0, 0) = true;
            while(!server_flag_ptr->write(flag)) {
                // try again
                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }
             // signals the client the write operation was completed

            while (flag(0, 0)){

                server_flag_ptr->read(flag); // continue reading the flag

                std::this_thread::sleep_for(std::chrono::microseconds(1)); // avoid busy wait loop

            }
            // client sets flag to false -> it has read the data and copied to the pong data

            // reads data from pong memory
            while(!string_t_pong_ptr->read(data_read, 0)) {

                std::this_thread::sleep_for(std::chrono::microseconds(1));
            }

//            std::string read = collapseStrVec(data_read);
//            std::cout << "Read: " << read << std::endl;

            checks.push_back(data_read == data_written);

        }

        terminate(0, 0) = true;
        while(!server_terminate_ptr->write(terminate)) {

            std::string read = collapseStrVec(data_written);

            std::cout << "Read: " << read <<std::endl;

            // try again
            std::this_thread::sleep_for(std::chrono::microseconds(1));

        }

    }

    int length;
    int iterations;
    typename StringTensor<StrServer>::UniquePtr string_t_ping_ptr,
                                                string_t_pong_ptr;


    typename Server<bool>::UniquePtr server_terminate_ptr,
                                server_flag_ptr;

    std::vector<std::string> data_read, data_written;

    Tensor<bool> terminate = Tensor<bool>::Zero(1, 1); // flags to false
    Tensor<bool> flag = Tensor<bool>::Zero(1, 1);

    std::vector<bool> checks;

};

// Register the tests for normal tensor API
TYPED_TEST_SUITE_P(ConsistencyChecks);

TYPED_TEST_P(ConsistencyChecks, CheckConsistencyDTypes) {

    check_comp_type(journal);

    this->Run(); // run round-trip checks with client

    int n_failures = 0;

    bool success = allTrue(this->checks, n_failures);

    std::cout << "Test completed. Number of failures: " << n_failures <<
              "/" << N_ITERATIONS << std::endl;

    ASSERT_TRUE(success);

}

REGISTER_TYPED_TEST_SUITE_P(ConsistencyChecks, CheckConsistencyDTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(ConsistencyCheck, ConsistencyChecks, MyTypes);

// Register the tests for view API
TYPED_TEST_SUITE_P(ViewsConsistencyChecks);

TYPED_TEST_P(ViewsConsistencyChecks, CheckConsistencyViewDTypes) {

    check_comp_type(journal);

    this->Run(); // run round-trip checks with client

    int n_failures = 0;

    bool success = allTrue(this->checks, n_failures);

    std::cout << "Test completed. Number of failures: " << n_failures <<
              "/" << N_ITERATIONS << std::endl;

    ASSERT_TRUE(success);

}

REGISTER_TYPED_TEST_SUITE_P(ViewsConsistencyChecks, CheckConsistencyViewDTypes);

INSTANTIATE_TYPED_TEST_SUITE_P(ConsistencyViewCheck, ViewsConsistencyChecks, MyTypes);

// StringTensor checks
TEST_F(StringTensorCheck, StringTensorChecks) {

    check_comp_type(journal);

    this->Run();

    int n_failures = 0;

    bool success = allTrue(this->checks, n_failures);

    std::cout << "Test completed. Number of failures: " << n_failures <<
              "/" << N_ITERATIONS_STR << std::endl;

    ASSERT_TRUE(success);

}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
