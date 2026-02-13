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
#include <thread>
#include <chrono>
#include <numeric>
#include <vector>
#include <string>

#include <EigenIPC/Server.hpp>
#include <EigenIPC/Client.hpp>
#include <EigenIPC/StringTensor.hpp>

#include <EigenIPC/Helpers.hpp>
#include <EigenIPC/Journal.hpp>

#include <test_utils.hpp>

int N_ITER = 100;
int N_ITER_STR = 20;

int BLOCK_SIZE = 3;

using namespace EigenIPC;

using VLevel = Journal::VLevel;

static std::string name_space = "ConnectionTests";

static Journal journal("ConnectionTestsP2");

class ClientReadsBool : public ::testing::Test {
protected:

    ClientReadsBool() :
                   client_ptr(new Client<bool, EigenIPC::MemLayoutDefault>(
                                     "SharedMemBool", name_space,
                                     true,
                                     VLevel::V3)) {

        client_ptr->attach();

        getMetaData();

        tensor_copy = Tensor<bool, EigenIPC::MemLayoutDefault>::Zero(rows,
                                         cols);

        tensor_block_copy = Tensor<bool, EigenIPC::MemLayoutDefault>::Zero(rows - 2,
                                         cols - 2);

        std::cout << "Detected data of size " << rows << "x" << cols << std::endl;
    }

    void SetUp() override {

        // Initialization code (if needed)

    }

    void TearDown() override {

        client_ptr->close();

        // Cleanup code (if needed)

    }

    void getMetaData() {

        rows = client_ptr->getNRows();
        cols = client_ptr->getNCols();
    }

    void readData() {

        client_ptr->read(tensor_block_copy,
                               1, 1);

        client_ptr->read(tensor_copy.block(1, 1,
                                                 rows - 2, cols - 2),
                               1, 1); // only update block

        std::cout << "Read tensor (copy):" << std::endl;
        std::cout << tensor_copy << std::endl;
        std::cout << "Read tensor block (copy):" << std::endl;
        std::cout << tensor_block_copy << std::endl;
        std::cout << "##############" << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(100));

    }

    int rows = -1;
    int cols = -1;
    int dtype = -1;

    Client<bool, EigenIPC::MemLayoutDefault>::UniquePtr client_ptr;

    Tensor<bool, EigenIPC::MemLayoutDefault> tensor_copy;
    Tensor<bool, EigenIPC::MemLayoutDefault> tensor_block_copy;

};

class StringTensorRead : public ::testing::Test {
protected:

    StringTensorRead() :
                   string_t_ptr(new StringTensor<StrClient>(
                                     "SharedStrTensor", name_space,
                                     true,
                                     VLevel::V3)) {

        string_t_ptr->run();

        str_vec.resize(string_t_ptr->getLength());

    }

    void SetUp() override {

    }

    void TearDown() override {

        string_t_ptr->close();

    }

    void readData() {

        string_t_ptr->read(str_vec, 0);

        std::string delimiter = ", ";

        std::string result = std::accumulate(str_vec.begin(), str_vec.end(),
                                             std::string(),
                                             [&delimiter](const std::string& a, const std::string& b) {
                                                 return a.empty() ? b : a + delimiter + b;
                                             });
        journal.log("StringTensorRead",
                    std::string("\n") + result + std::string("\n"),
                    Journal::LogType::STAT);

        std::this_thread::sleep_for(std::chrono::milliseconds(500));

    }

    StringTensor<StrClient>::UniquePtr string_t_ptr;
    std::vector<std::string> str_vec;

};

TEST_F(StringTensorRead, StringTensorCheck) {

    check_comp_type(journal);

    journal.log("StringTensorWrite", "\n Starting to read string tensor ...\n",
                Journal::LogType::STAT);

    for (int i = 0; i < N_ITER_STR; ++i) {

        readData();
    }

}

TEST_F(ClientReadsBool, ClientReadsRandBoolBlock) {

    check_comp_type(journal);

    journal.log("ClientReadsBool", "\n Starting to read ...\n",
                Journal::LogType::STAT);

    for (int i = 0; i < N_ITER; ++i) {

        readData();
    }
}

int main(int argc, char** argv) {

//    ::testing::GTEST_FLAG(filter) =
//        ":ClientReadsInt.ClientReadingInt";

    ::testing::GTEST_FLAG(filter) =
        ":ClientReadsBool.ClientReadsRandBoolBlock";

//    ::testing::GTEST_FLAG(filter) =
//        ":ClientReadsFloat.ClientReadRandFloat";

//    ::testing::GTEST_FLAG(filter) =
//        ":StringTensorRead.StringTensorCheck";

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
