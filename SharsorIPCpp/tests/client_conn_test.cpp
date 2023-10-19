#include <gtest/gtest.h>
#include <chrono>
#include <vector>
#include <limits>
#include <Eigen/Dense>
#include <csignal>
#include <thread>
#include <chrono>

#include <SharsorIPCpp/Server.hpp>
#include <SharsorIPCpp/Client.hpp>

#include <SharsorIPCpp/Journal.hpp>

#include <test_utils.hpp>

int N_ITER = 10;
int BLOCK_SIZE = 3;

using namespace SharsorIPCpp;

using VLevel = Journal::VLevel;

static std::string name_space = "ConnectionTests";

static Journal journal("ConnectionTestsP2");

class ClientReadsInt : public ::testing::Test {
protected:

    ClientReadsInt() :
                   client_ptr(new Client<int>(
                                     "SharsorInt", name_space,
                                     true,
                                     VLevel::V3)) {

        client_ptr->attach();

        getMetaData();

        tensor_copy = Tensor<int>::Zero(rows,
                                         cols);

        tensor_block_copy = Tensor<int>::Zero(BLOCK_SIZE,
                                         BLOCK_SIZE);

        client_ptr->readTensor(tensor_copy);

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

        rows = client_ptr->n_rows;
        cols = client_ptr->n_cols;
    }

    void readData() {

        client_ptr->readTensor(tensor_copy);
        client_ptr->readTensor(tensor_block_copy, 0, 0);

        std::cout << "Read tensor (copy):" << std::endl;
        std::cout << tensor_copy << std::endl;
        std::cout << "Read tensor block (copy):" << std::endl;
        std::cout << tensor_block_copy << std::endl;
        std::cout << "##############" << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(1));

    }

    int rows = -1;
    int cols = -1;
    int dtype = -1;

    Client<int>::UniquePtr client_ptr;

    Tensor<int> tensor_copy;
    Tensor<int> tensor_block_copy;
};

class ClientReadsBool : public ::testing::Test {
protected:

    ClientReadsBool() :
                   client_ptr(new Client<bool>(
                                     "SharsorBool", name_space,
                                     true,
                                     VLevel::V3)) {

        client_ptr->attach();

        getMetaData();

        tensor_copy = Tensor<bool>::Zero(rows,
                                         cols);

        tensor_block_copy = Tensor<bool>::Zero(rows - 2,
                                         cols - 2);

        client_ptr->readTensor(tensor_copy);

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

        rows = client_ptr->n_rows;
        cols = client_ptr->n_cols;
    }

    void readData() {

        client_ptr->readTensor(tensor_copy);
        client_ptr->readTensor(tensor_block_copy,
                               1, 1);
        std::cout << "Read tensor (copy):" << std::endl;
        std::cout << tensor_copy << std::endl;
        std::cout << "Read tensor block (copy):" << std::endl;
        std::cout << tensor_block_copy << std::endl;
        std::cout << "##############" << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(1));

    }

    int rows = -1;
    int cols = -1;
    int dtype = -1;

    Client<bool>::UniquePtr client_ptr;

    Tensor<bool> tensor_copy;
    Tensor<bool> tensor_block_copy;

};

class ClientReadsFloat : public ::testing::Test {
protected:

    ClientReadsFloat() :
                   client_ptr(new Client<float>(
                                     "SharsorFloat", name_space,
                                     true,
                                     VLevel::V3)) {

        client_ptr->attach();

        getMetaData();

        tensor_copy = Tensor<float>::Zero(rows,
                                         cols);
        tensor_block_copy = Tensor<float>::Zero(rows - 2,
                                                cols - 2);

        client_ptr->readTensor(tensor_copy);

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

        rows = client_ptr->n_rows;
        cols = client_ptr->n_cols;
    }

    void readData() {

        client_ptr->readTensor(tensor_copy);
        client_ptr->readTensor(tensor_block_copy,
                               1, 1);

        std::cout << "Read tensor (copy):" << std::endl;
        std::cout << tensor_copy << std::endl;
        std::cout << "Read tensor block (copy):" << std::endl;
        std::cout << tensor_block_copy << std::endl;
        std::cout << "##############" << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(1));

    }

    int rows = -1;
    int cols = -1;
    int dtype = -1;

    Client<float>::UniquePtr client_ptr;

    Tensor<float> tensor_copy;
    Tensor<float> tensor_block_copy;

};

class ClientReadsDouble : public ::testing::Test {
protected:

    ClientReadsDouble() :
                   client_ptr(new Client<double>(
                                     "SharsorDouble", name_space,
                                     true,
                                     VLevel::V3)) {

        client_ptr->attach();

        getMetaData();

        tensor_copy = Tensor<double>::Zero(rows,
                                         cols);

        tensor_block_copy = Tensor<double>::Zero(rows - 2,
                                                cols - 2);

        client_ptr->readTensor(tensor_copy);

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

        rows = client_ptr->n_rows;
        cols = client_ptr->n_cols;
    }

    void readData() {

        client_ptr->readTensor(tensor_copy);
        client_ptr->readTensor(tensor_block_copy,
                               1, 1);

        std::cout << "Read tensor (copy):" << std::endl;
        std::cout << tensor_copy << std::endl;
        std::cout << "Read tensor block (copy):" << std::endl;
        std::cout << tensor_block_copy << std::endl;
        std::cout << "##############" << std::endl;

        std::this_thread::sleep_for(std::chrono::seconds(1));

    }

    int rows = -1;
    int cols = -1;
    int dtype = -1;

    Client<double>::UniquePtr client_ptr;

    Tensor<double> tensor_copy;
    Tensor<double> tensor_block_copy;

};

TEST_F(ClientReadsInt, ClientReadingInt) {

    check_comp_type(journal);

    journal.log("ClientReadsInt", "\n Starting to read ...\n",
                Journal::LogType::STAT);

    for (int i = 0; i < N_ITER; ++i) {

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

TEST_F(ClientReadsFloat, ClientReadRandFloat) {

    check_comp_type(journal);

    journal.log("ClientReadsFloat", "\n Starting to read ...\n",
                Journal::LogType::STAT);

    for (int i = 0; i < N_ITER; ++i) {

        readData();
    }
}

int main(int argc, char** argv) {

//    ::testing::GTEST_FLAG(filter) =
//        ":ClientReadsInt.ClientReadingInt";

//    ::testing::GTEST_FLAG(filter) +=
//        ":ClientReadsBool.ClientReadsRandBoolBlock";

//    ::testing::GTEST_FLAG(filter) +=
//        ":ClientReadsFloat.ClientReadRandFloat";

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
