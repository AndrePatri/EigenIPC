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
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <EigenIPC/Client.hpp>
#include <EigenIPC/Server.hpp>

#include <PyEigenIPC/PyStringTensor.hpp>

namespace py = pybind11;
using namespace EigenIPC;

void PyEigenIPC::PyStringTensor::declare_StringTensorServer(py::module &m) {

    std::string pyclass_name = std::string("StringTensorServer");

    py::class_<StringTensor<StrServer>>(m, pyclass_name.c_str())

        .def(py::init<int, std::string, std::string, bool, VLevel, bool, bool>(),
             py::arg("length"),
             py::arg("basename") = "MySharedMemory",
             py::arg("name_space") = "",
             py::arg("verbose") = false,
             py::arg("vlevel") = VLevel::V0,
             py::arg("force_reconnection") = false,
             py::arg("safe") = true)

        .def("run", &StringTensor<StrServer>::run)

        // Add other method bindings here ...
        .def("isRunning", &StringTensor<StrServer>::isRunning)

        .def("length", &StringTensor<StrServer>::getLength)

        .def("getNamespace", &StringTensor<StrServer>::getNamespace)

        .def("getBasename", &StringTensor<StrServer>::getBasename)

        .def("close", &StringTensor<StrServer>::close)

        .def("write_vec",
            (bool (StringTensor<StrServer>::*)(const std::vector<std::string>&, int))
            &StringTensor<StrServer>::write,
            py::arg("str_list"), py::arg("index") = 0)

        .def("read_vec", [](EigenIPC::StringTensor<EigenIPC::StrServer>& self, 
                    py::list py_list, 
                    int index) {
            
            // Check if the Python list is empty
            if (py_list.empty()) {
                return false;  // Return false if the list is empty
            }

            // Convert py::list to std::vector<std::string>
            std::vector<std::string> result_vec;
            for (auto item : py_list) {
                result_vec.push_back(item.cast<std::string>());
            }

            // Call the C++ method
            bool success = self.read(result_vec, index);

            // Clear the original Python list and refill with updated values
            py_list.attr("clear")();
            for (auto& str : result_vec) {
                py_list.append(str);
            }

            return success;

        }, py::arg("str_list"), py::arg("index") = 0)

        .def("write_str",
            (bool (StringTensor<StrServer>::*)(const std::string&, int))
            &StringTensor<StrServer>::write,
            py::arg("to_write"), py::arg("index") = 0)

        .def("read_str", [](EigenIPC::StringTensor<EigenIPC::StrClient>& self, int index) {
            std::string result_str;
            bool success = self.read(result_str, index);

            // Return a tuple containing the string and the success flag
            return std::make_tuple(result_str, success);
            
        }, py::arg("index") = 0)

        .def("get_raw_buffer", [](EigenIPC::StringTensor<EigenIPC::StrServer>& self) {
            
            EigenIPC::Tensor<int> buffer = self.get_raw_buffer();

            // Get the shape of the buffer tensor
            auto shape = py::array_t<int>({buffer.rows(), buffer.cols()});

            // initialize a pybind array of the right shape
            py::array_t<int> buffer_copy(shape);

            auto result_ptr = buffer_copy.mutable_data();

            // manually copy data from the buffer to the buffer copy
            std::memcpy(result_ptr, buffer.data(), buffer.size() * sizeof(int));

            return buffer_copy;

        })

        .def("get_shared_mem",
            [](EigenIPC::StringTensor<EigenIPC::StrServer>& self) -> EigenIPC::StrServer& {
                return self.getSharedMem();
            },
            py::return_value_policy::reference_internal)

        ;

}

void PyEigenIPC::PyStringTensor::declare_StringTensorClient(py::module &m) {

    std::string pyclass_name = std::string("StringTensorClient");

    py::class_<StringTensor<StrClient>>(m, pyclass_name.c_str())

        .def(py::init<std::string, std::string, bool, VLevel, bool>(),
             py::arg("basename") = "MySharedMemory",
             py::arg("name_space") = "",
             py::arg("verbose") = false,
             py::arg("vlevel") = VLevel::V0,
             py::arg("safe") = true)

        .def("run", &StringTensor<StrClient>::run)

        // Add other method bindings here ...
        .def("isRunning", &StringTensor<StrClient>::isRunning)

        .def("length", &StringTensor<StrClient>::getLength)

        .def("getNamespace", &StringTensor<StrClient>::getNamespace)

        .def("getBasename", &StringTensor<StrClient>::getBasename)

        .def("close", &StringTensor<StrClient>::close)

        .def("write_vec",
            (bool (StringTensor<StrClient>::*)(const std::vector<std::string>&, int))
            &StringTensor<StrClient>::write,
            py::arg("str_list"), py::arg("index") = 0)

        .def("read_vec", [](EigenIPC::StringTensor<EigenIPC::StrClient>& self, 
            py::list py_list, 
            int index) {
            
            // Check if the Python list is empty
            if (py_list.empty()) {
                return false;  // Return false if the list is empty
            }

            // Convert py::list to std::vector<std::string>
            std::vector<std::string> result_vec;
            for (auto item : py_list) {
                result_vec.push_back(item.cast<std::string>());
            }

            // Call the C++ method
            bool success = self.read(result_vec, index);

            // Clear the original Python list and refill with updated values
            py_list.attr("clear")();
            for (auto& str : result_vec) {
                py_list.append(str);
            }

            return success;

        }, py::arg("str_list"), py::arg("index") = 0)

        .def("write_str",
            (bool (StringTensor<StrClient>::*)(const std::string&, int))
            &StringTensor<StrClient>::write,
            py::arg("to_write"), py::arg("index") = 0)

        .def("read_str", [](EigenIPC::StringTensor<EigenIPC::StrClient>& self, int index) {
            std::string result_str;
            bool success = self.read(result_str, index);

            // Return a tuple containing the string and the success flag
            return std::make_tuple(result_str, success);

        }, py::arg("index") = 0)

        .def("get_raw_buffer", [](EigenIPC::StringTensor<EigenIPC::StrClient>& self) {
            
            EigenIPC::Tensor<int> buffer = self.get_raw_buffer();

            // Get the shape of the buffer tensor
            auto shape = py::array_t<int>({buffer.rows(), buffer.cols()});

            // initialize a pybind array of the right shape
            py::array_t<int> buffer_copy(shape);

            auto result_ptr = buffer_copy.mutable_data();

            // manually copy data from the buffer to the buffer copy
            std::memcpy(result_ptr, buffer.data(), buffer.size() * sizeof(int));

            return buffer_copy;

        })

        .def("get_shared_mem",
            [](EigenIPC::StringTensor<EigenIPC::StrClient>& self) -> EigenIPC::StrClient& {
                return self.getSharedMem();
            },
            py::return_value_policy::reference_internal)

        ;
}
