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
#include <pybind11/pybind11.h>

#include <EigenIPC/Journal.hpp>

#include <PyEigenIPC/PyStringTensor.hpp>
#include <PyEigenIPC/PyServer.hpp>
#include <PyEigenIPC/PyClient.hpp>
#include <PyEigenIPC/PyProducer.hpp>
#include <PyEigenIPC/PyConsumer.hpp>

using namespace EigenIPC;
using namespace PyEigenIPC;
using VLevel = Journal::VLevel;

inline bool isRelease() {

    #ifdef IS_RELEASE

        return true;

    #else

        return false;

    #endif
}

void bind_Journal(py::module &m) {

    py::enum_<EigenIPC::Journal::LogType>(m, "LogType")

        .value("WARN", EigenIPC::Journal::LogType::WARN)
        .value("EXCEP", EigenIPC::Journal::LogType::EXCEP)
        .value("INFO", EigenIPC::Journal::LogType::INFO)
        .value("STAT", EigenIPC::Journal::LogType::STAT)

        .export_values();

    py::class_<EigenIPC::Journal>(m, "Journal")

//        .def(py::init<const std::string &>())

//        .def("log", py::overload_cast<const std::string &,
//                                     const std::string &,
//                                     const std::string &,
//                                     EigenIPC::Journal::LogType,
//                                     bool>(&EigenIPC::Journal::log))

        .def_static("log", py::overload_cast<const std::string &,
                                            const std::string &,
                                            const std::string &,
                                            EigenIPC::Journal::LogType,
                                            bool>(&EigenIPC::Journal::log),
                    py::arg("classname"), py::arg("methodname"),
                    py::arg("message"), py::arg("log_type"),
                    py::arg("throw_when_excep") = false);
}

PYBIND11_MODULE(PyEigenIPC, m) {

    m.doc() = "pybind11 EigenIPC bindings";

    m.def("isRelease", &isRelease);

    pybind11::enum_<DType>(m, "dtype")
        .value("Bool", DType::Bool)
        .value("Int", DType::Int)
        .value("Float", DType::Float)
        .value("Double", DType::Double);

    m.attr("RowMajor") = RowMajor;
    m.attr("ColMajor") = ColMajor;

    pybind11::enum_<VLevel>(m, "VLevel")
        .value("V0", Journal::VLevel::V0)
        .value("V1", Journal::VLevel::V1)
        .value("V2", Journal::VLevel::V2)
        .value("V3", Journal::VLevel::V3)
        .export_values();

    // In your Pybind11 bindings:
    m.def("toNumpyDType", [](DType dtype) {
        switch(dtype) {
            case DType::Bool: return pybind11::dtype::of<bool>();
            case DType::Int: return pybind11::dtype::of<int>();
            case DType::Float: return pybind11::dtype::of<float>();
            case DType::Double: return pybind11::dtype::of<double>();

            default: throw std::runtime_error("Unsupported DType conversion!");
        }
    });

    bind_Journal(m);

    // Client bindings
    PyClient::bindClients(m); // binds all client types

    PyClient::bind_ClientWrapper(m); // binds the client wrapper

    PyClient::bindClientFactory(m); // binds the factory for Clients

    // Server bindings

    PyServer::bindServers(m); // binds all client types

    PyServer::bind_ServerWrapper(m); // binds the client wrapper

    PyServer::bindServerFactory(m); // binds the factory for Clients

    // String tensor bindings
    PyStringTensor::declare_StringTensorServer(m);

    PyStringTensor::declare_StringTensorClient(m);

    // Cond. var. bindings 

    PyConsumer::bind_Consumer(m);
    PyProducer::bind_Producer(m);

}

