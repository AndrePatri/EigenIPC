#include <pybind11/pybind11.h>

#include <SharsorIPCpp/Journal.hpp>

#include <PySharsorIPC/PyStringTensor.hpp>
#include <PySharsorIPC/PyServer.hpp>
#include <PySharsorIPC/PyClient.hpp>

using namespace SharsorIPCpp;
using namespace PySharsorIPC;

inline bool isRelease() {

    #ifdef IS_RELEASE

        return true;

    #else

        return false;

    #endif
}

void bind_Journal(py::module &m) {

    py::enum_<SharsorIPCpp::Journal::LogType>(m, "LogType")

        .value("WARN", SharsorIPCpp::Journal::LogType::WARN)
        .value("EXCEP", SharsorIPCpp::Journal::LogType::EXCEP)
        .value("INFO", SharsorIPCpp::Journal::LogType::INFO)
        .value("STAT", SharsorIPCpp::Journal::LogType::STAT)

        .export_values();

    py::class_<SharsorIPCpp::Journal>(m, "Journal")

        .def(py::init<const std::string &>())

        .def("log", py::overload_cast<const std::string &,
                                     const std::string &,
                                     const std::string &,
                                     SharsorIPCpp::Journal::LogType,
                                     bool>(&SharsorIPCpp::Journal::log))

        .def_static("log", py::overload_cast<const std::string &,
                                            const std::string &,
                                            const std::string &,
                                            SharsorIPCpp::Journal::LogType,
                                            bool>(&SharsorIPCpp::Journal::log));
}

PYBIND11_MODULE(PySharsorIPC, m) {

    m.doc() = "pybind11 SharsorIPCpp bindings";

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
    bindClients(m); // binds all client types

    bind_ClientWrapper(m); // binds the client wrapper

    bindClientFactory(m); // binds the factory for Clients

    // Server bindings

    bindServers(m); // binds all client types

    bind_ServerWrapper(m); // binds the client wrapper

    bindServerFactory(m); // binds the factory for Clients

    // String tensor bindings
    declare_StringTensorServer(m);

    declare_StringTensorClient(m);

}

