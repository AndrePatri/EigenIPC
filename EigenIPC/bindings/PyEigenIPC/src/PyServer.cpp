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
#include <PyEigenIPC/PyServer.hpp>

template <typename Scalar, int Layout>
void PyEigenIPC::PyServer::bindServerT(pybind11::module &m, const char* name) {

    // bindings of Server for Python

    pybind11::class_<EigenIPC::Server<Scalar, Layout>,
            std::shared_ptr<EigenIPC::Server<Scalar, Layout>>>(m, name)

        .def(pybind11::init<int, int, std::string, std::string, bool, VLevel, bool, bool>())

        .def("write", [](EigenIPC::Server<Scalar, Layout>& self,
                       PyEigenIPC::NumpyArray<Scalar>& arr,
                       int row, int col) {

            // we get the strides directly from the array
            // (since we use the strides, we don't care if it's rowmajor
            // or colmajor, the read will handle it smoothly)

            // Ensure the numpy array is mutable and get a pointer to its data
            pybind11::buffer_info buf_info = arr.request();

            if (PyEigenIPC::Utils::CheckInputBuffer<Layout>(buf_info)) { // check if buffer is OK

                // we first create a lightweight TensorView of the buffer memory
                Scalar* start_ptr = static_cast<Scalar*>(buf_info.ptr);
                EigenIPC::TensorView<Scalar, Layout> output_t(start_ptr, // start pointer
                                  buf_info.shape[0], // rows
                                  buf_info.shape[1], // cols
                                  PyEigenIPC::Utils::ToEigenStrides<Scalar, Layout>(buf_info) // strides
                                );

                // we use EigenIPC API to write to shared memory
                return self.write(output_t, row, col);

            } else {

                return false;

            }

        })

        .def("read", [](EigenIPC::Server<Scalar, Layout>& self,
                       PyEigenIPC::NumpyArray<Scalar>& arr,
                       int row, int col) {

            // we get the strides directly from the array
            // (since we use the strides, we don't care if it's rowmajor
            // or colmajor, the read will handle it smoothly)

            // Ensure the numpy array is mutable and get a pointer to its data
            pybind11::buffer_info buf_info = arr.request();

            if (PyEigenIPC::Utils::CheckInputBuffer<Layout>(buf_info)) { // check if buffer is OK

                // we first create a lightweight TensorView of the buffer memory
                Scalar* start_ptr = static_cast<Scalar*>(buf_info.ptr);
                EigenIPC::TensorView<Scalar, Layout> output_t(start_ptr, // start pointer
                                  buf_info.shape[0], // rows
                                  buf_info.shape[1], // cols
                                  PyEigenIPC::Utils::ToEigenStrides<Scalar, Layout>(buf_info) // strides
                                );

                // we use EigenIPC API to write to shared memory
                return self.read(output_t, row, col);

            } else {

                return false;

            }

        })

        .def("run", &EigenIPC::Server<Scalar, Layout>::run)

        .def("stop", &EigenIPC::Server<Scalar, Layout>::stop)

        .def("close", &EigenIPC::Server<Scalar, Layout>::close)

        .def("isRunning", &EigenIPC::Server<Scalar, Layout>::isRunning)

        .def("getScalarType", &EigenIPC::Server<Scalar, Layout>::getScalarType)

        .def("getNClients", &EigenIPC::Server<Scalar, Layout>::getNClients)

        .def("getNRows", &EigenIPC::Server<Scalar, Layout>::getNRows)

        .def("getNCols", &EigenIPC::Server<Scalar, Layout>::getNCols)
        
        .def("getNamespace", &EigenIPC::Server<Scalar, Layout>::getNamespace)
        .def("getBasename", &EigenIPC::Server<Scalar, Layout>::getBasename)

        .def("dataSemAcquire", [](EigenIPC::Server<Scalar, Layout>& self
                       ) {
            
            self.dataSemAcquire();

        })

        .def("dataSemRelease", &EigenIPC::Server<Scalar, Layout>::dataSemRelease);
}

pybind11::object PyEigenIPC::PyServer::ServerFactory(int n_rows,
                                    int n_cols,
                                    std::string basename,
                                    std::string name_space,
                                    bool verbose,
                                    VLevel vlevel,
                                    bool safe,
                                    bool force_reconnection,
                                    DType dtype,
                                    int layout) {

    switch (layout) {

        case EigenIPC::ColMajor:

            switch (dtype) {

                case DType::Bool:

                    return pybind11::cast(PyEigenIPC::ServerWrapper(new pybind11::object(
                                          pybind11::cast(std::make_shared<EigenIPC::Server<bool, EigenIPC::ColMajor>>(n_rows,
                                                                                                                        n_cols,
                                                                                                                        basename,
                                                                                                                        name_space,
                                                                                                                        verbose,
                                                                                                                        vlevel,
                                                                                                                        force_reconnection,
                                                                                                                        safe)))));
                case DType::Int:

                    return pybind11::cast(PyEigenIPC::ServerWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Server<int, EigenIPC::ColMajor>>(n_rows,
                                                                                                                                n_cols,
                                                                                                                                basename,
                                                                                                                                name_space,
                                                                                                                                verbose,
                                                                                                                                vlevel,
                                                                                                                                force_reconnection,
                                                                                                                                safe)))));
                case DType::Float:

                    return pybind11::cast(PyEigenIPC::ServerWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Server<float, EigenIPC::ColMajor>>(n_rows,
                                                                                                                                n_cols,
                                                                                                                                basename,
                                                                                                                                name_space,
                                                                                                                                verbose,
                                                                                                                                vlevel,
                                                                                                                                force_reconnection,
                                                                                                                                safe)))));
                case DType::Double:

                    return pybind11::cast(PyEigenIPC::ServerWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Server<double, EigenIPC::ColMajor>>(n_rows,
                                                                                                                                n_cols,
                                                                                                                                basename,
                                                                                                                                name_space,
                                                                                                                                verbose,
                                                                                                                                vlevel,
                                                                                                                                force_reconnection,
                                                                                                                                safe)))));
                default:

                    throw std::runtime_error("Invalid dtype specified!");
            }

        case EigenIPC::RowMajor:

            switch (dtype) {

                case DType::Bool:

                    return pybind11::cast(PyEigenIPC::ServerWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Server<bool, EigenIPC::RowMajor>>(n_rows,
                                                                                                                                n_cols,
                                                                                                                                basename,
                                                                                                                                name_space,
                                                                                                                                verbose,
                                                                                                                                vlevel,
                                                                                                                                force_reconnection,
                                                                                                                                safe)))));
                case DType::Int:
                    return pybind11::cast(PyEigenIPC::ServerWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Server<int, EigenIPC::RowMajor>>(n_rows,
                                                                                                                                n_cols,
                                                                                                                                basename,
                                                                                                                                name_space,
                                                                                                                                verbose,
                                                                                                                                vlevel,
                                                                                                                                force_reconnection,
                                                                                                                                safe)))));
                case DType::Float:
                    return pybind11::cast(PyEigenIPC::ServerWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Server<float, EigenIPC::RowMajor>>(n_rows,
                                                                                                                                n_cols,
                                                                                                                                basename,
                                                                                                                                name_space,
                                                                                                                                verbose,
                                                                                                                                vlevel,
                                                                                                                                force_reconnection,
                                                                                                                                safe)))));
                case DType::Double:
                    return pybind11::cast(PyEigenIPC::ServerWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Server<double, EigenIPC::RowMajor>>(n_rows,
                                                                                                                                    n_cols,
                                                                                                                                    basename,
                                                                                                                                    name_space,
                                                                                                                                    verbose,
                                                                                                                                    vlevel,
                                                                                                                                    force_reconnection,
                                                                                                                                    safe)))));
                default:

                    throw std::runtime_error("Invalid dtype specified!");
            }

        default:

            throw std::runtime_error("Invalid layout specified!");

    }
}

void PyEigenIPC::PyServer::bind_ServerWrapper(pybind11::module& m) {

    // By handling everything in Python-space,
    // we eliminate the need to handle the type
    // conversion in C++, making the ServerWrapper
    // agnostic to the actual data type of the underlying Server.

    // Here we lose a bit of performance wrt using the bare class bindings,
    // since additional logic and checks are necessary. However, we gain
    // much cleaner interface and flexibility

    // (here we are calling python code from C++)

    pybind11::class_<PyEigenIPC::ServerWrapper> cls(m, "Server");

    cls.def("run", [](PyEigenIPC::ServerWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& server) {

            return server.attr("run")();

        });

    });

    cls.def("stop", [](PyEigenIPC::ServerWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& server) {

            return server.attr("stop")();

        });

    });

    cls.def("close", [](PyEigenIPC::ServerWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& server) {

            return server.attr("close")();

        });

    });

    cls.def("isRunning", [](PyEigenIPC::ServerWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& server) {

            return server.attr("isRunning")().cast<bool>();

        });

    });

    cls.def("getNClients", [](PyEigenIPC::ServerWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& server) {

            return server.attr("getNClients")().cast<int>();

        });

    });

    cls.def("getNRows", [](PyEigenIPC::ServerWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& server) {

            return server.attr("getNRows")().cast<int>();

        });

    });

    cls.def("getNCols", [](PyEigenIPC::ServerWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& server) {

            return server.attr("getNCols")().cast<int>();

        });

    });

    cls.def("getScalarType", [](PyEigenIPC::ServerWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& server) {

            return server.attr("getScalarType")().cast<DType>();

        });

    });

    cls.def("getNamespace", [](PyEigenIPC::ServerWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& server) {

            return server.attr("getNamespace")().cast<std::string>();

        });

    });

    cls.def("getBasename", [](PyEigenIPC::ServerWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& server) {

            return server.attr("getBasename")().cast<std::string>();

        });

    });

    cls.def("dataSemAcquire", [](PyEigenIPC::ServerWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& server) {

            return server.attr("dataSemAcquire")();

        });

    });

    cls.def("dataSemRelease", [](PyEigenIPC::ServerWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& server) {

            return server.attr("dataSemRelease")();

        });

    });

    cls.def("write", [](PyEigenIPC::ServerWrapper& wrapper,
                             pybind11::array& np_array,
                             int row, int col) {

        return wrapper.execute([&](pybind11::object& server) -> bool {

            // we need to check data compatibility at runtime
            // (this was not necessary in C++, since it's something
            // checked at compile time)
            // this introduces a slight overhead, but avoids unpredicted
            // behavior when the user passes non-compatible types

            // and of the input np array
            pybind11::dtype np_dtype = np_array.dtype();

            pybind11::buffer_info buf_info = np_array.request();

            switch (server.attr("getScalarType")().cast<DType>()) {

            case DType::Bool:

                if (!np_dtype.is(pybind11::dtype::of<bool>())) {

                    std::string error = std::string("Mismatched dtype: expected bool numpy array but got ") +
                                    pybind11::str(np_dtype).cast<std::string>();

                    EigenIPC::Journal::log("Server",
                                 "write",
                                 error,
                                 LogType::EXCEP,
                                 true);

                }

                break;

            case DType::Int:

                if (!np_dtype.is(pybind11::dtype::of<int>())) {

                    std::string error = std::string("Mismatched dtype: expected int numpy array but got ") +
                                    pybind11::str(np_dtype).cast<std::string>();

                    EigenIPC::Journal::log("Server",
                                 "write",
                                 error,
                                 LogType::EXCEP,
                                 true);
                }

                break;

            case DType::Float:

                if (!np_dtype.is(pybind11::dtype::of<float>())) {

                    std::string error = std::string("Mismatched dtype: expected float numpy array but got ") +
                                    pybind11::str(np_dtype).cast<std::string>();

                    EigenIPC::Journal::log("Server",
                                 "write",
                                 error,
                                 LogType::EXCEP,
                                 true);
                }

                break;

            case DType::Double:

                if (!np_dtype.is(pybind11::dtype::of<double>())) {

                    std::string error = std::string("Mismatched dtype: expected double numpy array but got ") +
                                    pybind11::str(np_dtype).cast<std::string>();

                    EigenIPC::Journal::log("Server",
                                 "write",
                                 error,
                                 LogType::EXCEP,
                                 true);
                }

                break;

            default:

                EigenIPC::Journal::log("Server",
                             "write",
                             "Mismatched dtype: provided dtype not supported.",
                             LogType::EXCEP,
                             true);

            }

            // now we can safely read the tensor
            return server.attr("write")(np_array,
                                             row, col).cast<bool>();

        });

    }, pybind11::arg("data"), pybind11::arg("row") = 0, pybind11::arg("col") = 0);

    cls.def("read", [](PyEigenIPC::ServerWrapper& wrapper,
                             pybind11::array& np_array,
                             int row, int col) {

        return wrapper.execute([&](pybind11::object& server) -> bool {

            // we need to check data compatibility at runtime
            // (this was not necessary in C++, since it's something
            // checked at compile time)
            // this introduces a slight overhead, but avoids unpredicted
            // behavior when the user passes non-compatible types

            // and of the input np array
            pybind11::dtype np_dtype = np_array.dtype();

            switch (server.attr("getScalarType")().cast<DType>()) {

            case DType::Bool:

                if (!np_dtype.is(pybind11::dtype::of<bool>())) {

                    std::string error = std::string("Mismatched dtype: expected bool numpy array but got ") +
                                    pybind11::str(np_dtype).cast<std::string>();

                    EigenIPC::Journal::log("Server",
                                 "read",
                                 error,
                                 LogType::EXCEP,
                                 true);

                }

                break;

            case DType::Int:

                if (!np_dtype.is(pybind11::dtype::of<int>())) {

                    std::string error = std::string("Mismatched dtype: expected int numpy array but got ") +
                                    pybind11::str(np_dtype).cast<std::string>();

                    EigenIPC::Journal::log("Server",
                                 "read",
                                 error,
                                 LogType::EXCEP,
                                 true);
                }

                break;

            case DType::Float:

                if (!np_dtype.is(pybind11::dtype::of<float>())) {

                    std::string error = std::string("Mismatched dtype: expected float numpy array but got ") +
                                    pybind11::str(np_dtype).cast<std::string>();

                    EigenIPC::Journal::log("Server",
                                 "read",
                                 error,
                                 LogType::EXCEP,
                                 true);
                }

                break;

            case DType::Double:

                if (!np_dtype.is(pybind11::dtype::of<double>())) {

                    std::string error = std::string("Mismatched dtype: expected double numpy array but got ") +
                                    pybind11::str(np_dtype).cast<std::string>();

                    EigenIPC::Journal::log("Server",
                                 "read",
                                 error,
                                 LogType::EXCEP,
                                 true);
                }

                break;

            default:

                EigenIPC::Journal::log("Server",
                             "read",
                             "Mismatched dtype: provided dtype not supported.",
                             LogType::EXCEP,
                             true);

            }

            // now we can safely read the tensor
            return server.attr("read")(np_array,
                                             row, col).cast<bool>();

        });

    }, pybind11::arg("tensor"), pybind11::arg("row") = 0, pybind11::arg("col") = 0);
}

void PyEigenIPC::PyServer::bindServers(pybind11::module& m) {

    bindServerT<bool, EigenIPC::ColMajor>(m, "PyServerBoolColMaj");
    bindServerT<bool, EigenIPC::RowMajor>(m, "PyServerBoolRowMaj");

    bindServerT<int, EigenIPC::ColMajor>(m, "PyServerIntColMaj");
    bindServerT<int, EigenIPC::RowMajor>(m, "PyServerIntRowMaj");

    bindServerT<float, EigenIPC::ColMajor>(m, "PyServerFloatColMaj");
    bindServerT<float, EigenIPC::RowMajor>(m, "PyServerFloatRowMaj");

    bindServerT<double, EigenIPC::ColMajor>(m, "PyServerDoubleColMaj");
    bindServerT<double, EigenIPC::RowMajor>(m, "PyServerDoubleRowMaj");

}

void PyEigenIPC::PyServer::bindServerFactory(pybind11::module& m,
                           const char* name) {

    m.def(name, &ServerFactory,
        pybind11::arg("n_rows"),
        pybind11::arg("n_cols"),
        pybind11::arg("basename"),
        pybind11::arg("namespace") = "",
        pybind11::arg("verbose") = false,
        pybind11::arg("vlevel") = VLevel::V0,
        pybind11::arg("safe") = true,
        pybind11::arg("force_reconnection") = false,
        pybind11::arg("dtype") = DType::Float,
        pybind11::arg("layout") = EigenIPC::RowMajor, // default of numpy and pytorch
        "Create a new server with the specified arguments and dtype.",
        pybind11::return_value_policy::reference_internal); // reference_internal keeps the underlying object alive,
        // as long as the python is

}
