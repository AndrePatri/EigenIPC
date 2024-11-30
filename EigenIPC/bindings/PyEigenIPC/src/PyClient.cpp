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
#include <PyEigenIPC/PyClient.hpp>
#include <EigenIPC/Journal.hpp>

template <typename Scalar, int Layout>
void PyEigenIPC::PyClient::bindClientT(pybind11::module &m, const char* name) {

    // bindings of Client for Python

    pybind11::class_<EigenIPC::Client<Scalar, Layout>,
            std::shared_ptr<EigenIPC::Client<Scalar, Layout>>>(m, name)

        .def(pybind11::init<std::string, std::string, bool, VLevel, bool>())

        .def("write", [](EigenIPC::Client<Scalar, Layout>& self,
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

        .def("read", [](EigenIPC::Client<Scalar, Layout>& self,
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

        .def("attach", &EigenIPC::Client<Scalar, Layout>::attach)

        .def("detach", &EigenIPC::Client<Scalar, Layout>::detach)

        .def("close", &EigenIPC::Client<Scalar, Layout>::close)

        .def("isRunning", &EigenIPC::Client<Scalar, Layout>::isAttached)

        .def("getScalarType", &EigenIPC::Client<Scalar, Layout>::getScalarType)

        .def("getNRows", &EigenIPC::Client<Scalar, Layout>::getNRows)

        .def("getNCols", &EigenIPC::Client<Scalar, Layout>::getNCols)
        
        .def("getNamespace", &EigenIPC::Client<Scalar, Layout>::getNamespace)
        .def("getBasename", &EigenIPC::Client<Scalar, Layout>::getBasename)

        .def("dataSemAcquire", [](EigenIPC::Client<Scalar, Layout>& self
                       ) {
            
            self.dataSemAcquire();

        })

        .def("dataSemRelease", &EigenIPC::Client<Scalar, Layout>::dataSemRelease);
}

pybind11::object PyEigenIPC::PyClient::ClientFactory(std::string basename,
                        std::string name_space,
                        bool verbose,
                        VLevel vlevel,
                        bool safe,
                        DType dtype,
                        int layout) {

    switch (layout) {

        case EigenIPC::ColMajor:

            switch (dtype) {

                case DType::Bool:

                    return pybind11::cast(PyEigenIPC::ClientWrapper(new pybind11::object(
                                          pybind11::cast(std::make_shared<EigenIPC::Client<bool, EigenIPC::ColMajor>>(basename,
                                                                                                                              name_space,
                                                                                                                              verbose,
                                                                                                                              vlevel,
                                                                                                                              safe)))));
                case DType::Int:

                    return pybind11::cast(PyEigenIPC::ClientWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Client<int, EigenIPC::ColMajor>>(basename,
                                                                                                                                        name_space,
                                                                                                                                        verbose,
                                                                                                                                        vlevel,
                                                                                                                                        safe)))));
                case DType::Float:

                    return pybind11::cast(PyEigenIPC::ClientWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Client<float, EigenIPC::ColMajor>>(basename,
                                                                                                                                          name_space,
                                                                                                                                          verbose,
                                                                                                                                          vlevel,
                                                                                                                                          safe)))));
                case DType::Double:

                    return pybind11::cast(PyEigenIPC::ClientWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Client<double, EigenIPC::ColMajor>>(basename,
                                                                                                                                            name_space,
                                                                                                                                            verbose,
                                                                                                                                            vlevel, 
                                                                                                                                            safe)))));
                default:

                    throw std::runtime_error("Invalid dtype specified!");
            }

        case EigenIPC::RowMajor:

            switch (dtype) {

                case DType::Bool:

                    return pybind11::cast(PyEigenIPC::ClientWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Client<bool, EigenIPC::RowMajor>>(basename,
                                                                                                                                          name_space,
                                                                                                                                          verbose,
                                                                                                                                          vlevel,
                                                                                                                                          safe)))));
                case DType::Int:
                    return pybind11::cast(PyEigenIPC::ClientWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Client<int, EigenIPC::RowMajor>>(basename,
                                                                                                                                        name_space,
                                                                                                                                        verbose,
                                                                                                                                        vlevel,
                                                                                                                                        safe)))));
                case DType::Float:
                    return pybind11::cast(PyEigenIPC::ClientWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Client<float, EigenIPC::RowMajor>>(basename,
                                                                                                                                          name_space,
                                                                                                                                          verbose,
                                                                                                                                          vlevel,
                                                                                                                                          safe)))));
                case DType::Double:
                    return pybind11::cast(PyEigenIPC::ClientWrapper(new pybind11::object(
                                                      pybind11::cast(std::make_shared<EigenIPC::Client<double, EigenIPC::RowMajor>>(basename,
                                                                                                                                          name_space,
                                                                                                                                          verbose,
                                                                                                                                          vlevel,
                                                                                                                                          safe)))));
                default:

                    throw std::runtime_error("Invalid dtype specified!");
            }

        default:

            throw std::runtime_error("Invalid layout specified!");

    }
}

void PyEigenIPC::PyClient::bind_ClientWrapper(pybind11::module& m) {

    // By handling everything in Python-space,
    // we eliminate the need to handle the type
    // conversion in C++, making the ClientWrapper
    // agnostic to the actual data type of the underlying Client.

    // Here we lose a bit of performance wrt using the bare class bindings,
    // since additional logic and checks are necessary. However, we gain
    // much cleaner interface and flexibility

    // (here we are calling python code from C++)

    pybind11::class_<PyEigenIPC::ClientWrapper> cls(m, "Client");

    cls.def("attach", [](PyEigenIPC::ClientWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& client) {

            return client.attr("attach")(); // calls the attach (from Client's PY bindings)

        });

    });

    cls.def("detach", [](PyEigenIPC::ClientWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& client) {

            return client.attr("detach")();

        });

    });

    cls.def("close", [](PyEigenIPC::ClientWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& client) {

            return client.attr("close")();

        });

    });

    cls.def("isRunning", [](PyEigenIPC::ClientWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& client) {

            return client.attr("isRunning")().cast<bool>();

        });

    });

    cls.def("getNRows", [](PyEigenIPC::ClientWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& client) {

            return client.attr("getNRows")().cast<int>();

        });

    });

    cls.def("getNCols", [](PyEigenIPC::ClientWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& client) {

            return client.attr("getNCols")().cast<int>();

        });

    });

    cls.def("getScalarType", [](PyEigenIPC::ClientWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& client) {

            return client.attr("getScalarType")().cast<DType>();

        });

    });

    cls.def("getNamespace", [](PyEigenIPC::ClientWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& client) {

            return client.attr("getNamespace")().cast<std::string>();

        });

    });

    cls.def("getBasename", [](PyEigenIPC::ClientWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& client) {

            return client.attr("getBasename")().cast<std::string>();

        });

    });

    cls.def("dataSemAcquire", [](PyEigenIPC::ClientWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& client) {

            return client.attr("dataSemAcquire")();

        });

    });

    cls.def("dataSemRelease", [](PyEigenIPC::ClientWrapper& wrapper) {

        return wrapper.execute([&](pybind11::object& client) {

            return client.attr("dataSemRelease")();

        });

    });

    cls.def("write", [](PyEigenIPC::ClientWrapper& wrapper,
                             pybind11::array& np_array,
                             int row, int col) {

        return wrapper.execute([&](pybind11::object& client) -> bool {

            // we need to check data compatibility at runtime
            // (this was not necessary in C++, since it's something
            // checked at compile time)
            // this introduces a slight overhead, but avoids unpredicted
            // behavior when the user passes non-compatible types

            // and of the input np array
            pybind11::dtype np_dtype = np_array.dtype();

            switch (client.attr("getScalarType")().cast<DType>()) {

            case DType::Bool:

                if (!np_dtype.is(pybind11::dtype::of<bool>())) {

                    std::string error = std::string("Mismatched dtype: expected bool numpy array but got ") +
                                    pybind11::str(np_dtype).cast<std::string>();

                    EigenIPC::Journal::log("Client",
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

                    EigenIPC::Journal::log("Client",
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

                    EigenIPC::Journal::log("Client",
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

                    EigenIPC::Journal::log("Client",
                                 "write",
                                 error,
                                 LogType::EXCEP,
                                 true);
                }

                break;

            default:

                EigenIPC::Journal::log("Client",
                             "write",
                             "Mismatched dtype: provided dtype not supported.",
                             LogType::EXCEP,
                             true);

            }

            // now we can safely read the tensor
            return client.attr("write")(np_array,
                                             row, col).cast<bool>();

        });

    }, pybind11::arg("data"), pybind11::arg("row") = 0, pybind11::arg("col") = 0);

    cls.def("read", [](PyEigenIPC::ClientWrapper& wrapper,
                             pybind11::array& np_array,
                             int row, int col) {

        return wrapper.execute([&](pybind11::object& client) -> bool {

            // we need to check data compatibility at runtime
            // (this was not necessary in C++, since it's something
            // checked at compile time)
            // this introduces a slight overhead, but avoids unpredicted
            // behavior when the user passes non-compatible types

            // and of the input np array
            pybind11::dtype np_dtype = np_array.dtype();

            switch (client.attr("getScalarType")().cast<DType>()) {

            case DType::Bool:

                if (!np_dtype.is(pybind11::dtype::of<bool>())) {

                    std::string error = std::string("Mismatched dtype: expected bool numpy array but got ") +
                                    pybind11::str(np_dtype).cast<std::string>();

                    EigenIPC::Journal::log("Client",
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

                    EigenIPC::Journal::log("Client",
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

                    EigenIPC::Journal::log("Client",
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

                    EigenIPC::Journal::log("Client",
                                 "read",
                                 error,
                                 LogType::EXCEP,
                                 true);
                }

                break;

            default:

                EigenIPC::Journal::log("Client",
                             "read",
                             "Mismatched dtype: provided dtype not supported.",
                             LogType::EXCEP,
                             true);

            }

            // now we can safely read the tensor
            return client.attr("read")(np_array,
                                             row, col).cast<bool>();

        });

    }, pybind11::arg("tensor"), pybind11::arg("row") = 0, pybind11::arg("col") = 0);
}

void PyEigenIPC::PyClient::bindClients(pybind11::module& m) {

    bindClientT<bool, EigenIPC::ColMajor>(m, "PyClientBoolColMaj");
    bindClientT<bool, EigenIPC::RowMajor>(m, "PyClientBoolRowMaj");

    bindClientT<int, EigenIPC::ColMajor>(m, "PyClientIntColMaj");
    bindClientT<int, EigenIPC::RowMajor>(m, "PyClientIntRowMaj");

    bindClientT<float, EigenIPC::ColMajor>(m, "PyClientFloatColMaj");
    bindClientT<float, EigenIPC::RowMajor>(m, "PyClientFloatRowMaj");

    bindClientT<double, EigenIPC::ColMajor>(m, "PyClientDoubleColMaj");
    bindClientT<double, EigenIPC::RowMajor>(m, "PyClientDoubleRowMaj");

}

void PyEigenIPC::PyClient::bindClientFactory(pybind11::module& m,
                           const char* name) {

    m.def(name, &ClientFactory,
        pybind11::arg("basename") = "MySharedMemory",
        pybind11::arg("namespace") = "",
        pybind11::arg("verbose") = false,
        pybind11::arg("vlevel") = VLevel::V0,
        pybind11::arg("safe") = true,
        pybind11::arg("dtype") = DType::Float,
        pybind11::arg("layout") = EigenIPC::RowMajor, // default of numpy and pytorch
        "Create a new client with the specified arguments and dtype.",
        pybind11::return_value_policy::reference_internal); // reference_internal keeps the underlying object alive,
        // as long as the python is

}
