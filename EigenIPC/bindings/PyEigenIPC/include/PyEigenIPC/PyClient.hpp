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
#ifndef PYCLIENT_HPP
#define PYCLIENT_HPP

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <memory>

#include <EigenIPC/Client.hpp>
#include <EigenIPC/Journal.hpp>
#include <EigenIPC/DTypes.hpp>
#include <EigenIPC/Journal.hpp>

#include <PyEigenIPC/PyDTypes.hpp>

#include <PyEigenIPCUtils.hpp>
#include <WrapUtils.hpp>

namespace PyEigenIPC{

    namespace PyClient {

        using VLevel = EigenIPC::Journal::VLevel;
        using LogType = EigenIPC::Journal::LogType;
        using DType = EigenIPC::DType;

        pybind11::object ClientFactory(std::string basename = "MySharedMemory",
                                std::string name_space = "",
                                bool verbose = false,
                                VLevel vlevel = VLevel::V0,
                                bool safe = true,
                                EigenIPC::DType dtype = EigenIPC::DType::Float,
                                int layout = EigenIPC::ColMajor);

        void bind_ClientWrapper(pybind11::module& m);

        template <typename Scalar,
                int Layout = EigenIPC::MemLayoutDefault>
        void bindClientT(pybind11::module &m, const char* name);

        void bindClients(pybind11::module &m);

        void bindClientFactory(pybind11::module &m,
                            const char* name = "ClientFactory");

    }

}


#endif // PYCLIENT_HPP
