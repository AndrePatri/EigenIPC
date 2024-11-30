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
#ifndef WRAPUTILS_HPP
#define WRAPUTILS_HPP

#include <pybind11/pybind11.h>
#include <memory>
#include <type_traits>

namespace PyEigenIPC {

    class ClientWrapper {

        public:

            ClientWrapper(pybind11::object* Obj);

            template<typename Func>
            auto execute(Func&& f) -> decltype(f(std::declval<pybind11::object&>()));

        public:

            std::unique_ptr<pybind11::object> wrapped;

    };

    class ServerWrapper {

        public:

            ServerWrapper(pybind11::object* Obj);

            template<typename Func>
            auto execute(Func&& f) -> decltype(f(std::declval<pybind11::object&>()));

        public:

            std::unique_ptr<pybind11::object> wrapped;

    };

}

#include "WrapUtils.inl"

#endif  // WRAPUTILS_HPP

