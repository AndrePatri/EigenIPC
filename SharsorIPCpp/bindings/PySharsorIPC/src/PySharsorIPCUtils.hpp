// Copyright (C) 2023  Andrea Patrizi (AndrePatri)
// 
// This file is part of SharsorIPCpp and distributed under the General Public License version 2 license.
// 
// SharsorIPCpp is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 2 of the License, or
// (at your option) any later version.
// 
// SharsorIPCpp is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with SharsorIPCpp.  If not, see <http://www.gnu.org/licenses/>.
// 
#ifndef PYSHARSORIPCUTILS_HPP
#define PYSHARSORIPCUTILS_HPP

#include <SharsorIPCpp/DTypes.hpp>
#include <SharsorIPCpp/Journal.hpp>
#include <SharsorIPCpp/ReturnCodes.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace PySharsorIPC {

    namespace Utils {

        using VLevel = SharsorIPCpp::Journal::VLevel;
        using LogType = SharsorIPCpp::Journal::LogType;

        template<int Layout>
        bool CheckInputBuffer(pybind11::buffer_info& buf_info){

            if (buf_info.ndim != 2) {

                // print non-blocking exception for the user
                std::string message = std::string("Expected a 2D array, but got ") +
                        std::string("an array of dimension ") + std::to_string(buf_info.ndim);

                SharsorIPCpp::Journal::log("PySharsorIPC::Utils",
                            "CheckInputBuffer",
                            message,
                            LogType::EXCEP);

                return false;

            } // we restrict ourselves to work with only 2D tensors


            if ((Layout == SharsorIPCpp::RowMajor) &&
                    (buf_info.strides[0] < buf_info.strides[1]) ) { // not coherent -> stop

                // (in case strides are equal it means we are dealing with a Scalar ->
                // Layout does not matter)

                // print non-blocking exception for the user
                std::string message = std::string("Expected np array of layout RowMajor, but got ") +
                        std::string("ColMajor. Server and array layout must match! ") +
                        std::string("Provided strides are, respectively, ") +
                        std::to_string(buf_info.strides[0]) +
                        std::string(", ") + std::to_string(buf_info.strides[1]);

                SharsorIPCpp::Journal::log("PySharsorIPC::Utils",
                            "CheckInputBuffer",
                            message,
                            LogType::EXCEP);

                return false;
            }

            if ((Layout == SharsorIPCpp::ColMajor) &&
                    (buf_info.strides[0] > buf_info.strides[1]) ) { // not coherent -> stop

                // (in case strides are equal it means we are dealing with a Scalar ->
                // Layout does not matter)

                // print non-blocking exception for the user
                std::string message = std::string("Expected np array of layout ColMajor, but got ") +
                        std::string("RowMajor. Server and array layout must match! ") +
                        std::string("Provided strides are, respectively, ") +
                        std::to_string(buf_info.strides[0]) +
                        std::string(", ") + std::to_string(buf_info.strides[1]);

                SharsorIPCpp::Journal::log("PyServer",
                            "write",
                            message,
                            LogType::EXCEP);

                return false;
            }


            return true;

        }

        template<typename Scalar, int Layout>
        SharsorIPCpp::DStrides ToEigenStrides(pybind11::buffer_info& buf_info){

            // From Eigen doc:
            // The inner stride is the pointer increment between
            // two consecutive entries within a given row of a row-major matrix
            // or within a given column of a column-major matrix
            // The outer stride is the pointer increment between two consecutive
            // rows of a row-major matrix or between two consecutive columns of
            // a column-major matrix

            if (Layout == SharsorIPCpp::RowMajor) {

                SharsorIPCpp::DStrides strides = SharsorIPCpp::DStrides(
                                                    buf_info.strides[0] / sizeof(Scalar),
                                                    buf_info.strides[1] / sizeof(Scalar)
                                                );

                return strides;

            } else { // ColMajor

                SharsorIPCpp::DStrides strides = SharsorIPCpp::DStrides(
                                                    buf_info.strides[1] / sizeof(Scalar),
                                                    buf_info.strides[0] / sizeof(Scalar)
                                                );

                return strides;

            }

        }

    }

}

#endif // PYSHARSORIPCUTILS_HPP
