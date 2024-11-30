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
#ifndef HELPERS_HPP
#define HELPERS_HPP

#include <Eigen/Dense>

#include <EigenIPC/Journal.hpp>
#include <EigenIPC/DTypes.hpp>
#include <EigenIPC/ReturnCodes.hpp>

namespace EigenIPC {

    namespace helpers{
        
        using Index = Eigen::Index;
        using VLevel = Journal::VLevel;
        using LogType = Journal::LogType;

        inline bool canFitTensor(Index n_rows,
                        Index n_cols,
                        int i, int j,
                        Index n_rows2fit,
                        Index n_cols2fit,
                        Journal& journal,
                        ReturnCode& return_code,
                        bool verbose = true,
                        VLevel vlevel = Journal::VLevel::V0) {

            // Check if the indices (i, j) are within the matrix
            if (i < 0 || i >= n_rows || j < 0 || j >= n_cols) {

                if (verbose &&
                        vlevel > VLevel::V0) {

                    std::string warn =
                            std::string("Provided indeces are out of bounds wrt memory.");

                    journal.log(__FUNCTION__,
                                warn,
                                LogType::EXCEP);

                }

                return_code = return_code + ReturnCode::INDXOUT;

                return false;
            }

            // Check if there's enough space for the submatrix
            if (i + n_rows2fit > n_rows ||
                j + n_cols2fit > n_cols) {

                if (verbose &&
                        vlevel > VLevel::V0) {

                    std::string warn =
                            std::string("Out of bounds dimensions!");

                    journal.log(__FUNCTION__,
                                warn,
                                LogType::EXCEP);

                }

                return_code = return_code + ReturnCode::NOFIT;

                return false;
            }

            return true;
        }

        template <typename Scalar, int Layout = MemLayoutDefault>
        TensorView<Scalar, Layout> createViewFrom(
                                    Tensor<Scalar, Layout>& from, // original tensor
                                    int row_idx, int col_idx, // at indexes
                                    int n_rows, int n_cols); // of size

        template <typename Scalar, int Layout>
        TensorView<Scalar, Layout> createViewFrom(
            Tensor<Scalar, Layout>& from,
            int row_idx, int col_idx,
            int n_rows, int n_cols) {

            // note: this is not meant to be used in RT and hence should only be called during
            // initialization phases.

            ReturnCode return_code = ReturnCode::RESET; // unused; just to call the method

            Journal journal = Journal("");

            bool success = helpers::canFitTensor( // we check that are creating a valid view of "from"
                        from.rows(),
                        from.cols(),
                        row_idx, col_idx,
                        n_rows, n_cols,
                        journal,
                        return_code,
                        true,
                        VLevel::V3);

            if (!success) {

                std::string excep =
                        std::string("View does not fit in original tensor!!!");

                journal.log(__FUNCTION__,
                             excep,
                             LogType::EXCEP,
                             true // throw runtime exception
                             );
            } // if ok, we go on

            Scalar* startPtr; // start pointer from which to create the view

            // From Eigen doc:
            // The inner stride is the pointer increment between
            // two consecutive entries within a given row of a row-major matrix
            // or within a given column of a column-major matrix
            // The outer stride is the pointer increment between two consecutive
            // rows of a row-major matrix or between two consecutive columns of
            // a column-major matrix

            if (Layout == EigenIPC::ColMajor) {

                startPtr = from.data() +
                        col_idx * from.rows() + row_idx;

                // Define the stride based on the layout of the matrix

                DStrides stride(from.rows(), // outerstride
                                1); // innerstride

                // Use the mapped view with the provided stride to interpret the block correctly
                return TensorView<Scalar, Layout>(startPtr,
                             n_rows, n_cols,
                             stride);

            } else {

                startPtr = from.data() +
                        row_idx * from.cols() + col_idx;

                // Define the stride based on the layout of the matrix
                DStrides stride(from.cols(),
                                1);

                // Use the mapped view with the provided stride to interpret the block correctly
                return TensorView<Scalar, Layout>(startPtr,
                             n_rows, n_cols,
                             stride);
            }

        }

    }

}

#endif // HELPERS_HPP
