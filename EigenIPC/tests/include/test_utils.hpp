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
#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <EigenIPC/Journal.hpp>
#include <EigenIPC/DTypes.hpp>

#include <random>

namespace EigenIPC {

    void check_comp_type(Journal& journal)
    {
        std::string message;

        #ifdef NDEBUG

            #ifdef _RELWITHDEBINFO

                message = std::string("EigenIPC was compiled in RelWithDebInfo mode. ") +
                        std::string("For meaninful results, you should compile it in Release mode.\n");

                journal.log("check_comp_type",
                            message,
                            Journal::LogType::WARN);

            #else

                message = std::string("EigenIPC was compiled in Release mode. ") +
                    std::string("This is good and will ensure meaningful benchmarking results.\n");

                journal.log("check_comp_type",
                            message,
                            Journal::LogType::STAT);

            #endif

        #else

            message = std::string("EigenIPC was compiled in Debug mode. ") +
                std::string("For meaninful results, you should compile it in Release mode.\n");

            journal.log("check_comp_type",
                        message,
                        Journal::LogType::WARN);

        #endif

    }

    std::string random_string(size_t length) {

        static const char alphabet[] = "abcdefghijklmnopqrst)(/)/£0430*é°è{}]54309582'2'uvwxyz";  // Add more characters if needed

        static std::default_random_engine rng(std::random_device{}());
        static std::uniform_int_distribution<> dist(0, sizeof(alphabet) - 2);  // -2 because sizeof(alphabet) includes the null-terminator

        std::string result;
        result.reserve(length);

        for(size_t i = 0; i < length; ++i) {
            result.push_back(alphabet[dist(rng)]);
        }

        return result;
    }

    int random_int(int n) {

        static std::default_random_engine rng(std::random_device{}());

        std::uniform_int_distribution<> dist(0, n);

        return dist(rng);
    }

    template <typename T>
    std::string getTypeAsString() {
        return "Unknown";
    }

    template <>
    std::string getTypeAsString<bool>() {
        return "bool";
    }

    template <>
    std::string getTypeAsString<int>() {
        return "int";
    }

    template <>
    std::string getTypeAsString<float>() {
        return "float";
    }

    template <>
    std::string getTypeAsString<double>() {
        return "double";
    }

    template <typename T, int Layout>
    struct Thresholds;

    template <>
    struct Thresholds<bool, EigenIPC::RowMajor> {
        static constexpr double READ_T_MAX_THRESH = 10000000;
        static constexpr double WRITE_T_MAX_THRESH = 10000000;
        static constexpr double READ_TV_MAX_THRESH = 10000000;
        static constexpr double WRITE_TV_MAX_THRESH = 10000000;
        static constexpr double READ_T_AVRG_THRESH = 1500;
        static constexpr double WRITE_T_AVRG_THRESH = 1500;
        static constexpr double READ_TV_AVRG_THRESH = 1500;
        static constexpr double WRITE_TV_AVRG_THRESH = 1500;

    };

    template <>
    struct Thresholds<bool, EigenIPC::ColMajor> {
        static constexpr double READ_T_MAX_THRESH = 10000000;
        static constexpr double WRITE_T_MAX_THRESH = 10000000;
        static constexpr double READ_TV_MAX_THRESH = 10000000;
        static constexpr double WRITE_TV_MAX_THRESH = 10000000;
        static constexpr double READ_T_AVRG_THRESH = 1500;
        static constexpr double WRITE_T_AVRG_THRESH = 1500;
        static constexpr double READ_TV_AVRG_THRESH = 1500;
        static constexpr double WRITE_TV_AVRG_THRESH = 1500;

    };

    template <>
    struct Thresholds<int, EigenIPC::RowMajor> {
        static constexpr double READ_T_MAX_THRESH = 10000000;
        static constexpr double WRITE_T_MAX_THRESH = 10000000;
        static constexpr double READ_TV_MAX_THRESH = 10000000;
        static constexpr double WRITE_TV_MAX_THRESH = 10000000;
        static constexpr double READ_T_AVRG_THRESH = 2300;
        static constexpr double WRITE_T_AVRG_THRESH = 2300;
        static constexpr double READ_TV_AVRG_THRESH = 2300;
        static constexpr double WRITE_TV_AVRG_THRESH = 2300;

    };

    template <>
    struct Thresholds<int, EigenIPC::ColMajor> {
        static constexpr double READ_T_MAX_THRESH = 10000000;
        static constexpr double WRITE_T_MAX_THRESH = 10000000;
        static constexpr double READ_TV_MAX_THRESH = 10000000;
        static constexpr double WRITE_TV_MAX_THRESH = 10000000;
        static constexpr double READ_T_AVRG_THRESH = 2300;
        static constexpr double WRITE_T_AVRG_THRESH = 2300;
        static constexpr double READ_TV_AVRG_THRESH = 2300;
        static constexpr double WRITE_TV_AVRG_THRESH = 2300;

    };

    template <>
    struct Thresholds<float, EigenIPC::RowMajor> {
        static constexpr double READ_T_MAX_THRESH = 10000000;
        static constexpr double WRITE_T_MAX_THRESH = 10000000;
        static constexpr double READ_TV_MAX_THRESH = 10000000;
        static constexpr double WRITE_TV_MAX_THRESH = 10000000;
        static constexpr double READ_T_AVRG_THRESH = 2500;
        static constexpr double WRITE_T_AVRG_THRESH = 2500;
        static constexpr double READ_TV_AVRG_THRESH = 2500;
        static constexpr double WRITE_TV_AVRG_THRESH = 2500;

    };

    template <>
    struct Thresholds<float, EigenIPC::ColMajor> {
        static constexpr double READ_T_MAX_THRESH = 10000000;
        static constexpr double WRITE_T_MAX_THRESH = 10000000;
        static constexpr double READ_TV_MAX_THRESH = 10000000;
        static constexpr double WRITE_TV_MAX_THRESH = 10000000;
        static constexpr double READ_T_AVRG_THRESH = 2500;
        static constexpr double WRITE_T_AVRG_THRESH = 2500;
        static constexpr double READ_TV_AVRG_THRESH = 2500;
        static constexpr double WRITE_TV_AVRG_THRESH = 2500;

    };

    template <>
    struct Thresholds<double, EigenIPC::RowMajor> {
        static constexpr double READ_T_MAX_THRESH = 10000000;
        static constexpr double WRITE_T_MAX_THRESH = 10000000;
        static constexpr double READ_TV_MAX_THRESH = 10000000;
        static constexpr double WRITE_TV_MAX_THRESH = 10000000;
        static constexpr double READ_T_AVRG_THRESH = 5000;
        static constexpr double WRITE_T_AVRG_THRESH = 5000;
        static constexpr double READ_TV_AVRG_THRESH = 5000;
        static constexpr double WRITE_TV_AVRG_THRESH = 5000;

    };

    template <>
    struct Thresholds<double, EigenIPC::ColMajor> {
        static constexpr double READ_T_MAX_THRESH = 10000000;
        static constexpr double WRITE_T_MAX_THRESH = 10000000;
        static constexpr double READ_TV_MAX_THRESH = 10000000;
        static constexpr double WRITE_TV_MAX_THRESH = 10000000;
        static constexpr double READ_T_AVRG_THRESH = 5000;
        static constexpr double WRITE_T_AVRG_THRESH = 5000;
        static constexpr double READ_TV_AVRG_THRESH = 5000;
        static constexpr double WRITE_TV_AVRG_THRESH = 5000;

    };

    template <typename Scalar, const int Layout>
    bool areEqual(const Tensor<Scalar, Layout>& A,
                  const Tensor<Scalar, Layout>& B) {

        return A.isApprox(B); // suitable for float and double

    }

    // Specialization for int
    template <>
    bool areEqual<int, ColMajor>(const Tensor<int, ColMajor>& A,
                  const Tensor<int, ColMajor>& B) {

        return A == B;

    }

    template <>
    bool areEqual<int, RowMajor>(const Tensor<int, RowMajor>& A,
                  const Tensor<int, RowMajor>& B) {

        return A == B;

    }

    // Specialization for bool
    template <>
    bool areEqual<bool, ColMajor>(const Tensor<bool, ColMajor>& A,
                  const Tensor<bool, ColMajor>& B) {

        return A == B;

    }

    template <>
    bool areEqual<bool, RowMajor>(const Tensor<bool, RowMajor>& A,
                  const Tensor<bool, RowMajor>& B) {

        return A == B;

    }

    bool allTrue(const std::vector<bool>& vec,
                 int& countFalse) {
        countFalse = 0;
        for (bool val : vec) {
            if (!val) {
                countFalse++;
            }
        }
        return countFalse == 0;
    }

}


#endif // TEST_UTILS_HPP
