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
#ifndef STRINGTENSOR_HPP
#define STRINGTENSOR_HPP

#include <vector>
#include <array>

#include <string>
#include <thread>

#include <EigenIPC/Client.hpp>
#include <EigenIPC/Server.hpp>

#include <EigenIPC/Journal.hpp>

namespace EigenIPC {

        // uses default memory layout
        using StrServer = Server<int>;
        using StrClient = Client<int>;

        template <typename ShMemType>
        class StringTensor {
        
        // Wrapper on top of Server and Client to also share
        // tensor (list) of strings. 
        // The idea is to use a shared tensor of ints with the following properties:
        // Each string in the list will be encoded along the columns of
        // the tensor. Given an input string, the string is first split
        // into chunks of size equal to size_of(int), which is typically 4 bytes or 32 bits.
        // This means, for most systems, we can represent each chunk of a string as a sequence
        // of up to 4 characters (UTF-8 encoded). 
        // Each chunck can then distributed along the column of the tensor, at 
        // the corresponding column index. Strings are encoded using UTF8 and 
        // then decoded with the same logic.

        using VLevel = Journal::VLevel;

        public:

            typedef std::weak_ptr<StringTensor> WeakPtr;
            typedef std::shared_ptr<StringTensor> Ptr;
            typedef std::unique_ptr<StringTensor> UniquePtr;

            StringTensor(std::string basename = "MySharedMemory",
                         std::string name_space = "",
                         bool verbose = false,
                         VLevel vlevel = VLevel::V0,
                         bool safe = true); // used when client

            StringTensor(int lenght,
                         std::string basename = "MySharedMemory",
                         std::string name_space = "",
                         bool verbose = false,
                         VLevel vlevel = VLevel::V0,
                         bool force_reconnection = false,
                         bool safe = false); // used when server

            ~StringTensor();

            void run();

            bool write(const std::vector<std::string>& vec,
                       int index = 0); // to be preferred,
            // more efficient

            bool write(const std::string& str,
                       int index = 0); // less efficient if writing
            // multiple strings

            bool read(std::vector<std::string>& vec,
                      int index = 0);

            bool read(std::string& str,
                      int index = 0);

            void close();

            bool isRunning();

            bool isServer();

            int getNClients();

            int getLength();

            ShMemType& getSharedMem();

            std::string getNamespace() const;
            std::string getBasename() const;

            Tensor<int> get_raw_buffer(); // gets a COPY of the raw buffer

        private:

            int _length = -1; // string tensor length

//            constexpr int MAX_THREADS = std::thread::hardware_concurrency() ?
//                        std::thread::hardware_concurrency() : 4; // fallback to 4 threads

            bool _running = false;

            bool _is_server = false;
            
            // assuming maximum characters in a string are max_chars
            static constexpr int _max_chars = 1024;

            // max number of rows needed
            static constexpr int _n_rows = _max_chars / sizeof(int);

            int _tmp_value = 0; // tmp val to avoid dyn allocations

            Tensor<int> _buffer; // to avoid dyn. allocations

            ShMemType _sh_mem;

//            std::array<std::thread, MAX_THREADS> threads; // preallocate threads

            // Helper methods to initialize _sh_mem
            ShMemType _initServer(int length,
                                  std::string basename,
                                  std::string name_space,
                                  bool verbose,
                                  VLevel vlevel,
                                  bool force_reconnection,
                                  bool safe);

            ShMemType _initClient(std::string basename,
                                  std::string name_space,
                                  bool verbose,
                                  VLevel vlevel,
                                  bool safe);

            bool _fits(const std::vector<std::string>& vec,
                       int index);

            void _encode_str(const std::string& str,
                             int col_index);

            void _decode_str(std::string& str,
                             int col_index);

            bool _encode_vec(const std::vector<std::string>& vec,
                             int col_index);

            bool _decode_vec(std::vector<std::string>& vec,
                           int col_index);

        };

}

#endif // STRINGTENSOR_HPP
