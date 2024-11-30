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

#ifndef SERVER_HPP
#define SERVER_HPP

#include <Eigen/Dense>
#include <string>
#include <stdexcept>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <csignal>
#include <memory>

// public headers
#include <EigenIPC/SharedMemConfig.hpp>
#include <EigenIPC/Journal.hpp>
#include <EigenIPC/DTypes.hpp>
#include <EigenIPC/ReturnCodes.hpp>

namespace EigenIPC{

    template <typename Scalar,
              int Layout = MemLayoutDefault>
    class Server {
        
        using VLevel = Journal::VLevel;
        using LogType = Journal::LogType;

        public:

            typedef std::weak_ptr<Server> WeakPtr;
            typedef std::shared_ptr<Server> Ptr;
            typedef std::unique_ptr<Server> UniquePtr;

            Server(int n_rows,
                   int n_cols,
                   std::string basename = "MySharedMemory",
                   std::string name_space = "",
                   bool verbose = false,
                   VLevel vlevel = VLevel::V0,
                   bool force_reconnection = false,
                   bool safe = true);

            ~Server();

            bool write(const TRef<Scalar, Layout> data,
                             int row = 0,
                             int col = 0
                             );

            bool write(const TensorView<Scalar, Layout>& data,
                             int row,
                             int col
                             );

            bool read(TRef<Scalar, Layout> output,
                            int row = 0, int col = 0
                            ); // copies underlying shared tensor data to the output

            bool read(TensorView<Scalar, Layout> &output,
                            int row = 0, int col = 0
                            ); // copies underlying shared tensor data to the output Map

            void run();
            void stop();
            void close();

            bool isRunning();

            int getNClients();

            int getNRows();
            int getNCols();

            DType getScalarType() const;

            int getMemLayout() const;

            std::string getNamespace() const;
            std::string getBasename() const;

            void dataSemAcquire(); // acquire data sem (blocking)
            void dataSemRelease(); // acquire data sem (blocking)
            
        protected:

            bool _unlink_data = true; // will also unlink data
            // when freeing shared memory

            bool _verbose = false;

            bool _safe = false;
            
            bool _terminated = false;

            bool _running = false;

            bool _force_reconnection = false;

            bool _data_acquired = false; // aux. variable,
            // preallocated for efficiency

            int _n_rows = -1;
            int _n_cols = -1;

            int _n_clients = -1;

            int _data_shm_fd; // shared memory file descriptor
            int _nrows_shm_fd,
                _ncols_shm_fd,
                _n_clients_shm_fd,
                _dtype_shm_fd,
                _isrunning_shm_fd,
                _mem_layout_shm_fd;

            static const int _mem_layout = Layout;

            std::string THISNAME = "EigenIPC::Server";
            std::string _basename, _namespace;

            float _sem_acq_timeout = 0.0001;

            struct timespec _sem_timeout;

            VLevel _vlevel = VLevel::V0; // minimal debug info

            SharedMemConfig _mem_config;

            sem_t* _srvr_sem; // semaphore for servers uniqueness
            sem_t* _data_sem; // semaphore for safe data access

            Journal _journal; // for rt-friendly logging

            ReturnCode _return_code = ReturnCode::NONE; // overwritten by all methods
            // this is to avoid dyn. allocation

            Tensor<Scalar, Layout> _tensor_copy; // copy (not view) of the tensor

            MMap<Scalar, Layout> _tensor_view; // view of the tensor
            // auxiliary views
            MMap<int, Layout> _n_rows_view,
                      _n_cols_view,
                      _n_clients_view,
                      _dtype_view,
                      _mem_layout_view;
            MMap<bool, Layout> _isrunning_view;

            std::string _getThisName();

            void _initDataMem();
            void _initMetaMem();

            void _initSems();

            void _acquireSemTimeout(const std::string& sem_path,
                            sem_t*& sem,
                            bool verbose = false);

            bool _acquireSemOneShot(const std::string& sem_path,
                        sem_t*& sem);
            
            void _acquireSemBlocking(const std::string& sem_path,
                            sem_t*& sem,
                            bool verbose = false);

            void _releaseSem(const std::string& sem_path,
                        sem_t*& sem,
                        bool verbose = false);

            bool _acquireData(bool blocking = false,
                                bool verbose = false);
            void _releaseData();

            void _closeSems();

            void _cleanMetaMem();
            void _cleanMems();

            void _checkIsRunning();

    };

}

#endif // SERVER_HPP

