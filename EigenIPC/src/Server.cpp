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
#include <iostream>
#include <Eigen/Dense>
#include <cstring>
#include <typeinfo>
#include <ctime>

#include <EigenIPC/Server.hpp>

// private headers
#include <MemUtils.hpp>

namespace EigenIPC {

    template <typename Scalar, int Layout>
    Server<Scalar, Layout>::Server(int n_rows,
                   int n_cols,
                   std::string basename,
                   std::string name_space,
                   bool verbose,
                   VLevel vlevel,
                   bool force_reconnection,
                   bool safe)
        : _n_rows(n_rows),
        _n_cols(n_cols),
        _mem_config(basename, name_space),
        _basename(basename), _namespace(name_space),
        _verbose(verbose),
        _vlevel(vlevel),
        _safe(safe),
        _force_reconnection(force_reconnection),
        _tensor_view(nullptr,
                    n_rows,
                    n_cols),
        _n_rows_view(nullptr,
                    1,
                    1),
        _n_cols_view(nullptr,
                    1,
                    1),
        _n_clients_view(nullptr,
                        1,
                        1),
        _dtype_view(nullptr,
                    1,
                    1),
        _isrunning_view(nullptr,
                    1,
                    1),
        _mem_layout_view(nullptr,
                    1,
                    1),
        _journal(Journal(_getThisName()))
    {

        static_assert(MemUtils::IsValidDType<Scalar>::value, "Invalid data type provided.");

        if (_force_reconnection &&
                _verbose &&
                _vlevel > VLevel::V1)
        {
            std::string warn = std::string("Server at ") + _mem_config.mem_path + 
                    std::string(" will be initialized with force_reconnection to true. ") +
                    std::string("This can cause destructive behaviour if trying to run two servers concurrently on the ") +
                    std::string("same memory.");

            _journal.log(__FUNCTION__,
                warn,
                LogType::WARN);
        }

        if (_verbose &&
            _vlevel > VLevel::V1) {

            std::string info = std::string("Initializing Server at ") +
                    _mem_config.mem_path;

            _journal.log(__FUNCTION__,
                info,
                LogType::STAT);

        }
        
        // sem acquisition timeout settings
        long timeoutInNanoseconds = (long)(_sem_acq_timeout * 1e9);
        _sem_timeout.tv_sec = 0;
        _sem_timeout.tv_nsec = timeoutInNanoseconds % 1000000000;

        _initSems(); // creates necessary semaphores

        _acquireSemTimeout(_mem_config.mem_path_data_sem,
                        _data_sem,
                        _verbose); 

        _return_code = _return_code + ReturnCode::RESET; // resets to None

        MemUtils::checkMem(_mem_config.mem_path,
                            _data_shm_fd,
                            _journal,
                            _return_code,
                            _verbose,
                            _vlevel,
                            _unlink_data); // checks if memory was already allocated
        // if yes, cleans it up

        _return_code = _return_code + ReturnCode::RESET;

        // data memory
        _initDataMem();

        // auxiliary data
        _initMetaMem();

        _tensor_copy = Tensor<Scalar, Layout>::Zero(_n_rows,
                                            _n_cols); // used to hold
        // a copy of the shared tensor data

        _terminated = false; // just in case

        if (_verbose &&
            _vlevel > VLevel::V1) {

            std::string info = std::string("Server at ") +
                    _mem_config.mem_path + std::string(" initialized. Ready to run");

            _journal.log(__FUNCTION__,
                info,
                LogType::STAT);

        }

    }

    template <typename Scalar, int Layout>
    Server<Scalar, Layout>::~Server() {

        if (!_terminated) {

            close();
        }

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::_checkIsRunning()
    {
        if (!_running && _verbose) {

            std::string error = std::string("Server ") + 
                    _mem_config.mem_path +
                    std::string(" is not running. ") +
                    std::string("Did you remember to call the run() method?");

            _journal.log(__FUNCTION__,
                 error,
                 LogType::EXCEP); // nonblocking

        }
    }
    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::run()
    {

        if (!isRunning()) {
            
            _acquireSemTimeout(_mem_config.mem_path_server_sem,
                        _srvr_sem,
                        _verbose); // blocking. from this point on, 
            // other servers trying to transition to running state will fail
            // due to the sever semaphore being acquired

            _releaseSem(_mem_config.mem_path_data_sem,
                        _data_sem,
                        _verbose); // data can now be acquired by clients

            // set the running flag to true
            _running = true;
            _isrunning_view(0, 0) = 1; // for the clients

            if (_verbose &&
                _vlevel > VLevel::V1) {

                std::string info = std::string("Server at ") +
                        _mem_config.mem_path + std::string(" transitioned to running state.");

                _journal.log(__FUNCTION__,
                    info,
                    LogType::STAT);

            }

        }

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::stop()
    {

        if (isRunning()) {
            
            _running = false;
            _isrunning_view(0, 0) = 0; // for the clients

            MemUtils::releaseSem(_mem_config.mem_path_server_sem,
                                _srvr_sem,
                                _journal,
                                _return_code,
                                _verbose,
                                _vlevel);

        }

    }

    template <typename Scalar, int Layout>
    bool Server<Scalar, Layout>::isRunning()
    {

        return _running;

    }

    template <typename Scalar, int Layout>
    int Server<Scalar, Layout>::getNRows()
    {

        return _n_rows;

    }

    template <typename Scalar, int Layout>
    int Server<Scalar, Layout>::getNCols()
    {

        return _n_cols;

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::close()
    {

        stop(); // stop server if running

        _cleanMems(); // cleans up all memory,
        // semaphores included (if necessary)

        if (_verbose &&
            _vlevel > VLevel::V1) {

            std::string info = std::string("Closed server at ") +
                    _mem_config.mem_path;

            _journal.log(__FUNCTION__,
                 info,
                 LogType::STAT);

        }
    }

    template <typename Scalar, int Layout>
    int Server<Scalar, Layout>::getNClients() {

        _acquireData(true, false);

        _n_clients = _n_clients_view(0, 0);

        _releaseData();

        return _n_clients;
    }

    template <typename Scalar, int Layout>
    DType Server<Scalar, Layout>::getScalarType() const {

        return CppTypeToDType<Scalar>::value;

    }

    template <typename Scalar, int Layout>
    int Server<Scalar, Layout>::getMemLayout() const {

        return _mem_layout;

    }

    template <typename Scalar, int Layout>
    std::string Server<Scalar, Layout>::getNamespace() const {

        return _namespace;

    }

    template <typename Scalar, int Layout>
    std::string Server<Scalar, Layout>::getBasename() const {

        return _basename;

    }

    template <typename Scalar, int Layout>
    bool Server<Scalar, Layout>::write(const TRef<Scalar, Layout> data,
                                 int row,
                                 int col) {

        if (_running) {

            _data_acquired = true;

            if (_safe) {

                // first acquire data semaphore
                _data_acquired = _acquireData(false, false);
            }

            if(_data_acquired) {

                bool success_write = MemUtils::write<Scalar, Layout>(
                                        data,
                                        _tensor_view,
                                        row, col,
                                        _journal,
                                        _return_code,
                                        false,
                                        _vlevel);

                if (_safe) {
                    _releaseData();
                }

                return success_write;

            } else {

                return false; // failed to acquire sem
            }

        }

        _checkIsRunning();

        return false;

    }

    template <typename Scalar, int Layout>
    bool Server<Scalar, Layout>::write(const TensorView<Scalar, Layout>& data,
                                     int row,
                                     int col) {

        if (_running) {

            _data_acquired = true;

            if (_safe) {

                // first acquire data semaphore
                _data_acquired = _acquireData(false, false);
            }

            if(_data_acquired) {

                bool success_write = MemUtils::write<Scalar, Layout>(
                                        data,
                                        _tensor_view,
                                        row, col,
                                        _journal,
                                        _return_code,
                                        false,
                                        _vlevel);

                if (_safe) {
                    _releaseData();
                }

                return success_write;

            } else {

                return false; // failed to acquire sem
            }

        }

        _checkIsRunning();

        return false;

    }

    template <typename Scalar, int Layout>
    bool Server<Scalar, Layout>::read(TRef<Scalar, Layout> output,
                                    int row, int col) {

        if (_running) {

            _data_acquired = true;

            if (_safe) {

                // first acquire data semaphore
                _data_acquired = _acquireData(false, false);
            }

            if(_data_acquired) {

                bool success_read = MemUtils::read<Scalar, Layout>(
                            row, col,
                            output,
                            _tensor_view,
                            _journal,
                            _return_code,
                            false,
                            _vlevel);

                if (_safe) {
                    _releaseData();
                }

                return success_read;

            } else {

                return false; // failed to acquire sem
            }

        }

        _checkIsRunning();

        return false;

    }

    template <typename Scalar, int Layout>
    bool Server<Scalar, Layout>::read(TensorView<Scalar, Layout>& output,
                                    int row, int col) {

        if (_running) {

            _data_acquired = true;

            if (_safe) {

                // first acquire data semaphore
                _data_acquired = _acquireData(false, false);
            }

            if(_data_acquired) {

                bool success_read = MemUtils::read<Scalar, Layout>(
                                       row, col,
                                       output,
                                       _tensor_view,
                                       _journal,
                                       _return_code,
                                       false,
                                       _vlevel);
                
                if (_safe) {
                    _releaseData();
                }
                
                return success_read;

            } else {

                return false; // failed to acquire sem

            }

        }

        _checkIsRunning();

        return false;

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::dataSemAcquire() 
    {

        _acquireSemBlocking(_mem_config.mem_path_data_sem,
                    _data_sem,
                    _verbose);

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::dataSemRelease() 
    {

        _releaseSem(_mem_config.mem_path_data_sem,
                    _data_sem,
                    _verbose);

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::_acquireSemTimeout(const std::string& sem_path,
                                    sem_t*& sem,
                                    bool verbose)
    {
        _return_code = _return_code + ReturnCode::RESET;

        MemUtils::acquireSemTimeout(sem_path,
                        sem,
                        _journal,
                        _return_code,
                        _sem_timeout, // [s]
                        _force_reconnection,
                        verbose,
                        _vlevel);

        if (isin(ReturnCode::SEMACQFAIL, _return_code)) {

            MemUtils::failWithCode(_return_code,
                                   _journal,
                                   __FUNCTION__,
                                   sem_path);

        }

        _return_code = _return_code + ReturnCode::RESET;

    }

    template <typename Scalar, int Layout>
    bool Server<Scalar, Layout>::_acquireSemOneShot(const std::string& sem_path,
                                     sem_t*& sem)
    {
        _return_code = _return_code + ReturnCode::RESET;

        MemUtils::acquireSemOneShot(sem_path,
                             sem,
                             _journal,
                             _return_code,
                             _verbose,
                             VLevel::V0); // minimal verbosity (if enabled at all)

        if (isin(ReturnCode::SEMACQFAIL,
                 _return_code)) {

            return false;

        }

        _return_code = _return_code + ReturnCode::RESET;


        return true;

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::_acquireSemBlocking(const std::string& sem_path,
                                    sem_t*& sem,
                                    bool verbose)
    {
        _return_code = _return_code + ReturnCode::RESET;

        MemUtils::acquireSemBlocking(sem_path,
                        sem,
                        _journal,
                        _return_code,
                        verbose,
                        _vlevel);

        if (isin(ReturnCode::SEMACQFAIL, _return_code)) {

            MemUtils::failWithCode(_return_code,
                                   _journal,
                                   __FUNCTION__);

        }

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::_releaseSem(const std::string& sem_path,
                                    sem_t*& sem,
                                    bool verbose)
    {
        _return_code = _return_code + ReturnCode::RESET;


        MemUtils::releaseSem(sem_path,
                        sem,
                        _journal,
                        _return_code,
                        verbose, // no verbosity (this is called very frequently)
                        _vlevel);

        _return_code = _return_code + ReturnCode::RESET;


        if (isin(ReturnCode::SEMRELFAIL, _return_code)) {

            MemUtils::failWithCode(_return_code,
                                   _journal,
                                   __FUNCTION__);

        }

    }

    template <typename Scalar, int Layout>
    bool Server<Scalar, Layout>::_acquireData(bool blocking,
                            bool verbose)
    {

        if (blocking) {

            _acquireSemBlocking(_mem_config.mem_path_data_sem,
                            _data_sem,
                            verbose); // this is blocking

            return true;

        }
        else {

            return _acquireSemOneShot(_mem_config.mem_path_data_sem,
                        _data_sem);

        }

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::_releaseData()
    {

        _releaseSem(_mem_config.mem_path_data_sem,
                    _data_sem,
                    false // no verbosity (called frequently)
                    );

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::_cleanMetaMem()
    {
        // closing file descriptors and also unlinking
        // memory

        _return_code = _return_code + ReturnCode::RESET;


        MemUtils::cleanUpMem(_mem_config.mem_path_nrows,
                             _nrows_shm_fd,
                             _journal,
                             _return_code,
                             _verbose,
                             _vlevel,
                             _unlink_data);

        MemUtils::cleanUpMem(_mem_config.mem_path_ncols,
                             _ncols_shm_fd,
                             _journal,
                             _return_code,
                             _verbose,
                             _vlevel,
                             _unlink_data);

        MemUtils::cleanUpMem(_mem_config.mem_path_clients_counter,
                             _n_clients_shm_fd,
                             _journal,
                             _return_code,
                             _verbose,
                             _vlevel,
                             _unlink_data);

        MemUtils::cleanUpMem(_mem_config.mem_path_dtype,
                             _dtype_shm_fd,
                             _journal,
                             _return_code,
                             _verbose,
                             _vlevel,
                             _unlink_data);

        MemUtils::cleanUpMem(_mem_config.mem_path_mem_layout,
                             _mem_layout_shm_fd,
                             _journal,
                             _return_code,
                             _verbose,
                             _vlevel,
                             _unlink_data);

        _return_code = _return_code + ReturnCode::RESET;


    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::_cleanMems()
    {

        if (!_terminated) {

            _return_code = _return_code + ReturnCode::RESET;


            MemUtils::cleanUpMem(_mem_config.mem_path,
                                 _data_shm_fd,
                                 _journal,
                                 _return_code,
                                 _verbose,
                                 _vlevel);

            _return_code = _return_code + ReturnCode::RESET;


            _cleanMetaMem();

            _closeSems(); // closing semaphores

            if (_verbose &&
                _vlevel > VLevel::V1) {

                std::string info = std::string("Cleaning after server at ") +
                        _mem_config.mem_path;

                _journal.log(__FUNCTION__,
                     info,
                     LogType::STAT);

            }

            _terminated = true;

        }

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::_initMetaMem()
    {
        _return_code = _return_code + ReturnCode::RESET;
 // resets return code

        MemUtils::initMem<int>(1,
                        1,
                        _mem_config.mem_path_nrows,
                        _nrows_shm_fd,
                        _n_rows_view,
                        _journal,
                        _return_code,
                        _verbose,
                        _vlevel);

        MemUtils::initMem<int>(1,
                        1,
                        _mem_config.mem_path_ncols,
                        _ncols_shm_fd,
                        _n_cols_view,
                        _journal,
                        _return_code,
                        _verbose,
                        _vlevel);

        MemUtils::initMem<int>(1,
                        1,
                        _mem_config.mem_path_clients_counter,
                        _n_clients_shm_fd,
                        _n_clients_view,
                        _journal,
                        _return_code,
                        _verbose,
                        _vlevel);

        MemUtils::initMem<int>(1,
                        1,
                        _mem_config.mem_path_dtype,
                        _dtype_shm_fd,
                        _dtype_view,
                        _journal,
                        _return_code,
                        _verbose,
                        _vlevel);

        MemUtils::initMem<bool>(1,
                        1,
                        _mem_config.mem_path_isrunning,
                        _isrunning_shm_fd,
                        _isrunning_view,
                        _journal,
                        _return_code,
                        _verbose,
                        _vlevel);

        MemUtils::initMem<int>(1,
                        1,
                        _mem_config.mem_path_mem_layout,
                        _mem_layout_shm_fd,
                        _mem_layout_view,
                        _journal,
                        _return_code,
                        _verbose,
                        _vlevel);

        if (!isin(ReturnCode::MEMCREATFAIL,
                _return_code) &&
            !isin(ReturnCode::MEMSETFAIL,
                _return_code) &&
            !isin(ReturnCode::MEMMAPFAIL,
                 _return_code)) {

            // all memory creations where successful

            _n_rows_view(0, 0) = _n_rows;
            _n_cols_view(0, 0) = _n_cols;
            _n_clients_view(0, 0) = 0; // to be improved
            // (what happens when server crashes and clients remain appended?)
            _isrunning_view(0, 0) = 0;

            _mem_layout_view(0, 0) = _mem_layout; // mem layout

            _dtype_view(0, 0) = sizeof(Scalar);

            _return_code = _return_code + ReturnCode::RESET;

        }
        else {

            MemUtils::failWithCode(_return_code,
                                   _journal,
                                   __FUNCTION__);

        }

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::_initDataMem()
    {

        _return_code = _return_code + ReturnCode::RESET;

        if (!isin(ReturnCode::MEMCREATFAIL,
                _return_code) &&
            !isin(ReturnCode::MEMSETFAIL,
                _return_code) &&
            !isin(ReturnCode::MEMMAPFAIL,
                 _return_code)) {

            MemUtils::initMem<Scalar, Layout>(
                            _n_rows,
                            _n_cols,
                            _mem_config.mem_path,
                            _data_shm_fd,
                            _tensor_view,
                            _journal,
                            _return_code,
                            _verbose,
                            _vlevel);

            _return_code = _return_code + ReturnCode::RESET;

        }
        else {

            MemUtils::failWithCode(_return_code,
                                   _journal,
                                   __FUNCTION__);

        }

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::_initSems()
    {
        _return_code = _return_code + ReturnCode::RESET;

        MemUtils::semInit(_mem_config.mem_path_server_sem,
                          _srvr_sem,
                          _journal,
                          _return_code,
                          _verbose,
                          _vlevel);

        if (isin(ReturnCode::SEMOPENFAIL, _return_code)) {

            MemUtils::failWithCode(_return_code,
                                   _journal,
                                   __FUNCTION__,
                                   _mem_config.mem_path_server_sem);
        }

        _return_code = _return_code + ReturnCode::RESET;

        MemUtils::semInit(_mem_config.mem_path_data_sem,
                          _data_sem,
                          _journal,
                          _return_code,
                          _verbose,
                          _vlevel);

        if (isin(ReturnCode::SEMOPENFAIL, _return_code)) {

            MemUtils::failWithCode(_return_code,
                                   _journal,
                                   __FUNCTION__,
                                   _mem_config.mem_path_data_sem);
        }

        _return_code = _return_code + ReturnCode::RESET;

    }

    template <typename Scalar, int Layout>
    void Server<Scalar, Layout>::_closeSems()
    {
        // closes semaphores and also unlinks it
        // Other processes who had it open can still use it, but no new
        // process can access it
        MemUtils::semClose(_mem_config.mem_path_server_sem,
                           _srvr_sem,
                           _journal,
                           _return_code,
                           _verbose,
                           _vlevel,
                           true);

        MemUtils::semClose(_mem_config.mem_path_data_sem,
                           _data_sem,
                           _journal,
                           _return_code,
                           _verbose,
                           _vlevel,
                           true);

    }

    template <typename Scalar, int Layout>
    std::string Server<Scalar, Layout>::_getThisName()
    {

        return THISNAME;
    }

    // explicit instantiations for specific supported types
    // and layouts
    template class Server<double, ColMajor>;
    template class Server<float, ColMajor>;
    template class Server<int, ColMajor>;
    template class Server<bool, ColMajor>;

    template class Server<double, RowMajor>;
    template class Server<float, RowMajor>;
    template class Server<int, RowMajor>;
    template class Server<bool, RowMajor>;
}
