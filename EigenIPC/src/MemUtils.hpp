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
#ifndef MEMUTILS_HPP
#define MEMUTILS_HPP

#include <Eigen/Dense>
#include <string>
#include <stdexcept>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <semaphore.h>
#include <csignal>
#include <memory>

#include <EigenIPC/DTypes.hpp>
#include <EigenIPC/Journal.hpp>
#include <EigenIPC/ReturnCodes.hpp>
#include <EigenIPC/Helpers.hpp>

namespace EigenIPC{

    namespace MemUtils{
        
        using VLevel = Journal::VLevel;

        using LogType = Journal::LogType;

        template <typename Scalar>
        struct IsValidDType {

            // Type trait to check if a given type is a valid DType
            static constexpr bool value =
                std::is_same<Scalar, typename DTypeToCppType<DType::Float>::type>::value ||
                std::is_same<Scalar, typename DTypeToCppType<DType::Double>::type>::value ||
                std::is_same<Scalar, typename DTypeToCppType<DType::Int>::type>::value ||
                std::is_same<Scalar, typename DTypeToCppType<DType::Bool>::type>::value;
        };

        inline std::string getLayoutName(int layout) {

            switch(layout) {

                case RowMajor:

                    return "EigenIPC::RowMajor";

                case ColMajor:

                    return "EigenIPC::ColMajor";

                default:

                    return "Unknown memory layout";
            }
        }

        inline void failWithCode(ReturnCode fail_code,
                            Journal journal,
                            std::string calling_fun,
                            std::string add_info = "") {

            std::string error = std::string("Failed with error code: ") +
                    toString(fail_code) + 
                    std::string(", which corresponds to ") +
                    getDescriptions(fail_code) + "\n" + 
                    std::string("Additional info: \n" + 
                    add_info) + "\n";
        
            // we throw an exception
            journal.log(calling_fun,
                        error,
                        LogType::EXCEP,
                        true);

        }

        // shared mem data utilities

        template <typename Scalar,
                  int Layout = MemLayoutDefault>
        void initMem(
            std::size_t n_rows,
            std::size_t n_cols,
            const std::string& mem_path,
            int& shm_fd,
            MMap<Scalar, Layout>& tensor_view,
            Journal& journal,
            ReturnCode& return_code,
            bool verbose = true,
            VLevel vlevel = Journal::VLevel::V0
            ){

            // Determine the size based on the Scalar type
            std::size_t data_size = sizeof(Scalar) * n_rows * n_cols;

            // Create shared memory
            shm_fd = shm_open(mem_path.c_str(),
                              O_CREAT | O_RDWR,
                              S_IRUSR | S_IWUSR);

            if (shm_fd == -1) {

                if (verbose) {

                    std::string error = "Could not create shared memory at " +
                            mem_path;

                    journal.log(__FUNCTION__,
                        error,
                        LogType::EXCEP);
                }

                return_code = return_code + ReturnCode::MEMCREATFAIL;

                return;

            }

            // Set size
            if (ftruncate(shm_fd, data_size) == -1) {

                if (verbose) {

                    std::string error = "Could not set shared memory at " +
                            mem_path;

                    journal.log(__FUNCTION__,
                                error,
                                LogType::EXCEP);
                }

                return_code = return_code + ReturnCode::MEMSETFAIL;

                return;

            }

            return_code = return_code + ReturnCode::MEMOPEN;

            if (verbose &&
                    vlevel > VLevel::V2) {

                std::string info = "Opened shared memory at " +
                        mem_path;

                journal.log(__FUNCTION__,
                            info,
                            LogType::INFO);

            }

            // Map the shared memory
            Scalar* matrix_data = static_cast<Scalar*>(mmap(nullptr,
                                                        data_size,
                                                        PROT_READ | PROT_WRITE,
                                                        MAP_SHARED,
                                                        shm_fd,
                                                        0));

            if (matrix_data == MAP_FAILED) {

                if (verbose) {
                    std::string map_error = "Could not map memory size for " +
                            mem_path;
                    journal.log(__FUNCTION__,
                                map_error,
                                LogType::EXCEP);
                }

                return_code = return_code + ReturnCode::MEMMAPFAIL;

                return;

            }

            new (&tensor_view) MMap<Scalar, Layout>(matrix_data,
                                           n_rows,
                                           n_cols); // contiguous memory
            // (no need to specify strides)

            return_code = return_code + ReturnCode::MEMMAP;

            if (verbose && vlevel > VLevel::V2) {

                std::string info = "Mapped shared memory at " +
                        mem_path;

                journal.log(__FUNCTION__,
                            info,
                            LogType::INFO);

            }

        }

        inline void cleanUpMem(
                        const std::string& mem_path,
                        int& shm_fd,
                        Journal& journal,
                        ReturnCode& return_code,
                        bool verbose = true,
                        VLevel vlevel = Journal::VLevel::V0,
                        bool unlink = false) {

            // Closing the file descriptor (for this process only)
            ::close(shm_fd);

            return_code = return_code + ReturnCode::MEMFDCLOSED;

            if (verbose
                    && vlevel > VLevel::V2) {

                std::string info = "Closed file descriptor for " +
                                    mem_path;

                journal.log(__FUNCTION__,
                             info,
                             LogType::INFO);
            }

            if (unlink) {

                // Unlinking from shared memory data (system-wide)
                // processed which have access to the memory can
                // still access it, but no new process can access it

                shm_unlink(mem_path.c_str());

                if (verbose &&
                        vlevel > VLevel::V2) {

                    std::string info = "Unlinked memory at " +
                                        mem_path;

                    journal.log(__FUNCTION__,
                                 info,
                                 LogType::INFO);
                }

                return_code = return_code + ReturnCode::MEMUNLINK;
            }

        }

        inline void checkMem(
                    const std::string& mem_path,
                    int& shm_fd,
                    Journal& journal,
                    ReturnCode& return_code,
                    bool verbose = true,
                    VLevel vlevel = Journal::VLevel::V0,
                    bool unlink = false) {

            shm_fd = shm_open(mem_path.c_str(),
                               O_RDWR,
                               0);

            if (shm_fd != -1) {
                // Shared memory already exists, so we need to clean it up

                if (verbose &&
                        vlevel > VLevel::V0) {

                    std::string warn = "Shared memory at " + mem_path +
                        " already exists. Clearing it up...";

                    journal.log(__FUNCTION__,
                                 warn,
                                 LogType::WARN);
                }

                return_code = return_code + ReturnCode::MEMEXISTS;

                cleanUpMem(mem_path,
                         shm_fd,
                         journal,
                         return_code,
                         verbose,
                         vlevel,
                         unlink);

                return_code = return_code + ReturnCode::MEMCLEAN;

                if (verbose &&
                        vlevel > VLevel::V0) {

                    std::string warn = "Cleanup Done.";

                    journal.log(__FUNCTION__,
                                 warn,
                                 LogType::WARN);

                }

                if (shm_fd != -1) {
                    // Close the file descriptor opened for checking existence

                    ::close(shm_fd);

                    return_code = return_code + ReturnCode::MEMFDCLOSED;

                }

            }

        }

        // read/write

        template <typename Scalar,
                  int Layout = MemLayoutDefault>
        bool write(const TRef<Scalar, Layout> data, // eigen reference (works also with blocks)
                   MMap<Scalar, Layout>& tensor_view,
                   int row, int col,
                   Journal& journal,
                   ReturnCode& return_code,
                   bool verbose = true,
                   VLevel vlevel = Journal::VLevel::V0) {

            bool success = helpers::canFitTensor(
                        tensor_view.rows(),
                        tensor_view.cols(),
                        row, col,
                        data.rows(), data.cols(),
                        journal,
                        return_code,
                        verbose,
                        vlevel);

            if (success) {

                tensor_view.block(row, col,
                              data.rows(),
                              data.cols()) = data;
            }

            if (!success) {

                return_code = return_code + ReturnCode::WRITEFAIL;
            }

            return success;

        }

        template <typename Scalar,
                  int Layout = MemLayoutDefault>
        bool write(const TensorView<Scalar, Layout>& data,
                   MMap<Scalar, Layout>& tensor_view,
                   int row, int col,
                   Journal& journal,
                   ReturnCode& return_code,
                   bool verbose = true,
                   VLevel vlevel = Journal::VLevel::V0) {

            bool success = helpers::canFitTensor(
                         tensor_view.rows(),
                         tensor_view.cols(),
                         row, col,
                         data.rows(), data.cols(),
                         journal,
                         return_code,
                         verbose,
                         vlevel);

            if (success) {

                tensor_view.block(row, col,
                              data.rows(),
                              data.cols()) = data;
            }

            if (!success) {

                return_code = return_code + ReturnCode::WRITEFAIL;
            }

            return success;

        }

        template <typename Scalar,
                  int Layout = MemLayoutDefault>
        bool read(int row, int col,
                  TRef<Scalar, Layout> output,
                  const MMap<Scalar, Layout>& tensor_view,
                  Journal& journal,
                  ReturnCode& return_code,
                  bool verbose = true,
                  VLevel vlevel = Journal::VLevel::V0) {

            bool success = helpers::canFitTensor(
                        tensor_view.rows(),
                        tensor_view.cols(),
                        row, col,
                        output.rows(), output.cols(),
                        journal,
                        return_code,
                        verbose,
                        vlevel);

            if (success) {

                output = tensor_view.block(row, col,
                                           output.rows(),
                                           output.cols());
            }

            if (!success) {

                return_code = return_code + ReturnCode::READFAIL;
            }

            return success;

        }

        template <typename Scalar,
                  int Layout = MemLayoutDefault>
        bool read(int row, int col,
                  TensorView<Scalar, Layout>& output,
                  const MMap<Scalar, Layout>& tensor_view,
                  Journal& journal,
                  ReturnCode& return_code,
                  bool verbose = true,
                  VLevel vlevel = Journal::VLevel::V0) {

            // copy data pointed from tensor_view
            // into the matrix output points
            bool success = helpers::canFitTensor(
                        tensor_view.rows(),
                        tensor_view.cols(),
                        row, col,
                        output.rows(), output.cols(),
                        journal,
                        return_code,
                        verbose,
                        vlevel);

            if (success) {

                output  = tensor_view.block(row, col,
                                           output.rows(),
                                           output.cols());
            }

            if (!success) {

                return_code = return_code + ReturnCode::READFAIL;
            }

            return success;

        }

        // semaphore stuff

        inline void semInit(const std::string& sem_path,
                    sem_t*& sem,
                    Journal& journal,
                    ReturnCode& return_code,
                    bool verbose = true,
                    VLevel vlevel = Journal::VLevel::V0) {

            sem = sem_open(sem_path.c_str(),
                                  O_CREAT, S_IRUSR | S_IWUSR,
                                  1); // initial val to 1 -> binary semaphore

            if (sem == SEM_FAILED) {
                // Handle semaphore creation error

                return_code = return_code + ReturnCode::SEMOPENFAIL;

                if (verbose) {

                    std::string error = std::string("Failed to open semaphore at ") +
                            sem_path;

                    journal.log(__FUNCTION__,
                        error,
                        LogType::WARN);

                }

                return;

            }
            else {

                return_code = return_code + ReturnCode::SEMOPEN;

                if (verbose &&
                    vlevel > VLevel::V2) {

                    std::string info = std::string("Opened semaphore at ") +
                            sem_path;

                    journal.log(__FUNCTION__,
                        info,
                        LogType::INFO);
                }
            }

        }

        inline void semClose(const std::string& sem_path,
                         sem_t *&sem,
                         Journal& journal,
                         ReturnCode& return_code,
                         bool verbose = true,
                         VLevel vlevel = Journal::VLevel::V0,
                         bool unlink = false) {

            sem_close(sem); // closes semaphore for the current process

            return_code = return_code + ReturnCode::SEMCLOSE;

            if (unlink) {

                sem_unlink(sem_path.c_str()); // unlinks semaphore system-wide.
                // Other processes who had it open can still use it, but no new
                // process can access it

                return_code = return_code + ReturnCode::SEMUNLINK;

                if (verbose &&
                        vlevel > VLevel::V2) {

                    std::string info = std::string("Closed and unlinked semaphore at ") +
                            sem_path;

                    journal.log(__FUNCTION__,
                         info,
                         LogType::INFO);

                }
            }

            if (!unlink) {

                if (verbose &&
                        vlevel > VLevel::V2) {

                    std::string info = std::string("Closed semaphore at ") +
                            sem_path;

                    journal.log(__FUNCTION__,
                         info,
                         LogType::INFO);

                }

            }

        }

        // sem release

        // free sem
        inline void releaseSem(const std::string& sem_path,
                         sem_t *&sem,
                         Journal& journal,
                         ReturnCode& return_code,
                         bool verbose = true,
                         VLevel vlevel = Journal::VLevel::V0) {

            if (verbose &&
                    vlevel > VLevel::V2) {

                std::string info = std::string("Releasing semaphore at ") +
                            sem_path;

                journal.log(__FUNCTION__,
                             info,
                             LogType::INFO);
            }

            // Release the semaphore
            if (sem_post(sem) == -1) {

                // Handle semaphore release error

                if (verbose) {

                    journal.log(__FUNCTION__,
                                "Failed to release semaphore at",
                                LogType::WARN);
                }

                return_code = return_code + ReturnCode::SEMRELFAIL;

                return;

            }

            return_code = return_code + ReturnCode::SEMREL;

            if (verbose &&
                    vlevel > VLevel::V2) {

                std::string info = std::string("Released semaphore at ") +
                        sem_path;

                journal.log(__FUNCTION__,
                             info,
                             LogType::INFO);

            }

        }

        // semaphore acquisition
        inline int semBlockingWait(sem_t* sem,
                        ReturnCode& return_code) {
            
            // this is blocking unless some signal is received

            return_code = return_code + ReturnCode::SEMACQTRY;

            int result = sem_wait(sem);

            if (result == 0) {

                // Successfully acquired the semaphore.

                return_code = return_code + ReturnCode::SEMACQ;

                return 0;

            } else { 
                
                if (errno == EINTR) {

                    return_code = return_code + ReturnCode::SEMACQSIGINT;

                } else {

                    return_code = return_code + ReturnCode::UNKNOWN;

                }

                return_code = return_code + ReturnCode::SEMACQFAIL;

                return -1;
            }

        }

        inline int semTimedWait(sem_t* sem,
                        struct timespec timeout,
                        ReturnCode& return_code) {
            
            // blocking wait with timeout

            clock_gettime(CLOCK_REALTIME, &timeout);

            return_code = return_code + ReturnCode::SEMACQTRY;

            int result = sem_timedwait(sem,
                                &timeout);

            if (result == 0) {

                // Successfully acquired the semaphore.

                return_code = return_code + ReturnCode::SEMACQ;

                return 0;

            } else { 
                
                if (errno == ETIMEDOUT) {
                    
                    return_code = return_code + ReturnCode::SEMACQTIMEOUT;

                } else if (errno == EINTR) {

                    return_code = return_code + ReturnCode::SEMACQSIGINT;

                } else {

                    return_code = return_code + ReturnCode::UNKNOWN;

                }

                return_code = return_code + ReturnCode::SEMACQFAIL;

                return -1;
            }

        }

        inline int SemTryWait(sem_t*& sem,
                         ReturnCode& return_code) {

            // this is nonblocking --> one-shot acquisition 

            // try to acquire the semaphore
            int result = sem_trywait(sem);

            return_code = return_code + ReturnCode::SEMACQTRY;

            if (result == 0) {
                
                // Successfully acquired the semaphore.

                return_code = return_code + ReturnCode::SEMACQ;

                return 0;

            } else {

                if (errno == EINTR) {

                    return_code = return_code + ReturnCode::SEMACQSIGINT;
            
                } else {

                    return_code = return_code + ReturnCode::UNKNOWN;

                }

                return_code = return_code + ReturnCode::SEMACQFAIL;

                return -1;
            
            }

        }

        // higher level semaphore acquisition wrappers

        inline void acquireSemBlocking(const std::string& sem_path,
                        sem_t*& sem,
                        Journal& journal,
                        ReturnCode& return_code,
                        bool verbose = true,
                        VLevel vlevel = Journal::VLevel::V0) {
            
            if (verbose &&
                    vlevel > VLevel::V2) {

                std::string info = std::string("Acquiring semaphore at ") +
                        sem_path;

                journal.log(__FUNCTION__,
                        info,
                        LogType::INFO);

            }

            if (semBlockingWait(sem,return_code) == -1) {
                
                if (verbose &&
                        vlevel > VLevel::V0) {

                    std::string exception = std::string("Semaphore acquisition at ") +
                            sem_path +
                            std::string("failed.");

                    journal.log(__FUNCTION__,
                                 exception,
                                 LogType::EXCEP,
                                 true // throw exception
                                 );
                    
                }

            }

        }

        inline void acquireSemTimeout(const std::string& sem_path,
                        sem_t*& sem,
                        Journal& journal,
                        ReturnCode& return_code,
                        struct timespec timeout,
                        bool force_reconnection = false,
                        bool verbose = true,
                        VLevel vlevel = Journal::VLevel::V0) {

            if (verbose &&
                    vlevel > VLevel::V2) {

                std::string info = std::string("Acquiring semaphore at ") +
                        sem_path;

                journal.log(__FUNCTION__,
                             info,
                             LogType::INFO);

            }   

            // Acquire the semaphore
            if (semTimedWait(sem, timeout,
                        return_code) == -1) {
                
                if (verbose &&
                        vlevel > VLevel::V0) {

                    std::string warn = std::string("Semaphore acquisition at ") +
                            sem_path + std::string(" timed out (") + 
                            std::to_string(timeout.tv_sec) + std::string(" s, ") +
                            std::to_string(timeout.tv_nsec) + std::string(" ns).");

                    journal.log(__FUNCTION__,
                                 warn,
                                 LogType::WARN);

                }

                if (force_reconnection) {
                    
                    return_code = return_code - ReturnCode::SEMACQFAIL; // removing acquisition fail
                    return_code = return_code - ReturnCode::SEMACQTIMEOUT;

                    // delete semaphore
                    semClose(sem_path,
                        sem, 
                        journal,
                        return_code,
                        verbose,
                        vlevel,
                        true 
                    );
                    // and recreate it
                    semInit(sem_path, 
                        sem, 
                        journal, 
                        return_code, 
                        verbose,
                        vlevel);

                    // recursive call. This will now work
                    acquireSemTimeout(sem_path,
                            sem,
                            journal,
                            return_code,
                            timeout,
                            force_reconnection,
                            verbose,
                            vlevel); 
                    
                }

            } else {

                if (verbose &&
                    vlevel > VLevel::V2) {

                    std::string info = std::string("Acquired semaphore at ") +
                            sem_path;

                    journal.log(__FUNCTION__,
                                info,
                                LogType::INFO);

                }
            }

        }
        
        inline void acquireSemOneShot(const std::string& sem_path,
                        sem_t*& sem,
                        Journal& journal,
                        ReturnCode& return_code,
                        bool verbose = true,
                        VLevel vlevel = Journal::VLevel::V0) {

            if (verbose &&
                    vlevel > VLevel::V2) {

                std::string info = std::string("Acquiring semaphore at ") +
                        sem_path;

                journal.log(__FUNCTION__,
                             info,
                             LogType::INFO);

            }

            // Acquire the semaphore
            if (SemTryWait(sem,
                        return_code) == -1) {

                return_code = return_code + ReturnCode::SEMACQFAIL;
                
                if (verbose &&
                        vlevel > VLevel::V0) {

                    std::string warn = std::string("Semaphore acquisition at ") +
                            sem_path + std::string("failed.");

                    journal.log(__FUNCTION__,
                                 warn,
                                 LogType::WARN);

                }

            }

            if (verbose &&
                    vlevel > VLevel::V2) {

                std::string info = std::string("Acquired semaphore at ") +
                        sem_path;

                journal.log(__FUNCTION__,
                             info,
                             LogType::INFO);

            }
            
        }


    }

}

#endif // MEMUTILS_HPP
