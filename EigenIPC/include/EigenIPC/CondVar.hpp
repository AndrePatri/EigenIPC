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

#ifndef CONDVAR_HPP
#define CONDVAR_HPP

#include <chrono>
#include <thread>
#include <memory>

#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

// public headers
#include <EigenIPC/SharedMemConfig.hpp>
#include <EigenIPC/Journal.hpp>
#include <EigenIPC/DTypes.hpp>
#include <EigenIPC/ReturnCodes.hpp>

namespace EigenIPC{

    class ConditionVariable{

        using VLevel = Journal::VLevel;
        using LogType = Journal::LogType;

        public:

            using NamedCondition = boost::interprocess::named_condition;
            using NamedMutex = boost::interprocess::named_mutex;
            using ScopedLock = boost::interprocess::scoped_lock<NamedMutex>;

            typedef std::weak_ptr<ConditionVariable> WeakPtr;
            typedef std::shared_ptr<ConditionVariable> Ptr;
            typedef std::unique_ptr<ConditionVariable> UniquePtr;

            ConditionVariable(bool is_server,
                    std::string basename,
                    std::string name_space = "",
                    bool verbose = false,
                    VLevel vlevel = VLevel::V0,
                   bool force_reconnection = false);

            ~ConditionVariable();

            static ScopedLock lock(NamedMutex& mutex);
            ScopedLock lock();

            static void unlock(ScopedLock& locked_lock);
            
            static NamedMutex create_named_mutex(std::string at);

            void wait(ScopedLock& named_lock);

            void wait_for(ScopedLock& named_lock, 
                    std::function<bool()> pred);

            bool timedwait(ScopedLock& named_lock,
                    unsigned int ms);
            
            bool timedwait_for(ScopedLock& named_lock,
                    unsigned int ms,
                    std::function<bool()> pred);

            void notify_one();

            void notify_all();
            
            void close();
            
            std::string mutex_path();
            std::string cond_var_path();

        private:

            bool _verbose = false;

            bool _is_server = false;
            
            bool _closed = false;
            
            bool _force_reconnection = false;

            std::string _basename, _namespace;
            
            std::string THISNAME = "EigenIPC::ConditionVariable";

            VLevel _vlevel = VLevel::V0; // minimal debug info

            Journal _journal; // for rt-friendly logging

            std::string _getThisName(); // used to get this class
            // name

            SharedMemConfig _mem_config;

            std::unique_ptr<NamedCondition> _cond_ptr;
            std::unique_ptr<NamedMutex> _mutex_ptr;

            boost::posix_time::ptime _utc_timeout;
            
            ScopedLock _lock;

            bool _cleanup_mem();
            

    };

}

#endif // CLIENT_HPP
