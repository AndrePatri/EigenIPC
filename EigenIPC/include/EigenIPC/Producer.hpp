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

#ifndef PRODUCER_HPP
#define PRODUCER_HPP

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
#include <EigenIPC/CondVar.hpp>
#include <EigenIPC/Server.hpp>

namespace EigenIPC{

    class Producer{

        using VLevel = Journal::VLevel;
        using LogType = Journal::LogType;
        using ConditionVariable = EigenIPC::ConditionVariable;
        using ScopedLock = ConditionVariable::ScopedLock;
        using SharedCounter = EigenIPC::Server<int>;
        using CounterView = EigenIPC::Tensor<int>;

        public:

            typedef std::weak_ptr<Producer> WeakPtr;
            typedef std::shared_ptr<Producer> Ptr;
            typedef std::unique_ptr<Producer> UniquePtr;

            Producer(std::string basename,
                    std::string name_space = "",
                    bool verbose = false,
                    VLevel vlevel = VLevel::V0,
                    bool force_reconnection = false);

            ~Producer();

            void run();
            
            void close();
            
            void trigger();

            bool wait_ack_from(int n_consumers,
                        int ms_timeout = -1);

        private:

            bool _verbose = false;

            bool _force_reconnection = false;

            bool _closed = false;
            bool _is_running = false;

            bool _ack_completed = false;

            bool _timeout = false;

            int _acks_before = 0;
            
            std::string _basename, _namespace, _unique_id;
            
            std::string THISNAME = "EigenIPC::Producer";
            
            std::string TRIGGER_COND_NAME = "TriggerCond";
            std::string ACK_COND_NAME = "AckCond";

            std::string TRIGGER_BASENAME = "Trigger";
            std::string ACK_BASENAME = "Ack";

            VLevel _vlevel = VLevel::V0; // minimal debug info

            Journal _journal; // for rt-friendly logging

            ConditionVariable::UniquePtr _trigger_cond_ptr;
            ConditionVariable::UniquePtr _ack_cond_ptr;

            SharedCounter _trigger_counter_srvr;
            CounterView _trigger_counter;

            SharedCounter _ack_counter_srvr;
            CounterView _ack_counter; 

            std::string _getThisName(); // used to get this class
            // name

            void _create_cond_vars();

            void _check_running(std::string calling_method);

            void _init_counters();

            void _increment_trigger();

            bool _check_ack_counter(int n_consumers);

            bool _wait(ScopedLock& ack_lock, int ms_timeout = -1);

    };

}

#endif // PRODUCER_HPP
