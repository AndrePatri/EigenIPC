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

#include <EigenIPC/Consumer.hpp>

namespace EigenIPC {

    Consumer::Consumer(
                std::string basename,
                std::string name_space,
                bool verbose,
                VLevel vlevel)
        : _verbose(verbose),
        _vlevel(vlevel),
        _journal(Journal(_getThisName())),
        _trigger_counter_clnt(basename + TRIGGER_BASENAME, 
            name_space,
            verbose,
            vlevel,
            false),
        _trigger_counter(1, 1),
        _ack_counter_clnt(basename + ACK_BASENAME, 
            name_space,
            verbose,
            vlevel,
            false),
        _ack_counter(1, 1),
        _closed(true),
        _basename(basename),
        _namespace(name_space),
        _unique_id(std::string("->")+ basename+std::string("-")+name_space)
    {

    }

    Consumer::~Consumer(){
        
        close();
    }

    void Consumer::run() {

        if (!_is_running) {
            
            _trigger_counter_clnt.attach();
            _ack_counter_clnt.attach();
            
            _open_cond_vars(); // we open the condition variables
            // only after client attachment has succeded (this guarantees mutexes
            // and cond. vars where created by the producer)

            _is_running = true;
            _closed = false;

            _internal_trigger_counter = 0;

            if (_verbose &&
                _vlevel > VLevel::V1) {

                _journal.log(__FUNCTION__+_unique_id,
                    "Transitioned to running state.",
                    LogType::STAT);

            }
        }
    }

    void Consumer::close() {

        if (!_closed) {
            
            _trigger_counter_clnt.close();
            _ack_counter_clnt.close();

            _closed = true;
        }
    }

    bool Consumer::wait(int ms_timeout) {

        _check_running(std::string(__FUNCTION__));

        ScopedLock trigger_lock = _trigger_cond_ptr->lock();

        _trigger_received = false;

        while (!_trigger_received) {

            _trigger_received = _check_trigger_received();

            if (!_trigger_received) {
                
                // wait (unlock and lock mutex atomically when
                // returning)
                if(!_wait(trigger_lock, ms_timeout)) {

                    return false;
                }

            }
        }

        return true;
    }

    bool Consumer::wait_and_ack(std::function<bool()> pre_ack,
                    int ms_timeout) {

        _fail_count = 0;
        
        if (!wait(ms_timeout)) {

            _fail_count += 1;

        } else {

            if (!pre_ack()) {

                _fail_count += 1;
            }

            if (!ack()) { 

                _fail_count += 1;

            }

        } 

        return _fail_count == 0;
    }

    bool Consumer::ack() {

        _check_running(std::string(__FUNCTION__));

        ScopedLock ack_lock = _ack_cond_ptr->lock();

        return _acknowledge();

    }

    bool Consumer::_acknowledge() {
        
        _fail_count = 0;

        if (!_ack_counter_clnt.read(_ack_counter, 0, 0)) {

            _journal.log(__FUNCTION__+_unique_id,
                "Could not read acknowledge counter!",
                LogType::EXCEP, 
                false); // throw exception

            _fail_count += 1;

        }

        _ack_counter.array() += 1; // increment shared ack counter and write to memory
        
        if (!_ack_counter_clnt.write(_ack_counter, 0, 0)) {

            _journal.log(__FUNCTION__+_unique_id,
                "Could not write acknowledge counter!",
                LogType::EXCEP, 
                false); // throw exception

            _fail_count += 1;
            
        }

        _ack_cond_ptr->notify_one();

        return _fail_count == 0;
        
    }

    void Consumer::_open_cond_vars() {
        
        bool is_server = false; // this is a consumer
        _trigger_cond_ptr = std::make_unique<ConditionVariable>(is_server,
                _basename + TRIGGER_COND_NAME,
                _namespace,
                _verbose,
                _vlevel);
        _ack_cond_ptr = std::make_unique<ConditionVariable>(is_server,
                _basename + ACK_COND_NAME,
                _namespace,
                _verbose,
                _vlevel);

    }

    bool Consumer::_check_trigger_received() {

        _trigger_counter_clnt.read(_trigger_counter, 0, 0); // reads current value
        // of trigger counter (only written by Producer)

        _trigger_counter_increment = _trigger_counter(0, 0) - _internal_trigger_counter;

        if (_trigger_counter_increment > 1 || 
            _trigger_counter_increment < 0) {
            
            std::string excep = std::string("Found trigger increment < 0 or > 1. Got ") + 
                std::to_string(_trigger_counter_increment);
            _journal.log(__FUNCTION__+_unique_id,
                excep,
                LogType::EXCEP, 
                false); // do throw exception
            return false;

        }

        if (_trigger_counter_increment == 1) {
            
            _internal_trigger_counter = _trigger_counter(0, 0);

            return true;

        } else {
            
            return false;
        }

    }
    
    bool Consumer::_wait(ScopedLock& lock, 
                int ms_timeout) {

        if (ms_timeout > 0) {

            _timeout = !(_trigger_cond_ptr->timedwait(lock, ms_timeout)); // wait with timeout
            
            return !_timeout;

        } else {

            _trigger_cond_ptr->wait(lock); // blocking

            return true;
        }

    }

    std::string Consumer::_getThisName(){

        return THISNAME;
    }

    void Consumer::_check_running(std::string calling_method) {

        if (!_is_running) {

            _journal.log(calling_method+_unique_id,
                "Not running. Did you call the run() method?",
                LogType::EXCEP, 
                true); // throw exception

        }
    }

}