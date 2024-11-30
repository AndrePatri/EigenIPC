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

#include <EigenIPC/CondVar.hpp>

namespace EigenIPC {

    ConditionVariable::ConditionVariable(
                   bool is_server,
                   std::string basename,
                   std::string name_space,
                   bool verbose,
                   VLevel vlevel,
                   bool force_reconnection)
        : _mem_config(basename, name_space),
        _basename(basename), _namespace(name_space),
        _is_server(is_server),
        _verbose(verbose),
        _vlevel(vlevel),
        _journal(Journal(_getThisName())),
        _force_reconnection(force_reconnection)
    {

        if (_is_server) {

            if (_force_reconnection) {

                if (_verbose &&
                    _vlevel > VLevel::V0) {
                    std::string warn = _basename + std::string("-") + _namespace + std::string(". About to preemptively ") + 
                        std::string(" delete mutex and cond var, since force_reconnection was set to true!");
                    _journal.log(__FUNCTION__,
                        warn,
                        LogType::WARN);
                }

                if (!_cleanup_mem()) {
                    if (_verbose) {
                        std::string warn = _basename + std::string("-") + _namespace + std::string(". Could not ") + 
                            std::string(" delete mutex and cond var!");
                        _journal.log(__FUNCTION__,
                            warn,
                            LogType::WARN), 
                            true;
                    }
                }
            }

            _cond_ptr = std::make_unique<NamedCondition>(
                            boost::interprocess::create_only, 
                            _mem_config.mem_path_cond_var.c_str());
                            
            _mutex_ptr = std::make_unique<NamedMutex>(
                                boost::interprocess::create_only,
                                _mem_config.mem_path_cond_var_mutex.c_str());

        } else {

            _cond_ptr = std::make_unique<NamedCondition>(
                        boost::interprocess::open_only, 
                        _mem_config.mem_path_cond_var.c_str());
                        
            _mutex_ptr = std::make_unique<NamedMutex>(
                                boost::interprocess::open_only,
                                _mem_config.mem_path_cond_var_mutex.c_str());
        }

    }

    ConditionVariable::~ConditionVariable(){
        
        close();
    }

    std::string ConditionVariable::cond_var_path() {

        return _mem_config.mem_path_cond_var;
    }

    std::string ConditionVariable::mutex_path() {

        return _mem_config.mem_path_cond_var_mutex;
    }

    ConditionVariable::NamedMutex ConditionVariable::create_named_mutex(std::string at) {
        
        return NamedMutex(boost::interprocess::open_or_create,
                at.c_str());
    }

    ConditionVariable::ScopedLock ConditionVariable::lock(NamedMutex& mutex) {
        
        // acquires mutex
        ScopedLock named_lock(mutex);

        return named_lock;

    }

    ConditionVariable::ScopedLock ConditionVariable::lock() {

        return ScopedLock(*_mutex_ptr);

    }

    void ConditionVariable::unlock(ScopedLock& locked_lock) {
        
        locked_lock.unlock();

    }

    void ConditionVariable::wait(ScopedLock& named_lock) {
        
        // acquire mutex
        _cond_ptr->wait(named_lock);

    }

    void ConditionVariable::wait_for(ScopedLock& named_lock,
                    std::function<bool()> pred) {
        
        // acquire mutex
        _cond_ptr->wait(named_lock, pred);

    }

    bool ConditionVariable::timedwait(ScopedLock& named_lock,
                    unsigned int ms) {
        
        _utc_timeout = boost::posix_time::microsec_clock::universal_time() + 
                boost::posix_time::millisec(ms);

        // acquire mutex
        return _cond_ptr->timed_wait(named_lock, _utc_timeout);

    }

    bool ConditionVariable::timedwait_for(ScopedLock& named_lock,
                    unsigned int ms,
                    std::function<bool()> pred) {
        
        _utc_timeout = boost::posix_time::microsec_clock::universal_time() + 
                boost::posix_time::millisec(ms);

        // acquire mutex
        return _cond_ptr->timed_wait(named_lock, _utc_timeout, pred);

    }

    void ConditionVariable::notify_one() {

        _cond_ptr->notify_one();

    }

    void ConditionVariable::notify_all() {

        _cond_ptr->notify_all();
    }

    void ConditionVariable::close() {

        if (_is_server && !_closed) {

            if (!_cleanup_mem()) { // clean mutex and condition var

                if (_verbose) {
                    std::string exception = _basename + std::string("-") + _namespace + std::string(". Could not ") + 
                        std::string(" delete mutex and cond var!");
                    _journal.log(__FUNCTION__,
                        exception,
                        LogType::EXCEP), 
                        true;
                }
            } 
        }

        _closed = true;
    
    }

    std::string ConditionVariable::_getThisName(){

        return THISNAME;
    }

    bool ConditionVariable::_cleanup_mem(){

        if (_verbose &&
                _vlevel > VLevel::V1) {

            std::string info = std::string("Cleaning up named mutex at ") + 
            std::string(_mem_config.mem_path_cond_var_mutex) + 
            std::string(" and condition variable at ") + 
            std::string(_mem_config.mem_path_cond_var);

            _journal.log(__FUNCTION__,
                info,
                LogType::STAT);

        }

        // Destroy named mutex and named condition variable
        bool named_mutex_rm_ok = NamedMutex::remove(_mem_config.mem_path_cond_var_mutex.c_str());
        bool named_cond_rm_ok = NamedCondition::remove(_mem_config.mem_path_cond_var.c_str());

        return named_mutex_rm_ok && named_cond_rm_ok;
    }

}