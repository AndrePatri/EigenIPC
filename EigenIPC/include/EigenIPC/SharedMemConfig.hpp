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
#ifndef SHAREDMEMCONFIG_HPP
#define SHAREDMEMCONFIG_HPP

#include <string>
#include <EigenIPC/MemDefs.hpp>

namespace EigenIPC{

    class SharedMemConfig {

    public:

        SharedMemConfig(const std::string& name,
                        const std::string& name_space = "")
            : _name(name),
              _namespace(name_space) {
            
            // original shared mem
            mem_path = "/" + _namespace + _name;

            mem_path_nrows = "/" + _namespace + _name + "_" + MemDef::sharedTensorNRowsName();

            mem_path_ncols = "/" + _namespace + _name + "_" + MemDef::sharedTensorNColsName();

            mem_path_dtype = "/" + _namespace + _name + "_" + MemDef::sharedTensorDTypeName();

            mem_path_clients_counter = "/" + _namespace + _name + "_" + MemDef::ClientsCountName();

            mem_path_isrunning = "/" + _namespace + _name + "_" + MemDef::isSrvrRunningName();

            mem_path_mem_layout = "/" + _namespace + _name + "_" + MemDef::memLayoutName();

            mem_path_server_sem = "/" + _namespace + _name + "_" + MemDef::SrvrSemName();

            mem_path_data_sem = "/" + _namespace + _name + "_" + MemDef::DataSemName();
            
            // conditon variable wrapper

            mem_path_cond_var = _namespace + _name + MemDef::CondVarName();

            mem_path_cond_var_mutex = _namespace + _name + MemDef::CondVarMutexName();

        }

        // shared data path
        std::string mem_path;

        // auxiliary data
        std::string mem_path_nrows;
        std::string mem_path_ncols;
        std::string mem_path_dtype;
        std::string mem_path_clients_counter;
        std::string mem_path_isrunning;
        std::string mem_path_mem_layout;

        // semaphores
        std::string mem_path_server_sem;
        std::string mem_path_data_sem;

        // cond. var. wrappers
        std::string mem_path_cond_var;
        std::string mem_path_cond_var_mutex;

    private:

        std::string _name;
        std::string _namespace;

    };

}
#endif // SHAREDMEMCONFIG_HPP
