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
#ifndef MEMDEFS_H
#define MEMDEFS_H

namespace EigenIPC{

    class MemDef {

        public:

            static std::string sharedTensorNRowsName() {

                return std::string("sharedTensorNRows");

            }

            static std::string sharedTensorNColsName() {

                return std::string("sharedTensorNCols");

            }

            static std::string sharedTensorDTypeName() {

                return std::string("sharedTensorDType");

            }

            static std::string ClientsCountName() {

                return std::string("clientsCount");

            }

            static std::string isSrvrRunningName() {

                return std::string("isSrvrRunning");


            }

            static std::string memLayoutName() {

                return std::string("memLayoutType");


            }

            static std::string SrvrSemName() {

                return std::string("srvrSem");

            }

            static std::string DataSemName() {

                return std::string("dataSem");

            }

            static std::string CondVarName() {

                return std::string("CondVar");

            }
            
            static std::string CondVarMutexName() {

                return std::string("CondVarMutexName");

            }

        };

}

#endif // MEMDEFS_H
