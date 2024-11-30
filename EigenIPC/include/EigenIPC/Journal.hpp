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
#ifndef JOURNAL_HPP
#define JOURNAL_HPP

#include <iostream>
#include <cstdio>
#include <string>
#include <cstring> // for std::strerror()

namespace EigenIPC{

    namespace Colors {
        const std::string kRed = "\033[0;31m";
        const std::string kBoldRed = "\033[1;31m";
        const std::string kGreen = "\033[0;32m";
        const std::string kBoldGreen = "\033[1;32m";
        const std::string kYellow = "\033[0;33m";
        const std::string kBoldYellow = "\033[1;33m";
        const std::string kBlue = "\033[0;34m";
        const std::string kBoldBlue = "\033[1;34m";
        const std::string kMagenta = "\033[0;35m";
        const std::string kBoldMagenta = "\033[1;35m";
        const std::string kCyan = "\033[0;36m";
        const std::string kBoldCyan = "\033[1;36m";
        const std::string kEndl = "\033[0m";
    };

    class Journal {

        public:

            enum class LogType {WARN,
                                EXCEP,
                                INFO,
                                STAT};

            enum class VLevel {V0, // nothing except errors
                    V1, // only warnings
                    V2, // warning + statistics
                    V3}; // warning + statistics + additional info

            Journal(const std::string& classname) :
                _classname(classname) {
            }

            void log(const std::string& methodname,
                     const std::string& message,
                     LogType log_type,
                     bool throw_when_excep = false) {

                // Map LogType to corresponding string
                const char* logTypeStr = logTypeToString(log_type);

                if (log_type == LogType::EXCEP &&
                    !throw_when_excep){

                    std::printf("%s[%s][%s][%s]: %s%s\n",
                            Colors::kBoldRed.c_str(),
                            _classname.c_str(),
                            methodname.c_str(),
                            logTypeStr,
                            message.c_str(),
                            Colors::kEndl.c_str());
                }

                if (log_type == LogType::EXCEP &&
                    throw_when_excep){

                    std::string exception =
                                    Colors::kBoldRed +
                                    std::string("[") +
                                    _classname +
                                    std::string("]") +
                                    std::string("[") +
                                    methodname +
                                    std::string("]") +
                                    std::string("[") +
                                    logTypeStr +
                                    std::string("]: ") +
                                    message +
                                    Colors::kEndl;

                    throw std::runtime_error(exception);

                }

                if (log_type == LogType::WARN){

                    std::printf("%s[%s][%s][%s]: %s%s\n",
                            Colors::kBoldYellow.c_str(),
                            _classname.c_str(),
                            methodname.c_str(),
                            logTypeStr,
                            message.c_str(),
                            Colors::kEndl.c_str());

                }

                if (log_type == LogType::INFO){

                    std::printf("%s[%s][%s][%s]: %s%s\n",
                            Colors::kBoldGreen.c_str(),
                            _classname.c_str(),
                            methodname.c_str(),
                            logTypeStr,
                            message.c_str(),
                            Colors::kEndl.c_str());

                }

                if (log_type == LogType::STAT){

                    std::printf("%s[%s][%s][%s]: %s%s\n",
                            Colors::kBoldBlue.c_str(),
                            _classname.c_str(),
                            methodname.c_str(),
                            logTypeStr,
                            message.c_str(),
                            Colors::kEndl.c_str());

                }


            }

            static void log(const std::string& classname,
                            const std::string& methodname,
                            const std::string& message,
                            LogType log_type,
                            bool throw_when_excep = false) {

                        // Map LogType to corresponding string
                        const char* logTypeStr = logTypeToString(log_type);

                        // You can simplify your code by using a switch or if-else chain
                        // without repeating the same printf structure
                        std::string color;
                        switch(log_type) {
                            case LogType::EXCEP: color = Colors::kBoldRed; break;
                            case LogType::WARN: color = Colors::kBoldYellow; break;
                            case LogType::INFO: color = Colors::kBoldGreen; break;
                            case LogType::STAT: color = Colors::kBoldBlue; break;
                            default: color = ""; break;
                        }

                        if (log_type == LogType::EXCEP && throw_when_excep) {
                            std::string exception =
                                            color +
                                            std::string("[") +
                                            classname +
                                            std::string("]") +
                                            std::string("[") +
                                            methodname +
                                            std::string("]") +
                                            std::string("[") +
                                            logTypeStr +
                                            std::string("]: ") +
                                            message +
                                            Colors::kEndl;

                            throw std::runtime_error(exception);
                        } else {
                            std::printf("%s[%s][%s][%s]: %s%s\n",
                                        color.c_str(),
                                        classname.c_str(),
                                        methodname.c_str(),
                                        logTypeStr,
                                        message.c_str(),
                                        Colors::kEndl.c_str());
                        }
                    }

            static const char* logTypeToString(LogType log_type) {

                switch (log_type) {
                    case LogType::WARN:
                        return "WARN";
                    case LogType::EXCEP:
                        return "EXCEP";
                    case LogType::INFO:
                        return "INFO";
                    case LogType::STAT:
                        return "STAT";
                    default:
                        return "~";
                }

            }

        private:

            std::string _classname;

    };

}
#endif // JOURNAL_HPP
