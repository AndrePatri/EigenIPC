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
#ifndef RETURNCODES_HPP
#define RETURNCODES_HPP

#include <unordered_map>

namespace EigenIPC {

    enum class ReturnCode: unsigned long long { // 64 bit --> max 64 unique base codes
        NONE = 0, // empty
        RESET = 1ULL << 0, // reset
        MEMOPEN = 1ULL << 1, // opened memory
        MEMOPENFAIL = 1ULL << 2, // failed to open memory
        MEMCREAT = 1ULL << 3, // created memory
        MEMCREATFAIL = 1ULL << 4, // failed to create memory
        MEMSET = 1ULL << 5, // memory set
        MEMSETFAIL = 1ULL << 6, // failed to set memory
        MEMEXISTS = 1ULL << 7, // memory already exists
        MEMCLEAN = 1ULL << 8, // cleaned memory
        MEMFDCLOSED = 1ULL << 9, // closed file descriptor for memory
        MEMUNLINK = 1ULL << 10, // unlinked memory
        MEMMAP = 1ULL << 11, // mapped memory
        MEMMAPFAIL = 1ULL << 12, // failed to map memory
        INDXOUT = 1ULL << 13, // indexes are out of bound for mem
        NOFIT = 1ULL << 14, // tensor does not fit in memory
        SEMACQTIMEOUT = 1ULL << 15, // semaphore acquisition timeout
        SEMACQ = 1ULL << 16, // semaphore acquired
        SEMACQFAIL = 1ULL << 17, // semaphore acquisition failed
        SEMACQTRY = 1ULL << 18, // trying semaphore acquisition
        SEMACQRETRY = 1ULL << 18, // retrying semaphore acquisition
        SEMACQSIGINT = 1ULL << 19, // retrying semaphore acquisition
        SEMOPEN = 1ULL << 20, // semaphore opened
        SEMOPENFAIL = 1ULL << 21, // failed to open semaphore
        SEMREL = 1ULL << 22, // reselased semaphore
        SEMRELFAIL = 1ULL << 23, // failed to release semaphore
        SEMCLOSE = 1ULL << 24, // close semaphore
        SEMUNLINK = 1ULL << 25, // unlinked semaphore
        WRITEFAIL = 1ULL << 26, // failed to write to memory
        READFAIL = 1ULL << 27, // failed to read from memory
        // ... up to 1ULL << 62
        OTHER = 1ULL << 62,
        UNKNOWN = 1ULL << 63,

    };

    inline std::string getDescription(ReturnCode code) {

            static const std::unordered_map<ReturnCode, std::string> CodeMap = {

                {ReturnCode::NONE, "NONE"},
                {ReturnCode::RESET, "RESET"},
                {ReturnCode::MEMOPEN, "MEMOPEN"},
                {ReturnCode::MEMOPENFAIL, "MEMOPENFAIL"},
                {ReturnCode::MEMCREAT, "MEMCREAT"},
                {ReturnCode::MEMCREATFAIL, "MEMCREATFAIL"},
                {ReturnCode::MEMSET, "MEMSET"},
                {ReturnCode::MEMSETFAIL, "MEMSETFAIL"},
                {ReturnCode::MEMEXISTS, "MEMEXISTS"},
                {ReturnCode::MEMCLEAN, "MEMCLEAN"},
                {ReturnCode::MEMFDCLOSED, "MEMFDCLOSED"},
                {ReturnCode::MEMUNLINK, "MEMUNLINK"},
                {ReturnCode::MEMMAP, "MEMMAP"},
                {ReturnCode::MEMMAPFAIL, "MEMMAPFAIL"},
                {ReturnCode::INDXOUT, "INDXOUT"},
                {ReturnCode::NOFIT, "NOFIT"},
                {ReturnCode::SEMACQTIMEOUT, "SEMACQTIMEOUT"},
                {ReturnCode::SEMACQ, "SEMACQ"},
                {ReturnCode::SEMACQFAIL, "SEMACQFAIL"},
                {ReturnCode::SEMACQTRY, "SEMACQTRY"},
                {ReturnCode::SEMACQRETRY, "SEMACQRETRY"},
                {ReturnCode::SEMOPEN, "SEMOPEN"},
                {ReturnCode::SEMOPENFAIL, "SEMOPENFAIL"},
                {ReturnCode::SEMREL, "SEMREL"},
                {ReturnCode::SEMRELFAIL, "SEMRELFAIL"},
                {ReturnCode::SEMCLOSE, "SEMCLOSE"},
                {ReturnCode::SEMUNLINK, "SEMUNLINK"},
                // ... other codes
                {ReturnCode::OTHER, "OTHER"},
                {ReturnCode::UNKNOWN, "UNKNOWN"},


            };

            auto it = CodeMap.find(code);

            if (it != CodeMap.end()) {

                return it->second;

            } else {

                return "UNKNOWN";

            }
        }

    // inline to avoid inclusion in multiple translation units
    inline ReturnCode operator+(ReturnCode lhs, ReturnCode rhs) {

        if (rhs == ReturnCode::RESET || lhs == ReturnCode::RESET) {

            return ReturnCode::NONE;

        }

        return static_cast<ReturnCode>(static_cast<unsigned long long>(lhs) |
                                       static_cast<unsigned long long>(rhs));
    }

    inline ReturnCode operator-(ReturnCode lhs, ReturnCode rhs) {
        if (rhs == ReturnCode::RESET || lhs == ReturnCode::RESET) {
            // If either side has RESET, subtracting does nothing
            return lhs;
        }

        // Use bitwise AND NOT operation to remove the specified code
        return static_cast<ReturnCode>(static_cast<unsigned long long>(lhs) & ~static_cast<unsigned long long>(rhs));
    }

    inline bool isin(ReturnCode subcode,
              ReturnCode code) {

        return (static_cast<unsigned long long>(code) &
                static_cast<unsigned long long>(subcode)) ==
                    static_cast<unsigned long long>(subcode);
    }

    inline std::string toString(ReturnCode code) {

        return std::to_string(static_cast<unsigned long long>(code));

    }

    inline std::string getDescriptions(ReturnCode combinedCode) {
        
        // gets all descriptions in combinedCode and puts them in a single string
        // this can be very useful for improving interpretability of debugging codes

        std::string descriptions = "[";

        bool first = true;

        // Iterate through each base code and check if it is present in the combinedCode
        for (unsigned long long i = 0; i <= 62; ++i) {

            ReturnCode baseCode = static_cast<ReturnCode>(1ULL << i);

            if (isin(baseCode, combinedCode)) {
                
                if (!first) {
                    descriptions += ", ";
                }

                descriptions += getDescription(baseCode);

                first = false;
            }
        }

        descriptions += "]";

        return descriptions;
    }

    inline void reset() {

    }

    // Example usage:
    // auto combinedCode = ReturnCode::MEMOPENED + ReturnCode::MEMEXISTS;
    // if (combinedCode == (ReturnCode::MEMOPENED + ReturnCode::MEMEXISTS)) { ... }
    // ReturnCode::MEMOPENED + ReturnCode::MEMOPENED will equal ReturnCode::MEMOPENED
    // code = combinedCode + InfoCodes::RESET; // This will set `code` to InfoCodes::RESET
}
#endif // RETURNCODES_HPP

