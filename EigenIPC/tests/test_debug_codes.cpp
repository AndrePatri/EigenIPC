#include <EigenIPC/ReturnCodes.hpp>
#include <EigenIPC/Journal.hpp>

using namespace EigenIPC;

using LogType = Journal::LogType;

int main(int argc, char** argv) {

    ReturnCode prova = ReturnCode::NONE;

    prova = prova + ReturnCode::MEMCREATFAIL;

    prova = prova + ReturnCode::MEMSETFAIL;

    prova = prova + ReturnCode::MEMOPEN;

    std::string descr = getDescriptions(prova);

    Journal journal = Journal("Aaaa");

    journal.log(__FUNCTION__,
                    descr,
                    LogType::WARN);


}
