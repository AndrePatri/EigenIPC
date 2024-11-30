#include <PyEigenIPC/PyConsumer.hpp>

void PyEigenIPC::PyConsumer::bind_Consumer(pybind11::module& m) {

        pybind11::class_<EigenIPC::Consumer> cls(m, "Consumer");
        
        cls.def(pybind11::init<std::string, std::string, bool, EigenIPC::Journal::VLevel>(),
             pybind11::arg("basename"),
             pybind11::arg("namespace"),
             pybind11::arg("verbose"), 
             pybind11::arg("vlevel") = VLevel::V0);
        cls.def("run", &EigenIPC::Consumer::run);
        cls.def("close", &EigenIPC::Consumer::close);
        cls.def("wait", [](EigenIPC::Consumer& self, 
                        int ms_timeout) {
            
            return self.wait(ms_timeout);

        }, pybind11::arg("ms_timeout") = -1);
        cls.def("ack", &EigenIPC::Consumer::ack);

}