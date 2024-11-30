#include <PyEigenIPC/PyProducer.hpp>

void PyEigenIPC::PyProducer::bind_Producer(pybind11::module& m) {

        pybind11::class_<EigenIPC::Producer> cls(m, "Producer");
        
        cls.def(pybind11::init<std::string, std::string, bool, EigenIPC::Journal::VLevel, bool>(),
             pybind11::arg("basename"),
             pybind11::arg("namespace"),
             pybind11::arg("verbose"), 
             pybind11::arg("vlevel") = VLevel::V0,
             pybind11::arg("force_reconnection") = false);

        cls.def("run", &EigenIPC::Producer::run);
        cls.def("close", &EigenIPC::Producer::close);
        cls.def("trigger", &EigenIPC::Producer::trigger);
        cls.def("wait_ack_from", [](EigenIPC::Producer& self, 
                        int n_consumers,
                        int ms_timeout) {
            
            return self.wait_ack_from(n_consumers, ms_timeout);

        }, pybind11::arg("n_consumers"), pybind11::arg("ms_timeout") = -1);

}