namespace PyEigenIPC {

    inline ClientWrapper::ClientWrapper(pybind11::object* Obj)
        : wrapped(std::unique_ptr<pybind11::object>(Obj)) {

    }

    template<typename Func>
    auto ClientWrapper::execute(Func&& f) -> decltype(f(std::declval<pybind11::object&>())) {
        return f(*wrapped);
    }

    inline ServerWrapper::ServerWrapper(pybind11::object* Obj)
        : wrapped(std::unique_ptr<pybind11::object>(Obj)) {

    }

    template<typename Func>
    auto ServerWrapper::execute(Func&& f) -> decltype(f(std::declval<pybind11::object&>())) {
        return f(*wrapped);
    }

}
