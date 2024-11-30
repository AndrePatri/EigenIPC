<h2 align="center" style="text-decoration: none;"> <img src="https://img.shields.io/badge/License-GPLv2-purple.svg" alt="License"> EigenIPC <img src="https://img.shields.io/badge/Docs-WIP-yellow" alt="Docs">

![icon.svg](docs/sphinx/source/_static/icon.svg)

</h2>

<!-- ![GitHub-Mark-Light](docs/icon-light.svg#gh-dark-mode-only)![GitHub-Mark-Dark](docs/icon-dark.svg#gh-light-mode-only) -->
Rt-friendly 2D **shared tensors** built on top of **POSIX IPC** standards and [**Eigen**](https://eigen.tuxfamily.org/index.php?title=Main_Page) library, shipped with Python bindings and [**NumPy**](https://numpy.org/) support.

### 1. Features:
- EigenIPC leverages *POSIX* *shared memory* and *semaphores* primitives in conjunction with Eigen's matrix API to create shared views of tensors over **multiple processes**, which can then be safely accessed and manipulated in a **rt-compatible** way.
- EigenIPC exposes to the user a convenient `Client/Server` API to create, read, write and manage shared tensors from separate processes (on the same machine) with **minimum latency**, and **internally takes care of** avoiding **race conditions** on the data. Note that while only one instance of a server is allowed, an arbitrary number of clients can be created. To make also non-atomic operations on the shared data safe, the EigenIPC also exposes methods for manually handling data acquisition and release.
- EigenIPC is templatized so as to support the creation of shared tensors with
  - different datatypes (`bool`, `int`, `float` and `double`).
  - `ColMajor` (column-major) and `RowMajor` (row-major) layouts.
- Additionally, a `StringTensor` wrapper object designed for sharing arrays of UTF8 encoded-strings is also provided.
- Producer/Consumer wrappers built on top of [boost::interprocess](https://www.boost.org/doc/libs/1_46_0/doc/html/interprocess/synchronization_mechanisms.html)'s named condition variables and mutex + EigenIPC's client/server for system-wide single producer - multiple consumers triggering

The library is also **fully binded** in Python, codename `PyEigenIPC`, and allows for seamless integration with the popular NumPy library.

### 2. Documentation: 

For more details on what EigenIPC offers, usage examples, performance benchmarks and so on and so forth, please have a look at the [documentation](https://andrepatri.github.io/EigenIPC/v0.1.0/index.html) (WIP).

### 3. Continous integration status: 

| *main* | *devel* |
|----------|----------|
| <img src="https://github.com/AndrePatri/EigenIPC/actions/workflows/focal_CI_build_main.yml/badge.svg" alt="Focal CI">  | <img src="https://github.com/AndrePatri/EigenIPC/actions/workflows/focal_CI_build_devel.yml/badge.svg" alt="CI Focal">  | 
| <img src="https://github.com/AndrePatri/EigenIPC/actions/workflows/jammy_CI_build_main.yml/badge.svg" alt="CI Jammy">  | <img src="https://github.com/AndrePatri/EigenIPC/actions/workflows/jammy_CI_build_devel.yml/badge.svg" alt="CI Jammy">  |


### 4. Install from source: 

Just clone this repo, build and install the library with CMake, ensuring that all dependencies are correctly installed in you system/environment. 

In case you need the python interface, turn on the cmake flag `WITH_PYTHON`, which is off by default. Additionally, you can compile and run the tests by turning on the `WITH_TESTS` flags. 

The tests include some consistency checks to ensure the library works properly and some performance benchmarks, for both the Cpp and Python interface and for all the supported dtypes and layouts. 

### 4. Install from Anaconda: 

The full library (including PyEigenIPC) is also deployed on Anaconda at [sharsor_ipcpp](https://anaconda.org/AndrePatri/sharsor_ipcpp/files), with Python support from versions 3.7 up to 3.11. 
Please note that these Anaconda versions are periodically updated starting from the *main* branch, so cutting-edge features might not be available there.

### 5. Using EigenIPC from another package 

Importing and linking against the library is super easy: have a look at an example CMakeLists.txt [here](docs/sphinx/source/_static/CMakeLists_example.txt).

### 6. External dependencies: 
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) - *required*: a C++ template library for linear algebra. On Linux, install it with ```sudo apt-get install libeigen3-dev```. Tensors on EigenIPC are exposed, at the Cpp level, as either Eigen matrices or Eigen Maps of the underlying memory.
- [boost::interprocess](https://www.boost.org/doc/libs/1_46_0/doc/html/interprocess/synchronization_mechanisms.html) - *required*: used by the Producer/Consumer classes.
- [GoogleTest](https://github.com/google/googletest) - *optional*: a C++ testing framework. On Linux, install it with ```sudo apt-get install libgtest-dev```.
<!-- - **Real-time library** (rt) - *required*: ```sudo apt-get install librt-dev```
- **pthread** - *required*: the POSIX Threads library. On Linux, install it with ```sudo apt-get install libpthread-stubs0-dev``` -->

To compile the bindings you'll need: 
- [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) - *required*
- [pybind11](https://github.com/pybind/pybind11) - *required*. 

<!-- Run-time dependencies for the bindings:
- **linux-vdso**
- **librt**
- **libstdc++**
- **libgcc**
- **libc**
- **libpthread**
- **libm** -->

### 7. Additional notes
If employed properly, the C++ version of the library can be employed in a rt-safe way:
- Dynamic allocations are reduced to the bare minimum.
- Run-time semaphore acquisitions (used by `write` and `read`) are designed to be non-blocking and rt-safe. It is then user's responsibility to handle, if necessary, possible write/read failures due to semaphore acquisition.
- Calls to `run()/attach()` and `stop()` are not guaranteed to be rt-friendly. For rt applications, these calls should only be done during initialization/closing steps or, at run-time, sporadically.
- As of now, the logging utility `Journal` is not guaranteed to be rt-friendly. It is very useful for debugging purposes but, if working with rt-code, it is strongly recommended to set the verbosity level to `VLevel::V0` (which prints only exceptions) or to disable logging altogether with `verbose = false`.

### 8. Roadmap:
- [ ] write some documentation!!
