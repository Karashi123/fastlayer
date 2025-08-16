#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double dot_cpp(py::array_t<double> a, py::array_t<double> b) {
    auto bufA = a.request(), bufB = b.request();
    if (bufA.size != bufB.size) throw std::runtime_error("Array size mismatch");
    double* ptrA = static_cast<double*>(bufA.ptr);
    double* ptrB = static_cast<double*>(bufB.ptr);
    double s = 0.0;
    for (ssize_t i = 0; i < bufA.size; i++) {
        s += ptrA[i] * ptrB[i];
    }
    return s;
}

PYBIND11_MODULE(cpp_hot, m) {
    m.def("dot_cpp", &dot_cpp, "Dot product implemented in C++ with pybind11");
}
