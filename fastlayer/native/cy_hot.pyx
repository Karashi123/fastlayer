# native/cy_hot.pyx
# cython: boundscheck=False, wraparound=False, cdivision=True
cpdef double dot_cy(double[:] a, double[:] b):
    cdef Py_ssize_t i, n = a.shape[0]
    cdef double s = 0.0
    for i in range(n):
        s += a[i] * b[i]
    return s
