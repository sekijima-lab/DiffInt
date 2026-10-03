/* Scalar libm operations used by the validated PyTorch 2.0.1 macOS CPU build. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <string.h>

static PyObject *apply_math(PyObject *args, int use_tanh) {
    PyObject *input, *output;
    Py_buffer source = {0}, destination = {0};
    if (!PyArg_ParseTuple(args, "OO", &input, &output)) return NULL;
    if (PyObject_GetBuffer(input, &source, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT) < 0) return NULL;
    if (PyObject_GetBuffer(output, &destination, PyBUF_C_CONTIGUOUS | PyBUF_FORMAT | PyBUF_WRITABLE) < 0) {
        PyBuffer_Release(&source); return NULL;
    }
    if (source.itemsize != sizeof(float) || destination.itemsize != sizeof(float) ||
        source.len != destination.len || source.len % sizeof(float) != 0 ||
        !source.format || !destination.format || strcmp(source.format, "f") || strcmp(destination.format, "f")) {
        PyErr_SetString(PyExc_ValueError, "Expected equally sized contiguous float32 buffers");
        PyBuffer_Release(&source); PyBuffer_Release(&destination); return NULL;
    }
    const float *x = source.buf;
    float *y = destination.buf;
    const Py_ssize_t n = source.len / sizeof(float);
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t i = 0; i < n; ++i) y[i] = use_tanh ? tanhf(x[i]) : expf(x[i]);
    Py_END_ALLOW_THREADS
    PyBuffer_Release(&source); PyBuffer_Release(&destination);
    Py_RETURN_NONE;
}
static PyObject *legacy_exp(PyObject *self, PyObject *args) { return apply_math(args, 0); }
static PyObject *legacy_tanh(PyObject *self, PyObject *args) { return apply_math(args, 1); }
static PyMethodDef methods[] = {
    {"exp", legacy_exp, METH_VARARGS, "Apply scalar expf to float32 buffers."},
    {"tanh", legacy_tanh, METH_VARARGS, "Apply scalar tanhf to float32 buffers."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_legacy_math", NULL, -1, methods};
PyMODINIT_FUNC PyInit__legacy_math(void) { return PyModule_Create(&module); }
