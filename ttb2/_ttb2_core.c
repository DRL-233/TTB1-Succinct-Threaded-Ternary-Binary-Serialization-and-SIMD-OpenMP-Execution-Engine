#define _POSIX_C_SOURCE 200809L
#include <Python.h>
#include "ttb2.h"

static PyObject* py_compile_ttb2(PyObject *self, PyObject *args) {
    Py_buffer view;
    PyObject *val_list = Py_None;
    long long skip_sentinel = 999999999LL;
    if (!PyArg_ParseTuple(args, "y*|OL", &view, &val_list, &skip_sentinel)) return NULL;

    TTB2Handle *h = NULL;
    if (val_list != Py_None && PyList_Check(val_list)) {
        Py_ssize_t n = PyList_Size(val_list);
        int64_t *ext = (int64_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int64_t));
        for (Py_ssize_t i = 0; i < n; i++) {
            PyObject *item = PyList_GetItem(val_list, i);
            ext[i] = item ? PyLong_AsLongLong(item) : 0;
        }
        h = ttb2_compile_with_values((const uint8_t *)view.buf, (size_t)view.len, ext, (int64_t)skip_sentinel);
        free(ext);
    } else {
        h = ttb2_compile((const uint8_t *)view.buf, (size_t)view.len, (int64_t)skip_sentinel);
    }
    PyBuffer_Release(&view);

    if (!h) {
        PyErr_SetString(PyExc_ValueError, "Invalid or truncated TTB2/TTB3 wire packet");
        return NULL;
    }
    return PyLong_FromVoidPtr((void *)h);
}

static PyObject* py_sum_leaves(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    if (!PyArg_ParseTuple(args, "K", &ptr_val)) return NULL;
    return PyLong_FromLongLong(ttb2_sum_leaves((const TTB2Handle *)(uintptr_t)ptr_val));
}

static PyObject* py_query_skip(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    long long skip_sentinel = 999999999LL;
    if (!PyArg_ParseTuple(args, "K|L", &ptr_val, &skip_sentinel)) return NULL;
    return PyLong_FromLongLong(ttb2_query_skip((const TTB2Handle *)(uintptr_t)ptr_val));
}

static PyObject* py_broadcast_center(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    if (!PyArg_ParseTuple(args, "K", &ptr_val)) return NULL;
    return PyLong_FromLongLong(ttb2_broadcast_center((const TTB2Handle *)(uintptr_t)ptr_val));
}

static PyObject* py_jump_sibling(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    uint32_t src_spoke, dst_spoke;
    if (!PyArg_ParseTuple(args, "KII", &ptr_val, &src_spoke, &dst_spoke)) return NULL;
    TTB2JumpResult r = ttb2_jump_sibling((const TTB2Handle *)(uintptr_t)ptr_val, src_spoke, dst_spoke);
    if (r.hops == 0) {
        PyErr_SetString(PyExc_IndexError, "Destination spoke out of range");
        return NULL;
    }
    return Py_BuildValue("(LI)", (long long)r.value, r.hops);
}

static PyObject* py_ascent_to_center(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    uint32_t node_idx;
    if (!PyArg_ParseTuple(args, "KI", &ptr_val, &node_idx)) return NULL;
    TTB2AscentResult r = ttb2_ascent_to_center((const TTB2Handle *)(uintptr_t)ptr_val, node_idx);
    if (r.spoke_id < 0 && r.hub_offset == 0 && node_idx > 0) {
        PyErr_SetString(PyExc_IndexError, "Node index out of range");
        return NULL;
    }
    return Py_BuildValue("(iI)", r.spoke_id, r.hub_offset);
}

static PyObject* py_update_leaf(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    uint32_t node_idx;
    long long new_val, skip_sentinel = 999999999LL;
    if (!PyArg_ParseTuple(args, "KIL|L", &ptr_val, &node_idx, &new_val, &skip_sentinel)) return NULL;
    int ok = ttb2_update_leaf((TTB2Handle *)(uintptr_t)ptr_val, node_idx, (int64_t)new_val, (int64_t)skip_sentinel);
    return PyBool_FromLong(ok);
}

static PyObject* py_set_spoke_status(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    uint32_t spoke_idx;
    int active;
    if (!PyArg_ParseTuple(args, "KIi", &ptr_val, &spoke_idx, &active)) return NULL;
    int ok = ttb2_set_spoke_status((TTB2Handle *)(uintptr_t)ptr_val, spoke_idx, active);
    return PyBool_FromLong(ok);
}

static PyObject* py_ozip_psi1(PyObject *self, PyObject *args) {
    uint32_t s, g, i;
    if (!PyArg_ParseTuple(args, "III", &s, &g, &i)) return NULL;
    return PyLong_FromUnsignedLong(ttb2_ozip_psi1(s, g, i));
}

static PyObject* py_ozip_contract_3x3x2(PyObject *self, PyObject *args) {
    Py_buffer a_view, b_view;
    if (!PyArg_ParseTuple(args, "y*y*", &a_view, &b_view)) return NULL;
    if (a_view.len < 9 * 8 || b_view.len < 6 * 8) {
        PyBuffer_Release(&a_view);
        PyBuffer_Release(&b_view);
        PyErr_SetString(PyExc_ValueError, "Expected 9 int64s for A and 6 int64s for B");
        return NULL;
    }
    int64_t c_out[6];
    ttb2_ozip_contract_3x3x2((const int64_t *)a_view.buf, (const int64_t *)b_view.buf, c_out);
    PyBuffer_Release(&a_view);
    PyBuffer_Release(&b_view);
    return PyBytes_FromStringAndSize((const char *)c_out, sizeof(c_out));
}

static PyObject* py_free_ttb2(PyObject *self, PyObject *args) {
    unsigned long long ptr_val;
    if (!PyArg_ParseTuple(args, "K", &ptr_val)) return NULL;
    ttb2_free((TTB2Handle *)(uintptr_t)ptr_val);
    Py_RETURN_NONE;
}

static PyMethodDef Methods[] = {
    {"compile_ttb2",            py_compile_ttb2,            METH_VARARGS, "Compile TTB2/TTB3 cycloidal hub"},
    {"sum_leaves",              py_sum_leaves,              METH_VARARGS, "SIMD/OpenMP full leaf sum"},
    {"query_skip",              py_query_skip,              METH_VARARGS, "Branchless AVX2 active query"},
    {"broadcast_center",        py_broadcast_center,        METH_VARARGS, "One-stroke AVX2 center broadcast"},
    {"jump_sibling",            py_jump_sibling,            METH_VARARGS, "O(1) dynamic hub sibling jump"},
    {"ascent_to_center",        py_ascent_to_center,        METH_VARARGS, "O(1) periphery-to-center ascent"},
    {"update_leaf",             py_update_leaf,             METH_VARARGS, "O(1) live leaf update"},
    {"set_spoke_status",        py_set_spoke_status,        METH_VARARGS, "O(1) live spoke active mask toggle"},
    {"ozip_psi1",               py_ozip_psi1,               METH_VARARGS, "OZIP mixed-radix bijection 9s+3g+i"},
    {"ozip_contract_3x3x2",     py_ozip_contract_3x3x2,     METH_VARARGS, "OZIP Z3-triality <3,3,2> contraction"},
    {"free_ttb2",               py_free_ttb2,               METH_VARARGS, "Free engine handle"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_ttb2_core", NULL, -1, Methods
};

PyMODINIT_FUNC PyInit__ttb2_core(void) {
    return PyModule_Create(&moduledef);
}
