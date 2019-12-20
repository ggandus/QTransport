#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;
using namespace py::literals;

void wrap_greenfunction(py::module& m);
void wrap_selfenergy(py::module& m);
void wrap_leadselfenergy(py::module& m);
