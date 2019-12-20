#include "pygreen.h"

void wrap_greenfunction(py::module& m) {
  // bindings to GreenFunction class
  using GF = GreenFunction;
  using PyClass = PyGreen<>;
  py::class_<GF, PyClass>(m, "GreenFunction")
      .def(py::init<const matrix &, const matrix &, real>()) //,
                    // "energy_"_a=0., "eta_"_a=1e-4)
      .def("retarded", &GF::retarded)
      .def_readonly("H", &PyClass::H)
      .def_readonly("S", &PyClass::S)
      .def_readonly("energy", &PyClass::energy)
      .def_readonly("eta", &PyClass::eta);

}
