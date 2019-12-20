#include "pyself.h"

void wrap_selfenergy(py::module& m) {
  // bindings to GreenFunction class
  using SE = SelfEnergy;
  using PyClass = PySelf<>;
  py::class_<SE, PyClass>(m, "SelfEnergy")
      .def(py::init<const matrix &, const matrix &,
                    const matrix &, const matrix &,
                    real>())
      .def("retarded", &SE::retarded)
      .def_readonly("H", &PyClass::H)
      .def_readonly("S", &PyClass::S);

}
