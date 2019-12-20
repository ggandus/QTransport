#include "pyleadself.h"

void wrap_leadselfenergy(py::module& m) {
  // bindings to GreenFunction class
  using LSE = LeadSelfEnergy;
  using PyClass = PyLeadSelf<>;
  py::class_<LSE, PyClass>(m, "LeadSelfEnergy")
      .def(py::init<const matrix &, const matrix &,
                    const matrix &, const matrix &,
                    const matrix &, const matrix &,
                    real>())
      .def("retarded", &LSE::retarded)
      .def_readonly("delta", &LSE::delta);
      // .def_readonly("S", &PyClass::S);

}
