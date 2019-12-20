#include "wrappers.h"

PYBIND11_MODULE(_cppmodule, m) {
    wrap_greenfunction(m);
    wrap_selfenergy(m);
    wrap_leadselfenergy(m);
}
