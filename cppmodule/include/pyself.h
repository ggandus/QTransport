#pragma once

#include "pygreen.h"
#include "selfenergy.h"

template <class SelfBase = SelfEnergy> class PySelf : public PyGreen<SelfBase> {
  public:
    /* Inherit the constructors */
    using PyGreen<SelfBase>::PyGreen;
  protected:
    /* Trampoline (need one for each virtual function) */
    const matrix& retarded(real energy_) {
        PYBIND11_OVERLOAD(
            void, /* Return type */
            SelfBase,      /* Parent class */
            retarded,      /* Name of function in C++ (must match Python name) */
            energy_        /* Argument(s) */
        );
    }
};
