#pragma once

#include "pyself.h"
#include "leadselfenergy.h"

template <class LeadSelfBase = LeadSelfEnergy> class PyLeadSelf : public PySelf<LeadSelfBase> {
  public:
    /* Inherit the constructors */
    using PySelf<LeadSelfBase>::PySelf;
  protected:
    /* Trampoline (need one for each virtual function) */
    void update() override {
        PYBIND11_OVERLOAD(
            void, /* Return type */
            LeadSelfBase,      /* Parent class */
            update,      /* Name of function in C++ (must match Python name) */
                    /* Argument(s) */
        );
    }
};
