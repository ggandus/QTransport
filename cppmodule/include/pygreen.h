#pragma once

#include "wrappers.h"
#include "greenfunction.h"

using namespace negf;

template <class GreenBase = GreenFunction> class PyGreen : public GreenBase {
  public:
    /* Inherit the constructors */
    using GreenBase::GreenBase;
    using GreenBase::H;
    using GreenBase::S;
    using GreenBase::energy;
    using GreenBase::eta;
  protected:
    /* Trampoline (need one for each virtual function) */
    void update() override {
        PYBIND11_OVERLOAD(
            void, /* Return type */
            GreenBase,      /* Parent class */
            update,          /* Name of function in C++ (must match Python name) */
                            /* Argument(s) */
        );
    }
};
