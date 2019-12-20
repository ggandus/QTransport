#pragma once
#include "globals.h"

namespace negf {

  using namespace types;

  class GreenFunction
  {
  protected:
    matrix Ginv, H, S;
    real energy = -1e4, eta;
    complex z;
    virtual void update () = 0;
  public:
    GreenFunction (const matrix&, const matrix&, real eta_ = 1e-4);
    ~ GreenFunction () { };
    const matrix& retarded (real);
  };

}
