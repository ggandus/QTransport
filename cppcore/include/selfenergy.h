#pragma once
#include "greenfunction.h"

namespace negf {

  class SelfEnergy : public GreenFunction
  {
  private:
    matrix tau_im, tau_mi;
  protected:
    matrix sigma_mm, h_im, s_im;
    // using GreenFunction::update;
    void update ();
  public:
    SelfEnergy (const matrix&, const matrix&,
                const matrix&, const matrix&,
                real eta_ = 1e-4);
    const matrix& retarded (real);
  };

}
