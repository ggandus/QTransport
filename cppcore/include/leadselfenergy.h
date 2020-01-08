#pragma once
#include "selfenergy.h"

namespace negf {

  class LeadSelfEnergy : public SelfEnergy
  {
  private:
    int m = 0;
    real conv = 1e-8;
    matrix a, b, v_11, v_10, v_01, v_01_dot_b;
    // matrix& a;
    // matrix b, v_11, v_10, v_01, v_01_dot_b;
  protected:
    matrix h_ij, s_ij;
    void update ();
  public:
    LeadSelfEnergy (const matrix&, const matrix&,
                    const matrix&, const matrix&,
                    const matrix&, const matrix&,
                    real eta_ = 1e-4);
    real delta = conv + 1;
  };

}
