#include "leadselfenergy.h"

namespace negf {

  LeadSelfEnergy::LeadSelfEnergy (const matrix& h_ii, const matrix& s_ii,
                                  const matrix& h_ij, const matrix& s_ij,
                                  const matrix& h_im, const matrix& s_im,
                                  real eta_)
    : SelfEnergy(h_ii, s_ii, h_im, s_im, eta_), h_ij(h_ij), s_ij(s_ij)
  {
    auto m = h_ij.rows();
    a.resize(m,1);
    b.resize(m,1);
    v_11.resize(m,m);
    v_10.resize(m,m);
    v_01.resize(m,m);
    v_01_dot_b.resize(m,m);
  }

  void LeadSelfEnergy::update ()
  {

    // SelfEnergy::update();
    z = energy + eta * 1i;
    Ginv = z * S.adjoint() - H.adjoint();
    v_11 = Ginv;
    v_10 = z * s_ij - h_ij;
    v_01 = z * s_ij.adjoint() - h_ij.adjoint();

    delta = conv + 1;
    while (delta > conv) {

        a.noalias() = v_11.partialPivLu().solve(v_01);
        b.noalias() = v_11.partialPivLu().solve(v_10);
        v_01_dot_b.noalias() = v_01 * b;
        Ginv -= v_01_dot_b;
        v_11 -= v_10 * a +  v_01_dot_b;
        v_01 = - (v_01 * a).eval();
        v_10 = - (v_10 * b).eval();
        delta = v_01.cwiseAbs().maxCoeff();

      }

  }

}
