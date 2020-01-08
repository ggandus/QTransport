#include "leadselfenergy.h"
#include <iostream>
namespace negf {

  LeadSelfEnergy::LeadSelfEnergy (const matrix& h_ii, const matrix& s_ii,
                                  const matrix& h_ij, const matrix& s_ij,
                                  const matrix& h_im, const matrix& s_im,
                                  real eta_)
    : SelfEnergy(h_ii, s_ii, h_im, s_im, eta_), h_ij(h_ij), s_ij(s_ij)
  {
    m = h_ij.rows();
    a.resize(m,m);
    b.resize(m,m);
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
    int iteration_number = 0;

    // both v_01 and a have the weird ass type:
    // const Eigen::RotationBase<OtherDerived, -1>
    // v_01 = 100x100 matrix, complex64
    // a.shape = 100,1
    // Eigen::PartialPivLU<matrix> lu_ii(m,m);
    // Eigen::HouseholderQR<Eigen::Ref<matrix>> qr_ii(m,m);
    // Eigen::HouseholderQR<Eigen::Ref<matrix>> lu_ii;

    while ((delta > conv) && (iteration_number < 13)) {
        iteration_number++;
        std::cout << iteration_number << "/* iteration */" << '\n';
        Eigen::HouseholderQR<Eigen::Ref<matrix>> lu_ii(v_11);
        lu_ii.solve(v_10);
        lu_ii.solve(v_01);
        // qr_ii.compute(v_11);
        // qr_ii.solve(v_01);
        // qr_ii.solve(v_01);
        // a.noalias() = qr_ii.solve(v_01); //slow
        // b.noalias() = qr_ii.solve(v_10); //slow
        // v_01_dot_b.noalias() = v_01 * b;
        // Ginv -= v_01_dot_b;
        // v_11 -= v_10 * a +  v_01_dot_b;
        // v_01 = - (v_01 * a).eval(); // slow
        // v_10 = - (v_10 * b).eval(); // slow
        // delta = v_01.cwiseAbs().maxCoeff();

      }

  }

}
