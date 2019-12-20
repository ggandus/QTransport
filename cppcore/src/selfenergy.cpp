#include "selfenergy.h"

namespace negf {

  SelfEnergy::SelfEnergy (const matrix& h_ii, const matrix& s_ii,
                          const matrix& h_im, const matrix& s_im,
                          real eta_)
    : GreenFunction(h_ii, s_ii, eta_), h_im(h_im), s_im(s_im)
  {
    auto m = h_im.cols(), i = h_im.rows();
    sigma_mm.resize(m,m);
    tau_im.resize(i,m);
    tau_mi.resize(m,i);
  }


  const matrix& SelfEnergy::retarded (real energy_)
  {
    if (energy != energy_) {

      energy = energy_;
      this->update();

      tau_im = z * s_im - h_im;
      tau_mi = z * s_im.adjoint() - h_im.adjoint();
      Eigen::PartialPivLU<Eigen::Ref<matrix> > lu_ii(Ginv);
      sigma_mm.noalias() = tau_mi * lu_ii.solve(tau_im);
    }

    return sigma_mm;

  }

  void SelfEnergy::update ()
  {
    GreenFunction::update();
  }

}
