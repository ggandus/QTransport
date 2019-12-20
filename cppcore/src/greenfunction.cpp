#include "greenfunction.h"

namespace negf{

  GreenFunction::GreenFunction (const matrix& hamilton, const matrix& overlap,
                                real eta_)
    : H(hamilton), S(overlap), eta(eta_)
  {
    auto m = hamilton.rows();
    Ginv.resize(m,m);
  }

  const matrix& GreenFunction::retarded (real energy_)
  {
    if (energy != energy_) {

      energy = energy_;
      update();

    }

    return Ginv;

  }

  void GreenFunction::update ()
  {
      z = energy + eta * 1i;
      Ginv = z * S - H;
  }

}
