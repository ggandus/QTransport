#pragma once
#include <Eigen/Dense>

namespace negf {

  namespace types {

    typedef float real;
    typedef std::complex<real> complex;
    typedef Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic> matrix;

  }

}
