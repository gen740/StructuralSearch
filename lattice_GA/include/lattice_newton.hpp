#pragma once
#include <simd/simd.h>

#include <algorithm>
#include <autodiff/forward/dual.hpp>
#include <iostream>
#include <numbers>
#include <random>
#include <vector>

#include "generator.hpp"

namespace Lattice {

using simd::double3;

using autodiff::at;
using autodiff::derivative;
using autodiff::dual;
using autodiff::wrt;

struct Atom {
  std::array<dual, 3> x;
};

template <int N = 2>
class LatticeN {
  std::array<Atom, N> atoms_;

  // dual r1 = 1;
  // dual r2 = 1;
  // dual phi2 = std::numbers::pi / 3;
  // dual r3 = 1.63;
  // dual phi3 = 0;
  // dual th3 = 0;

  std::array<dual, 3> a;
  std::array<dual, 3> b;
  std::array<dual, 3> c;

 public:
  LatticeN(int seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (auto& atom : atoms_) {
      for (int i = 0; i < 3; ++i) {
        atom.x[i] = dual(dis(gen));
      }
    }
    for (int i = 0; i < 3; ++i) {
      a[i] = dual(dis(gen));
      b[i] = dual(dis(gen));
      c[i] = dual(dis(gen));
    }
  }

  std::array<std::array<double, 3>, 3> get_unit_vector() const {
    return {{{static_cast<double>(a[0]), static_cast<double>(a[1]),
              static_cast<double>(a[2])},
             {static_cast<double>(b[0]), static_cast<double>(b[1]),
              static_cast<double>(b[2])},
             {static_cast<double>(c[0]), static_cast<double>(c[1]),
              static_cast<double>(c[2])}}};
  }

  std::array<std::array<double, 3>, 2> atom_positions() const {
    auto unit_vector = this->get_unit_vector();

    std::array<std::array<double, 3>, 2> result;
    result[0][0] =
        static_cast<double>(this->atoms_[0].x[0]) * unit_vector[0][0] +
        static_cast<double>(this->atoms_[0].x[1]) * unit_vector[1][0] +
        static_cast<double>(this->atoms_[0].x[2]) * unit_vector[2][0];
    result[0][1] =
        static_cast<double>(this->atoms_[0].x[0]) * unit_vector[0][1] +
        static_cast<double>(this->atoms_[0].x[1]) * unit_vector[1][1] +
        static_cast<double>(this->atoms_[0].x[2]) * unit_vector[2][1];
    result[0][2] =
        static_cast<double>(this->atoms_[0].x[0]) * unit_vector[0][2] +
        static_cast<double>(this->atoms_[0].x[1]) * unit_vector[1][2] +
        static_cast<double>(this->atoms_[0].x[2]) * unit_vector[2][2];
    result[1][0] =
        static_cast<double>(this->atoms_[1].x[0]) * unit_vector[0][0] +
        static_cast<double>(this->atoms_[1].x[1]) * unit_vector[1][0] +
        static_cast<double>(this->atoms_[1].x[2]) * unit_vector[2][0];
    result[1][1] =
        static_cast<double>(this->atoms_[1].x[0]) * unit_vector[0][1] +
        static_cast<double>(this->atoms_[1].x[1]) * unit_vector[1][1] +
        static_cast<double>(this->atoms_[1].x[2]) * unit_vector[2][1];
    result[1][2] =
        static_cast<double>(this->atoms_[1].x[0]) * unit_vector[0][2] +
        static_cast<double>(this->atoms_[1].x[1]) * unit_vector[1][2] +
        static_cast<double>(this->atoms_[1].x[2]) * unit_vector[2][2];
    return result;
  }

  static dual calcualte_energy(dual a1, dual a2, dual a3,  //
                               dual b1, dual b2, dual b3,  //
                               dual c1, dual c2, dual c3,

                               dual x11, dual x12, dual x13,  //
                               dual x21, dual x22, dual x23) {
    int SHIFT = 2;
    std::array<std::array<dual, 3>, 2> x = {{
        {x11 * a1 + x12 * b1 + x13 * c1, x11 * a2 + x12 * b2 + x13 * c2,
         x11 * a3 + x12 * b3 + x13 * c3},
        {x21 * a1 + x22 * b1 + x23 * c1, x21 * a2 + x22 * b2 + x23 * c2,
         x21 * a3 + x22 * b3 + x23 * c3},
    }};

    dual energy = 0;
    for (int i = -SHIFT; i <= SHIFT; i++) {
      for (int j = -SHIFT; j <= SHIFT; j++) {
        for (int k = -SHIFT; k <= SHIFT; k++) {
          for (int l = 0; l < N; l++) {
            for (int m = 0; m < N; m++) {
              if (l == m && i == 0 && j == 0 && k == 0) {
                continue;
              }
              std::array<dual, 3> r = {0, 0, 0};
              r[0] += x[l][0] - x[m][0] + a1 * i + b1 * j + c1 * k;
              r[1] += x[l][1] - x[m][1] + a2 * i + b2 * j + c2 * k;
              r[2] += x[l][2] - x[m][2] + a3 * i + b3 * j + c3 * k;
              dual dist2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
              energy += 1 / pow(dist2, 6) - 1 / pow(dist2, 3);
            }
          }
        }
      }
    }
    return energy;
  }

  void update() {
    double step = 6e-4;
    auto clamp_value = 6e-3;

    auto energy = calcualte_energy(
        a[0], a[1], a[2],                                                  //
        b[0], b[1], b[2],                                                  //
        c[0], c[1], c[2], atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],  //
        atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]);

    auto da0 = derivative(
        calcualte_energy, wrt(a[0]),
        at(a[0], a[1], a[2],                                                  //
           b[0], b[1], b[2],                                                  //
           c[0], c[1], c[2], atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],  //
           atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto da1 = derivative(
        calcualte_energy, wrt(a[1]),
        at(a[0], a[1], a[2],                                                  //
           b[0], b[1], b[2],                                                  //
           c[0], c[1], c[2], atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],  //
           atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto da2 = derivative(
        calcualte_energy, wrt(a[2]),
        at(a[0], a[1], a[2],                                                  //
           b[0], b[1], b[2],                                                  //
           c[0], c[1], c[2], atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],  //
           atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto db0 = derivative(
        calcualte_energy, wrt(b[0]),
        at(a[0], a[1], a[2],                                                  //
           b[0], b[1], b[2],                                                  //
           c[0], c[1], c[2], atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],  //
           atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto db1 = derivative(
        calcualte_energy, wrt(b[1]),
        at(a[0], a[1], a[2],                                                  //
           b[0], b[1], b[2],                                                  //
           c[0], c[1], c[2], atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],  //
           atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto db2 = derivative(
        calcualte_energy, wrt(b[2]),
        at(a[0], a[1], a[2],                                                  //
           b[0], b[1], b[2],                                                  //
           c[0], c[1], c[2], atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],  //
           atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto dc0 = derivative(
        calcualte_energy, wrt(c[0]),
        at(a[0], a[1], a[2],                                                  //
           b[0], b[1], b[2],                                                  //
           c[0], c[1], c[2], atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],  //
           atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto dc1 = derivative(
        calcualte_energy, wrt(c[1]),
        at(a[0], a[1], a[2],                                                  //
           b[0], b[1], b[2],                                                  //
           c[0], c[1], c[2], atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],  //
           atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto dc2 = derivative(
        calcualte_energy, wrt(c[2]),
        at(a[0], a[1], a[2],                                                  //
           b[0], b[1], b[2],                                                  //
           c[0], c[1], c[2], atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],  //
           atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto dx00 = derivative(
        calcualte_energy, wrt(atoms_[0].x[0]),
        at(a[0], a[1], a[2],                                                  //
           b[0], b[1], b[2],                                                  //
           c[0], c[1], c[2], atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],  //
           atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto dx01 = derivative(
        calcualte_energy, wrt(atoms_[0].x[1]),
        at(a[0], a[1], a[2],                                                  //
           b[0], b[1], b[2],                                                  //
           c[0], c[1], c[2], atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],  //
           atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto dx02 =
        derivative(calcualte_energy, wrt(atoms_[0].x[2]),
                   at(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2],
                      atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],
                      atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto dx10 =
        derivative(calcualte_energy, wrt(atoms_[1].x[0]),
                   at(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2],
                      atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],
                      atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto dx11 =
        derivative(calcualte_energy, wrt(atoms_[1].x[1]),
                   at(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2],
                      atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],
                      atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));
    auto dx12 =
        derivative(calcualte_energy, wrt(atoms_[1].x[2]),
                   at(a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2],
                      atoms_[0].x[0], atoms_[0].x[1], atoms_[0].x[2],
                      atoms_[1].x[0], atoms_[1].x[1], atoms_[1].x[2]));

    // clamp derivatives
    da0 = std::clamp(da0, -clamp_value, clamp_value);
    da1 = std::clamp(da1, -clamp_value, clamp_value);
    da2 = std::clamp(da2, -clamp_value, clamp_value);
    db0 = std::clamp(db0, -clamp_value, clamp_value);
    db1 = std::clamp(db1, -clamp_value, clamp_value);
    db2 = std::clamp(db2, -clamp_value, clamp_value);
    dc0 = std::clamp(dc0, -clamp_value, clamp_value);
    dc1 = std::clamp(dc1, -clamp_value, clamp_value);
    dc2 = std::clamp(dc2, -clamp_value, clamp_value);
    dx00 = std::clamp(dx00, -clamp_value, clamp_value);
    dx01 = std::clamp(dx01, -clamp_value, clamp_value);
    dx02 = std::clamp(dx02, -clamp_value, clamp_value);
    dx10 = std::clamp(dx10, -clamp_value, clamp_value);
    dx11 = std::clamp(dx11, -clamp_value, clamp_value);
    dx12 = std::clamp(dx12, -clamp_value, clamp_value);

    a[0] -= step * da0;
    a[1] -= step * da1;
    a[2] -= step * da2;
    b[0] -= step * db0;
    b[1] -= step * db1;
    b[2] -= step * db2;
    c[0] -= step * dc0;
    c[1] -= step * dc1;
    c[2] -= step * dc2;
    atoms_[0].x[0] -= step * dx00;
    atoms_[0].x[1] -= step * dx01;
    atoms_[0].x[2] -= step * dx02;
    atoms_[1].x[0] -= step * dx10;
    atoms_[1].x[1] -= step * dx11;
    atoms_[1].x[2] -= step * dx12;
    // clip position from 0.0 to 1.0
    atoms_[0].x[0] = std::clamp(static_cast<double>(atoms_[0].x[0]), 0.0, 1.0);
    atoms_[0].x[1] = std::clamp(static_cast<double>(atoms_[0].x[1]), 0.0, 1.0);
    atoms_[0].x[2] = std::clamp(static_cast<double>(atoms_[0].x[2]), 0.0, 1.0);
    atoms_[1].x[0] = std::clamp(static_cast<double>(atoms_[1].x[0]), 0.0, 1.0);
    atoms_[1].x[1] = std::clamp(static_cast<double>(atoms_[1].x[1]), 0.0, 1.0);
    atoms_[1].x[2] = std::clamp(static_cast<double>(atoms_[1].x[2]), 0.0, 1.0);
    static int count = 0;
    if (count++ % 1000 == 0) {
      std::cerr << energy << std::endl;
    }
  }
};

}  // namespace Lattice
