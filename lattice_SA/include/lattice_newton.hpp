#pragma once

#include <simd/simd.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <autodiff/variable.hpp>
#include <random>

namespace Lattice {

constexpr int ATOM_NUMBER = 2;

template <int N = 2>
class LatticeN {
  using VariableType = Autodiff::Variable<9 + 3 * N, 1>;

  struct Atom {
    std::array<VariableType, 3> x;
  };

  std::array<VariableType, 3> a;
  std::array<VariableType, 3> b;
  std::array<VariableType, 3> c;
  std::array<Atom, N> atoms_;
  double energy_;

 public:
  double get_energy() {
    return energy_;
  }

  LatticeN() {
    std::mt19937 gen(11);
    std::uniform_real_distribution<> dis(0.0, 1.0);
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < 3; ++j) {
        VariableType var = {};
        var.set({}, dis(gen));
        var.set({static_cast<uint64_t>(9 + 3 * i + j + 1)}, 1.0);
        atoms_[i].x[j] = var;
      }
    }
    for (int i = 0; i < 3; ++i) {
      VariableType vara = {};
      vara.set({}, dis(gen));
      vara.set({static_cast<uint64_t>(i + 1)}, 1.0);

      VariableType varb = {};
      varb.set({}, dis(gen));
      varb.set({static_cast<uint64_t>(3 + i + 1)}, 1.0);

      VariableType varc = {};
      varc.set({}, dis(gen));
      varc.set({static_cast<uint64_t>(6 + i + 1)}, 1.0);

      a[i] = vara;
      b[i] = varb;
      c[i] = varc;
    }
  }

  auto E() {
    int SHIFT = 2;
    std::array<std::array<VariableType, 3>, N> x{};
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < 3; ++j) {
        x.at(i).at(j) += atoms_.at(i).x[0] * a[j] + atoms_.at(i).x[1] * b[j] +
                         atoms_.at(i).x[2] * c[j];
      }
    }

    VariableType energy = {};

    for (int i = -SHIFT; i <= SHIFT; i++) {
      for (int j = -SHIFT; j <= SHIFT; j++) {
        for (int k = -SHIFT; k <= SHIFT; k++) {
          for (int l = 0; l < N; l++) {
            for (int m = 0; m < N; m++) {
              if (l == m && i == 0 && j == 0 && k == 0) {
                continue;
              }
              std::array<VariableType, 3> r = {};
              for (int idx = 0; idx < 3; ++idx) {
                r[idx] += x[l][idx] - x[m][idx] + a[idx] * i + b[idx] * j +
                          c[idx] * k;
              }
              VariableType dist2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
              energy += pow(dist2, -6) - pow(dist2, -3);
            }
          }
        }
      }
    }

    return energy;
  }

  void update() {
    double factor = 5e-4;
    auto energy = this->E();
    this->energy_ = static_cast<double>(energy);

    double clamp_max = 5e-3;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < 3; ++j) {
        auto val = atoms_[i].x[j].derivative(0) -
                   factor * std::clamp(energy.derivative(9 + 3 * i + j + 1),
                                       -clamp_max, clamp_max);
        val = std::clamp(val, 0.0, 1.0);
        atoms_[i].x[j].set({}, val);
      }
    }
    double norm_clamp_max = 10 * clamp_max;
    for (size_t i = 0; i < 3; ++i) {
      a[i].set({}, a[i].derivative(0) -
                       factor * std::clamp(energy.derivative(i + 1),
                                           -norm_clamp_max, norm_clamp_max));
      b[i].set({}, b[i].derivative(0) -
                       factor * std::clamp(energy.derivative(3 + i + 1),
                                           -norm_clamp_max, norm_clamp_max));
      c[i].set({}, c[i].derivative(0) -
                       factor * std::clamp(energy.derivative(6 + i + 1),
                                           -norm_clamp_max, norm_clamp_max));
    }
  }

  // void update2() {
  //   double factor = 1;
  //   auto energy = this->E();
  //   this->energy_ = static_cast<double>(energy);
  //
  //   Eigen::MatrixXd H(9 + 3 * N, 9 + 3 * N);
  //   for (int64_t i = 0; i < 9 + 3 * N; ++i) {
  //     for (int64_t j = 0; j < 9 + 3 * N; ++j) {
  //       H(i, j) = energy.derivative(i + 1, j + 1);
  //     }
  //   }
  //
  //   Eigen::VectorXd V(9 + 3 * N);
  //   for (int64_t i = 0; i < 3; ++i) {
  //     V(i) = energy.derivative(i + 1);
  //     V(3 + i) = energy.derivative(3 + i + 1);
  //     V(6 + i) = energy.derivative(6 + i + 1);
  //   }
  //   for (int64_t i = 0; i < N; ++i) {
  //     for (int64_t j = 0; j < 3; ++j) {
  //       V(9 + 3 * i + j) = energy.derivative(9 + 3 * i + j + 1);
  //     }
  //   }
  //
  //   Eigen::VectorXd dX(9 + 3 * N);
  //   dX = (H.inverse() * V).eval();
  //
  //   double clamp_max = 1e+4;
  //   for (int i = 0; i < N; ++i) {
  //     for (int j = 0; j < 3; ++j) {
  //       auto val =
  //           atoms_[i].x[j].derivative(0) -
  //           factor * std::clamp(dX(9 + 3 * i + j), -clamp_max, clamp_max);
  //       val = std::clamp(val, 0.0, 1.0);
  //       atoms_[i].x[j].set({}, val);
  //     }
  //   }
  //   double norm_clamp_max = 10 * clamp_max;
  //   for (int64_t i = 0; i < 3; ++i) {
  //     a[i].set({},
  //              a[i].derivative(0) -
  //                  factor * std::clamp(dX(i), -norm_clamp_max,
  //                  norm_clamp_max));
  //     b[i].set({}, b[i].derivative(0) - factor * std::clamp(dX(3 + i),
  //                                                           -norm_clamp_max,
  //                                                           norm_clamp_max));
  //     c[i].set({}, c[i].derivative(0) - factor * std::clamp(dX(6 + i),
  //                                                           -norm_clamp_max,
  //                                                           norm_clamp_max));
  //   }
  // }

  [[nodiscard]] std::array<std::array<double, 3>, 3> get_unit_vector() const {
    return {{{static_cast<double>(a[0]), static_cast<double>(a[1]),
              static_cast<double>(a[2])},
             {static_cast<double>(b[0]), static_cast<double>(b[1]),
              static_cast<double>(b[2])},
             {static_cast<double>(c[0]), static_cast<double>(c[1]),
              static_cast<double>(c[2])}}};
  }

  [[nodiscard]] std::array<std::array<double, 3>, N> get_atoms() const {
    std::array<std::array<double, 3>, N> ret;
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < 3; ++j) {
        ret.at(i).at(j) = static_cast<double>(this->atoms_.at(i).x.at(j));
      }
    }
    return ret;
  }
};

}  // namespace Lattice
