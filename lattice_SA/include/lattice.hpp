#pragma once

#include <simd/simd.h>

#include <algorithm>
#include <iostream>
#include <vector>

namespace Lattice {

inline std::ostream& operator<<(std::ostream& os, const simd::double3& v) {
  os << v[0] << " " << v[1] << " " << v[2];
  return os;
}

using simd::double3;

struct Atom {
  double3 x;
  double3 f;
};

template <int N>
class Lattice {
  double step = 1e-3;

  double clamp_max_ = 0.01;
  double3 clamp_max = {clamp_max_, clamp_max_, clamp_max_};

  double3 a{1.0, 0.0, 0.0};
  double3 b{0.0, 1.0, 0.0};
  double3 c{0.0, 0.0, 1.0};

 public:
  std::array<Atom, N> atoms{};

  explicit Lattice(std::array<Atom, N> atoms) : atoms(atoms) {}

  double potential(double r) {
    return std::pow(r, -12);
    // return 4 * (std::pow(r, -12) - std::pow(r, -6));
  }

  double force(double r) {
    return -12 * std::pow(r, -13);
    // return 24 * (2 * std::pow(r, -13) - std::pow(r, -7));
  }

  double E() {
    double E = 0;

    for (int i = 0; i < N; i++) {
      this->atoms[i].f = double3{0.0, 0.0, 0.0};
    }
    auto mn = 1;

    for (int i = 0; i < N; i++) {
      for (int j = i; j < N; j++) {
        for (int nx = -mn; nx <= mn; nx++) {
          for (int ny = -mn; ny <= mn; ny++) {
            for (int nz = -mn; nz <= mn; nz++) {
              if (nx == 0 && ny == 0 && nz == 0 && i == j) {
                // self interaction
                continue;
              }
              double3 diff = this->atoms[i].x - this->atoms[j].x + nx * a +
                             ny * b + nz * c;

              double r = simd::length(diff);

              double f = force(r);
              E += potential(r);
              this->atoms[i].f += f * diff / r;
              this->atoms[j].f -= f * diff / r;
            }
          }
        }
      }
    }
    return E;
  }

  void update() {
    for (int i = 0; i < N; i++) {
      // std::cout << this->atoms[i].x  << std::endl;
      this->atoms[i].x -=
          step * this->atoms[i].f / simd::length(this->atoms[i].f);
      // std::cout << this->atoms[i].x  << std::endl;
      this->atoms[i].x = simd::clamp(this->atoms[i].x, double3{0.0, 0.0, 0.0},
                                     double3{1.0, 1.0, 1.0});
    }
  }
};

Atom create_random_atom();

}  // namespace Lattice
