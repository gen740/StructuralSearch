#include "lattice.hpp"

#include <simd/simd.h>

#include <random>

namespace Lattice {

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(0.0, 1.0);

Atom create_random_atom() {
  Atom atom;
  atom.x[0] = dis(gen);
  atom.x[1] = dis(gen);
  atom.x[2] = dis(gen);
  return atom;
}

}  // namespace Lattice
