#include <simd/simd.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <valarray>

#include "generator.hpp"
// #include "lattice.hpp"
#include "lattice_newton.hpp"

// 3
auto newton_lattice(int seed) -> void {
  auto lattice = Lattice::LatticeN<2>(seed);
  for (int i = 0; i < 400'000; i++) {
    lattice.update();
  }
  auto atoms = lattice.atom_positions();
  auto unit_vector = lattice.get_unit_vector();
  for (int i = 0; i < 3; i++) {
    std::cout << unit_vector[i][0] << " " << unit_vector[i][1] << " "
              << unit_vector[i][2] << std::endl;
  }
  for (int i = 0; i < 2; i++) {
    std::cout << atoms[i][0] << " " << atoms[i][1] << " " << atoms[i][2]
              << std::endl;
  }
}

auto main(int argc, char *argv[]) -> int {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <seed>" << std::endl;
    return 1;
  }
  int seed = std::stoi(argv[1]);

  newton_lattice(seed);
  return 0;
}
