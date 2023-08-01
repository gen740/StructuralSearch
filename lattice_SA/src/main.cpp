#include <iostream>

#include "lattice_newton.hpp"

auto main() -> int {
  Lattice::LatticeN<2> lattice;
  for (int i = 0; i < 100'000; ++i) {
    lattice.update();
    if (i % 100 == 0) {
      std::cerr << lattice.get_energy() << std::endl;
    }
  }

  auto atoms = lattice.get_atoms();
  auto unit_vector = lattice.get_unit_vector();
  for (int i = 0; i < 3; i++) {
    std::cout << unit_vector[i][0] << " " << unit_vector[i][1] << " "
              << unit_vector[i][2] << std::endl;
  }
  for (int i = 0; i < 2; i++) {
    std::cout << atoms[i][0] << " " << atoms[i][1] << " " << atoms[i][2]
              << std::endl;
  }

  return 0;
}
