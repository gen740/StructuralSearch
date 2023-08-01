cmake_minimum_required(VERSION 3.20)

project(lattice_SA CXX OBJCXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS "on")

# Checking Build Type
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE DEBUG)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS
    "-mavx2 -Wall -Wextra -Werror -Wnon-virtual-dtor -Wfloat-equal -Winline -ffast-math")

# Building Metal

set(SHADER_SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/shader.metal)
set(SHADER_AIR ${CMAKE_CURRENT_BINARY_DIR}/shader.air)
set(SHADER_METALLIB ${CMAKE_CURRENT_BINARY_DIR}/shader.metallib)

find_program(XRUN xrun)

add_custom_command(
  OUTPUT ${SHADER_METALLIB}
  COMMAND xcrun -sdk macosx metal -c ${SHADER_SOURCE} -o ${SHADER_AIR} #
          && xcrun -sdk macosx metallib ${SHADER_AIR} -o ${SHADER_METALLIB}
  DEPENDS ${SHADER_SOURCE}
  COMMENT "Building Metal shader")

add_custom_target(lattice_SA_metallib ALL DEPENDS ${SHADER_METALLIB})

set_source_files_properties(
  ${SHADER_METALLIB} PROPERTIES MACOSX_PACKAGE_LOCATION "Resources/metallib")

file(GLOB SOURCE ${CMAKE_CURRENT_LIST_DIR}/src/*.cpp)
add_executable(${PROJECT_NAME} ${SOURCE})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

target_include_directories(${PROJECT_NAME}
                           PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)
target_include_directories(${PROJECT_NAME} SYSTEM
                           PUBLIC ${CMAKE_CURRENT_LIST_DIR}/lib/include)
