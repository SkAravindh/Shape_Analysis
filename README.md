# 3D ShapeAnalysis Algorithms Repository

A modular **C++ framework for 3D shape analysis and correspondence** based on the Functional Maps framework.

This repository provides **clean, from-scratch implementations** of core algorithms in:

- Spectral geometry
- Shape descriptors
- Functional maps
- Numerical optimization

**Goal:** To provide clean, from scratch implementations of core algorithms in 3D shape analysis and geometry processing, organized in a **reusable, extensible C++ codebase**.

---
## Dependencies

This project relies on the following third-party libraries:

| Library | Purpose                                   |
|---------|-------------------------------------------|
| **Geometry Central** | Mesh Processing and Differential Geometry |
| **Polyscope** | Visualization                             |
| **LBFGS++** | Numerical Optimization                    |
| **Eigen** | Linear Algebra                            |

---
## Setup

Clone the required repositories and place them inside the `Dependencies/` directory:

### Dependency Links
- Geometry Central: https://github.com/nmwsharp/geometry-central.git
- Polyscope: https://github.com/nmwsharp/polyscope.git
- LBFGS++: https://github.com/yixuan/LBFGSpp.git

### System Requirements

- **CMake ≥ 3.18**
- **Eigen**

Install Eigen on Debian/Ubuntu:

sudo apt install libeigen3-dev

---
### Build Instruction
From the project root do the following:
- mkdir build
- cd build
- cmake -DCMAKE_BUILD_TYPE=Release ..
- make
- cd bin
- ./shape_analysis
