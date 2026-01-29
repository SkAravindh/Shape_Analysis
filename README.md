# 3D ShapeAnalysis Algorithms Repository

A modular **C++ framework for 3D shape analysis and correspondence** based on the Functional Maps framework.

This repository focuses on **clean, from-scratch implementations** of spectral geometry, descriptors, and optimization-based functional correspondence.

**Goal:** To provide clean, from-scratch implementations of core algorithms in 3D shape analysis and geometry processing, organized in a **reusable, extensible C++ codebase**.

---

## Build Procedure

This repository depends on the following third-party libraries:

- **Geometry Central**
- **Polyscope**
- **LBFGS++**

Before building, clone the required repositories and place them inside the `Dependencies/` directory:

### Dependency Links
- Geometry Central: https://github.com/nmwsharp/geometry-central.git
- Polyscope: https://github.com/nmwsharp/polyscope.git
- LBFGS++: https://github.com/yixuan/LBFGSpp.git

---

### Build Instructions
From the project root:
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE="Release" ..
make
cd bin
./shape_analysis