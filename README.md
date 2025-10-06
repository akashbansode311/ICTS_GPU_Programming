# 🚀 CUDA Code Examples

This repository contains simple and practical CUDA C/C++ programs to help beginners and researchers understand the basics of **GPU programming with CUDA**.

## 📌 Contents

* **Hello World (device query)** → Check GPU details (compute capability, memory, etc.)
* **Vector Addition** → Introduction to parallelism
* **Matrix Multiplication** → Demonstrates 2D thread blocks and grids
* **Reduction (Sum of Array)** → Shows shared memory usage
* **Prefix Sum (Scan)** → Inclusive/exclusive scan examples
* **N-Body Simulation** → Example from astrophysics (gravitational interaction)
* More codes coming soon 🚧

## ⚡ Requirements

* NVIDIA GPU with CUDA support
* CUDA Toolkit installed (e.g., 11.x or later)
* C/C++ compiler (e.g., `gcc`, `g++`)

## ▶️ How to Compile & Run

Example for vector addition:

```bash
nvcc vector_add.cu -o vector_add
./vector_add
```

Example for matrix multiplication:

```bash
nvcc matrix_mul.cu -o matrix_mul
./matrix_mul
```

## 📚 Learning Goals

* Understand CUDA threads, blocks, and grids
* Learn memory hierarchy (global, shared, registers)
* Practice performance optimization with CUDA

## 🌌 Special Note

This repo also includes astrophysics-related demos (like **N-body simulation**) to show how CUDA accelerates real-world scientific computations.

## 🤝 Contribution

Feel free to fork, add new examples, or suggest improvements via pull requests.

---

💡 Maintainer: *Your Name*

# Sample Codes for CUDA
