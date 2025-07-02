
# CS 575 – Parallel Programming Projects (Spring 2025)

This repository contains assignments and projects submitted for **CS 575: Introduction to Parallel Programming** at Oregon State University, taught by Prof. Mike Bailey. The coursework includes hands-on implementations using OpenMP, SIMD (SSE), CUDA, OpenCL, and MPI.

---

## 📂 Project Summary

| # | File(s) | Topic | Description |
|---|---------|-------|-------------|
| 0 | `Project.cpp` | Simple OpenMP | Initial loop parallelism experiment using OpenMP (`proj00`) |
| 1 | `MonteCarlo.cpp` | OpenMP Monte Carlo | Estimate Pi using Monte Carlo simulation with OpenMP |
| 2 | `Project2.cpp` / `proj2.cpp` | Functional Decomposition | Divides workload across functions/threads (e.g., trapezoidal integration) |
| 3 | `Project3.cpp` / `proj3.cpp` | Parallel Programming Challenge | Custom task exploring parallel speedup and optimization |
| 4 | `Project 4 code.cpp` | SIMD Array Ops | Multiply and reduce large arrays using SSE vectorization |
| 5 | *(not uploaded)* | CUDA Monte Carlo | Monte Carlo simulation using CUDA for GPU acceleration |
| 6 | `prj06.cpp` | OpenCL | GPU parallelism using OpenCL kernels |
| 7 | `prj07.cpp` | MPI | Distributed computing using MPI (e.g., averaging, reduction) |
| – | `Paper Analysis.pdf` | Paper Project | Analytical report on speedups and computational trends (575-only) |

---

## 🛠️ How to Compile (OpenMP Example)

```bash
g++ -O3 -fopenmp MonteCarlo.cpp -o montecarlo
./montecarlo
```

Use appropriate flags for SSE (`-msse2`), CUDA (`nvcc`), OpenCL, and MPI (`mpic++`).

---

## 🔗 Resources

- [CS 575 Course Website](https://cs.oregonstate.edu/~mjb/cs575/)
- [Project 0 – Simple OpenMP](http://cs.oregonstate.edu/~mjb/cs575/Projects/proj00.html)

---

## 👩‍💻 Author

**Sanika Deshmukh**  
Graduate Student – Computer Science  
Oregon State University

---

## ⚠️ License

This code is intended for academic and reference purposes only.
