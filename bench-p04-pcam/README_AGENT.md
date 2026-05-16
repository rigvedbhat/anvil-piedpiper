# PCAM Precision Agent — R Backend Migration

This repository contains our submission for the **P-04 Precision-Controlled Associative Memory** benchmark. 

We have completely restructured the core intelligence layer to run natively in **R**, while leaving the Python evaluation harness 100% untouched to ensure zero deviation from the benchmark's constraints and scoring mechanics.

---

## 🧠 Our Approach

The agent dynamically controls the precision vector $\pi$ using a two-stage hybrid strategy based on the query's proximity to known stored patterns:

1.  **Robust Inverse-Amplitude Rule (Retrieval Phase):**
    When a highly corrupted query is received, the agent has low confidence about which attractor basin it belongs to. Instead of guessing the geometry, we apply a robust inverse-amplitude heuristic: $\pi \propto 1 / (|q| + \epsilon)$. This down-weights high-magnitude noise dimensions dynamically, pushing the state closer to the true signal before the dynamics commit to an attractor.

2.  **Cached Local Hessian Preconditioner (Anisotropy Phase):**
    Once the query is sufficiently close to a stored pattern (measured via cosine similarity margins), the agent switches to geometry-aware precision. It computes the **true equilibrium** of that pattern and evaluates the Hessian $H$. We then perform an iterative gradient descent to find a diagonal $\pi$ that minimizes the log-condition number of the symmetrized operator $\Pi^{1/2} H \Pi^{1/2}$. 
    *Note: Because Hessian evaluation and eigen-decompositions are expensive, we cache the optimal precision for each pattern.*

3.  **Persistent R-Bridge:**
    To bypass Python's limitations with complex matrix operations and leverage R's robust linear algebra routines, the mathematical model (`PCAMModel`) and the agent's logic (`Engine`) are implemented in `core.R`. The Python `Engine` wrapper spawns a persistent `Rscript` subprocess upon initialization. Queries are sent over standard input using JSON, eliminating the latency of starting a new R process for every query.

---

## ⚙️ Setup & Installation

### Dependencies (Beyond NumPy)
*   **R** (version 4.0 or higher)
*   **R Package:** `jsonlite` (for JSON serialization over the bridge)

### Instructions
1.  Ensure R is installed on your system.
2.  Install the `jsonlite` package in R:
    ```R
    install.packages("jsonlite")
    ```
    *(Note: If you run into permission issues, you can install it into a local directory like `.Rlibs` and add `.libPaths("./.Rlibs")` to the top of `adapters/core.R`)*
3.  Ensure `Rscript` is accessible in your system `PATH`, **OR** manually update the path in `adapters/myteam.py`:
    ```python
    self.process = subprocess.Popen(
        [r'C:\Program Files\R\R-4.x.x\bin\Rscript.exe', r_script], ...
    )
    ```
4.  Run the benchmark exactly as you normally would:
    ```bash
    python run.py --adapter adapters.myteam:Engine --seeds 7 13 31 97 211 503 1009 --out report.json
    ```

---

## 🔬 Tying Design to the PCAM Paper

*Reference: [MetaCognition Labs - Priyam Ghosh](https://www.researchgate.net/lab/MetaCognition-Labs-Priyam-Ghosh)*

Our implementation is directly inspired by the theoretical foundations laid out in the **Precision-Controlled Associative Memory (PCAM)** paper. 

In standard associative memory dynamics, convergence can severely stall along the "shallow" eigenvectors of the local Hessian, creating highly anisotropic (elliptical) attraction basins that are easily disrupted by specific directional noise. 

The paper demonstrates that introducing a dynamically controlled, positive-diagonal precision operator $\Pi$ allows us to radically reshape this energy landscape at inference time. Specifically, by using our **Local Hessian Preconditioner**, we directly target the phenomenon described in **Theorem F3** of the paper. We explicitly optimize $\pi$ to minimize the eigenvalue spread (condition number) of $\Pi^{1/2} H \Pi^{1/2}$ at the true equilibrium. 

By flattening the convergence basin and forcing the local contraction rates to become uniformly isotropic, our agent prevents adversarial noise from exploiting shallow valleys in the energy landscape, leading to the substantial spread reductions and highly robust retrieval accuracies seen in our benchmark scores.
