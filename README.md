# 🚀 MLCOE Summer Associate Internship Assessment (2026)

This repository contains the required code and documentation for the Machine Learning Center of Excellence (MLCOE) 2026 Summer Associate Internship assessment, focused on Time Series and Reinforcement Learning (RL).


## Overview

This repository contains the solution code and project report for **Part 1: From Classical Filters to Particle Flows** of the JP Morgan MLCOE TSRL 2026 Internship assessment.

The project explores the evolution of State Space Model (SSM) filtering, starting from foundational Gaussian approximations and Sequential Monte Carlo methods, and culminating in advanced, high-dimensional **Particle Flow Filters (PFF)**. The implementation focuses on numerical stability, particle degeneracy mitigation, and the application of invertible and kernel-based flows.

## Repository Contents

### 📚 Core Libraries
* **`filters.py`**: A complete, vectorized TensorFlow implementation of Bayesian Filters. It includes classical methods (KF, EKF, UKF) and advanced particle flows:
    * **Classical:** Kalman Filter (KF), Extended KF (EKF), Unscented KF (UKF), Ensemble Square Root Filter (ESRF).
    * **Particle Filters:** Bootstrap Particle Filter (BPF), Unscented Particle Filter (UPF), Gaussian Sum Monte Carlo (GSMC).
    * **Particle Flows:** Exact Daum-Huang (EDH), Local Exact Daum-Huang (LEDH), and the **Invertible Particle Flow Particle Filter (PFPF)** variants.
    * **Kernel Flows:** Kernel-Embedded Particle Flow Filter (KPFF) with support for both scalar and matrix-valued kernels.
* **`ssm_models.py`**: Defines the State Space Models used for experiments, including Jacobians and measurement functions:
    * **Stochastic Volatility Model:** A nonlinear 1D model for financial time series.
    * **Range-Bearing Model:** A standard nonlinear tracking benchmark.
    * **Acoustic Tracking Model:** A high-dimensional (16D state, 25D obs) multi-target tracking scenario used in Li & Coates (2017).
* **`test_filters.py`**: Integration tests ensuring the correctness of the filter pipelines and their compatibility with the SSM models.

### 🧪 Experiment Scripts

#### Literature Replication
* **`run_li17_experiments.py`**: Replicates key results from *Li & Coates (2017)*:
    * **Figure 1:** Visualizes multi-target acoustic tracking trajectories using PFPF-LEDH.
    * **Figure 2:** Benchmarks OMAT (Optimal Mass Transfer) error against baselines (EKF, UKF, GSMC) over 50 time steps.
    * **Figure 4:** Analyzes Effective Sample Size (ESS) evolution to demonstrate how PFPF mitigates weight degeneracy compared to standard SIS.
* **`run_hu21_experiments1.py`**: Replicates the "Marginal Collapse" experiment from *Hu & Van Leeuwen (2021)* on the Lorenz 96 model. It visualizes posterior contours to demonstrate how matrix-valued kernels preserve ensemble diversity in unobserved dimensions.

#### Benchmarking & Validation
* **`run_nonlinear_nonGaussian.py`**: A unified benchmarking script for Nonlinear/Non-Gaussian SSMs. It compares EKF, UKF, and PF on Stochastic Volatility and Range-Bearing models, generating RMSE metrics and trajectory plots.
* **`run_linear_Gaussian_SSM.py`**: Validates the optimality of the Kalman Filter implementation by comparing it against a Conditional Monte Carlo (CMC) baseline. It analyzes the Frobenius norm discrepancy between the analytic KF covariance and the empirical CMC covariance.
* **`run_EKF_UKF_PF_benchmark.py`**: A computational performance benchmark comparing Runtime, Peak Memory, and RMSE of EKF/UKF against Particle Filters with scaling particle counts (N=500 to N=100k) on both CPU and GPU.

#### Advanced Particle Flow Analysis
* **`run_robust_PFF_experiments.py`**: Conducts three specific robustness experiments:
    * **Exp A (Nonlinearity):** Tests filter performance on a highly nonlinear observation model ($y=x^2$).
    * **Exp B (Sparsity):** Evaluates performance on a high-dimensional (D=40) Lorenz 96 model with sparse observations (observing every 4th state).
    * **Exp C (Conditioning):** Analyzes numerical stiffness by measuring flow magnitude and condition numbers as measurement noise $R \to 0$.
* **`run_singularity_test.py`**: Specifically tests the singularity robustness of EDH vs. LEDH vs. Kernel PFF when a target passes close to a sensor origin, creating an infinite Jacobian slope.
* **`run_matrix_kernel_PFF.py`**: A dedicated runner for the Matrix-Valued Kernel PFF, comparing its tracking accuracy (RMSE) against analytic EDH and LEDH wrappers on the Range-Bearing model.
* **`run_EDH_LEDH_KPFF_comparison.py`**: A comparative analysis script that aggregates results from EDH, LEDH, and KPFF runs to generate summary comparison plots.

### 📄 Documentation
* **`Project_Report.pdf`**: The final submission document. It contains:
    * **Literature Review:** A comprehensive in-depth survey of Bayesian filter methods from Kalman Filters to modern Differentiable Particle Filters.
    * **Methodology:** Mathematical derivations of the EDH, LEDH, and KPFF algorithms.
    * **Results:** Detailed analysis of the experiments, including stability diagnostics (Jacobian conditioning), runtime comparisons, and error metrics.

## Implemented Methods (Part 1)

### 1. Classical Linear-Gaussian Filtering
* **Kalman Filter (KF):** A custom implementation for Multidimensional Linear-Gaussian SSMs (LGSSM).
    * Includes **Joseph stabilized covariance updates** to ensure positive semi-definiteness.
    * Analysis of filtering optimality and **numerical stability** (condition number analysis).

### 2. Nonlinear & Non-Gaussian Baselines
* **Models:** Implementation of challenging nonlinear environments (e.g., Stochastic Volatility or Range-Bearing Tracking).
* **Extended Kalman Filter (EKF) & Unscented Kalman Filter (UKF):**
    * Analysis of linearization limits and sigma-point approximation failures under strong nonlinearity.
* **Standard Particle Filter (PF):**
    * Custom implementation (avoiding `tfp` defaults) to study **particle degeneracy** and sample impoverishment.
    * Performance benchmarking (Runtime, Peak Memory, RMSE) against EKF/UKF.

### 3. Advanced Particle Flow Filters
This section implements state-of-the-art flow-based methods to address the limitations of standard PFs in high dimensions.

* **Daum-Huang Flows:**
    * **Exact Daum-Huang (EDH):** Global log-homotopy flow.
    * **Local Exact Daum-Huang (LEDH):** Pointwise flow for handling non-Gaussian structures.
* **Invertible Particle Flow (PF-PF):**
    * Implementation of the framework by **Li & Coates (2017)**, using flow as a deterministic proposal within Importance Sampling.
* **Kernel-Embedded Particle Flow (KPFF):**
    * Implementation in Reproducing Kernel Hilbert Space (RKHS) following **Hu & Van Leeuwen (2021)**.
    * **Matrix-Valued Kernels:** Comparison between scalar and diagonal matrix-valued kernels to demonstrate the prevention of "marginal collapse" in high-dimensional systems with sparse observations.

## Key Experiments & Analysis

The project report includes detailed analysis on:
1.  **Stability Diagnostics:** Jacobian conditioning and flow magnitude analysis during particle transport.
2.  **Marginal Collapse:** Visualizations (replicating Hu(21) Figs 2-3) showing how matrix-valued kernels maintain diversity in observed subspaces.
3.  **Comparative Study:** A rigorous comparison of EDH, LEDH, and KPFF across varying dimensions, nonlinearity levels, and observation sparsities.

## References

The implementations in this repository are based on the following literature:

* **[Doucet(09)]** Doucet, A., & Johansen, A. M. "A tutorial on particle filtering and smoothing: Fifteen years later." (2009).
* **[Daum(10)]** Daum, F., & Huang, J. "Exact particle flow for nonlinear filters." SPIE (2010).
* **[Daum(11)]** Daum, F., & Huang, J. "Particle degeneracy: root cause and solution." SPIE (2011).
* **[Li(17)]** Li, Y., & Coates, M. "Particle filtering with invertible particle flow." *IEEE Transactions on Signal Processing* (2017).
* **[Hu(21)]** Hu, C., & Van Leeuwen, P. J. "A particle flow filter for high‐dimensional system applications." *Q.J.R. Meteorol. Soc.* (2021).

## Getting Started

1.  Clone the repository:
    ```bash
    git clone [https://github.com/ljw9510/2026MLCOE.git](https://github.com/ljw9510/2026MLCOE.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Navigate to the directory and run any experiment script:
    ```bash
    python run_li17_experiments.py
    ```

---
*Author: Joowon Lee*
*Context: JP Morgan MLCOE TSRL 2026 Internship Assessment*
