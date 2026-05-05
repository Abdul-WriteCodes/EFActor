# 🔬 EFA / CFA Analyser

A rigorous, researcher-grade app for Exploratory and Confirmatory Factor Analysis.


<p align="center">
  <img src="assets/ef1.png" alt="DataSynthX Logo" width="800"/>
</p>

EFActor is a web-based AI-powered psychometric analysis platform designed for researchers, students, and data analysts working with survey and latent construct data. It removes the complexity of tools like R, SPSS, and Stata by providing a no-code, structured workflow for factor analysis

## Problem
Researchers face major friction when running factor analysis:
- High learning curve (R, SPSS syntax, SEM tools)
- Poor diagnostics for item-level issues
- Manual iteration (drop, rerun, repeat)
- No built-in data repair mechanisms
- Synthetic data generation is non-trivial

## The Solution
EFActor provides a fully integrated psychometric pipeline:
- Automated EFA + CFA workflow
- Built-in diagnostics + issue detection
- Intelligent auto-fix engine for problematic variables
- Synthetic dataset generation
- Export-ready Word reports (.docx)
  
---

# Core Features
1. EFA (Exploratory Factor Analysis)
   - KMO & Bartlett’s Test
   - Eigenvalues + Scree Plot
   - Factor Loadings (Varimax rotation)
   - Communalities & Variance Explained

2. Smart Diagnostics that detect and flag:
   - Low communality
   - Cross-loading variables
   - Weak factor loadings
   - Severity scoring + drop recommendations

3. Auto-Fix Engine (Key Differentiator) that iteratively fixes:
   - Outliers (Winsorization)
   - Skewness (Log transforms)
   - Kurtosis
   - Collinearity (Jittering)
   - Zero-variance issues
   - Preserves statistical integrity while improving model quality

4. CFA that auto-build measurement model and Fit indices (CFI, TLI, RMSEA, SRMR) and evaluate model
5. Synthetic Data Generation that supports:
   - Factor-based generation
   - Correlation-based generation
   - Validation against original dataset

6. Interactive Visualizations
   - Scree plots
   - Factor loading heatmaps
   - Communality charts
   - Correlation matrices
   - Synthetic vs original comparisons
     
# Workflow
1. 👉 Visit: [EFActor](https://efactor.streamlit.app/)
2. Upload dataset (CSV / Excel)
3. Run EFA → check structure
4. Diagnose problematic variables
5. Auto-fix issues (optional)
6. Drop problematic variables 
7. Build & validate CFA model
8. Generate synthetic data (optional)
9. Export results & report
