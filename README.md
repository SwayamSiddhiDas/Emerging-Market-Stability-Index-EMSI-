# Emerging Market Stability Index (EMSI)

A PCA-based composite index and Streamlit dashboard for monitoring macro–market stability across selected emerging markets.

**Live App:** <PASTE_STREAMLIT_LINK_HERE>

---

## Overview

This project aggregates market and macroeconomic indicators into a single **Stability Index**, enabling consistent comparison of stability trends across emerging economies.  
The index is built using **Principal Component Analysis (PCA)** on standardized features, and results are presented through an interactive Streamlit dashboard.

Covered markets in the current version:
- Brazil
- India
- South Africa
- Poland

---

## Key Components

### Stability Index
- Features are standardized to ensure comparability across indicators
- PCA is applied to derive a composite Stability Index
- PCA loadings are used to interpret major positive/negative contributors

### Dashboard
The Streamlit dashboard provides:
- Stability Index trends over time
- Cross-country comparison
- Key drivers derived from PCA loadings
- A stability-informed stock view (momentum vs defensive behavior)

---

## Repository Structure

```bash
.
├── src/
│   ├── dashboard.py              # Streamlit entrypoint
│   ├── backend.py                # Pipeline runner (data → features → index)
│   ├── data_pipeline.py          # Data ingestion and cleaning
│   ├── feature_engineering.py    # Feature creation (rolling volatility, etc.)
│   └── model.py                  # PCA model + index logic
├── data/                         # Precomputed outputs used by the dashboard
├── requirements.txt
└── README.md
