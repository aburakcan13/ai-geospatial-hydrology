# AI-Driven Geospatial Hydrology
> Accompanying codebase and reproducible workflows for the Springer Nature Monograph:  
> **"AI-Driven Geospatial Hydrology: Machine Learning and Decision Support for Surface Water Dynamics"** (2026) by *Burak Can*.

---

## 📖 Chapter to Codebase Mapping

| Chapter | Topic | Relevant Modules / Scripts | Notebooks |
| :--- | :--- | :--- | :--- |
| **Chap 1–2** | Big Data ETL & STAC Ingestion | `src/ingestion/` | `notebooks/01_stac_cube.ipynb` |
| **Chap 3** | Boundary-Aware ResUNet | `src/models/resunet.py`, `src/losses/` | `notebooks/02_resunet_train.ipynb` |
| **Chap 4** | Vision Transformers (SegFormer) | `src/models/segformer.py` | `notebooks/03_segformer_sra.ipynb` |
| **Chap 5** | SAR-Optical Fusion & FCLS Unmixing | `src/fusion/`, `src/unmixing/` | `notebooks/04_polarimetric_fcls.ipynb` |
| **Chap 6** | HydroLSTM, TFT & PINN Mass Loss | `src/forecasting/` | `notebooks/05_hydrolstm_forecast.ipynb` |
| **Chap 7** | Explainable AI (SHAP & IG) | `src/xai/` | `notebooks/06_integrated_gradients.ipynb` |
| **Chap 8** | FastAPI, TiTiler & PostGIS Tiling | `src/services/tiler/` | `notebooks/07_dynamic_tiling.ipynb` |
| **Chap 9** | Model Context Protocol (FastMCP) | `src/agent/mcp_server.py` | `notebooks/08_mcp_agent.ipynb` |
| **Chap 10** | EDSS Case Studies (Burdur, Tuz, etc.) | `src/edss/` | `notebooks/09_case_studies.ipynb` |

---

## 🚀 Quickstart

```bash
# Clone the repository
git clone [https://github.com/aburakcan13/ai-geospatial-hydrology.git](https://github.com/aburakcan13/ai-geospatial-hydrology.git)
cd ai-geospatial-hydrology

# Install dependencies via pyproject.toml
pip install -e .
