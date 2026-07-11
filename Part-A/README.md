# Part A — MLOps Foundations Lab (CupGuard)

![KAUST Academy](https://i.imgur.com/a3uAqnb.png)


Ingest match data, engineer team features, train a baseline outcome model, ship a Streamlit agent |

## Setup & quick start

Dependencies are declared in `pyproject.toml` and managed with [uv](https://docs.astral.sh/uv/). Follow the steps below from the **Part-A** folder (`MLOps-Abha/Part-A`).

### 1. Install Conda (optional but recommended)

Conda gives you an isolated Python environment. If you already have Miniconda or Anaconda, skip to step 2.

**macOS / Linux**

```bash
# Download Miniconda, then run the installer:
# https://docs.anaconda.com/miniconda/miniconda-install/

# After installation, restart your terminal, then verify:
conda --version
```

**Windows**

Download and run the Miniconda installer from [https://docs.anaconda.com/miniconda/miniconda-install/](https://docs.anaconda.com/miniconda/miniconda-install/), then open **Anaconda Prompt** and run `conda --version`.

Create and activate a Conda environment for the labs (Python 3.11):

```bash
conda create -n cupguard python=3.11 -y
conda activate cupguard
```

### 2. Install uv

[uv](https://docs.astral.sh/uv/) is a fast Python package installer. Install it once on your machine:

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify the installation:

```bash
uv --version
```

Official docs: [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### 3. Install dependencies

From the **Part-A** directory, with your Conda environment active (if using Conda):

```bash
cd MLOps-Abha/Part-A
uv pip install -r pyproject.toml
```

This reads project dependencies from `pyproject.toml` and installs them into the active environment.


### 4. Run the Streamlit application

Train the model first via the notebook or the **Run full pipeline** button in the app sidebar, then start the agent:

```bash
streamlit run cupguard/src/worldcup_agent_app.py
```

Streamlit opens a local URL (usually [http://localhost:8501](http://localhost:8501)) in your browser. Use the sidebar to pick teams and predict match outcomes.

### 5. MLflow and Experiment Tracking

- Open Source tool that logs **parameters**, **metrics**, and **artifacts** for every training run.
- Compare runs side by side and **promote winners** to a model registry.

After a successful **Run full pipeline** in Streamlit (or `run_lifecycle()` in the notebook), the lifecycle uses `mlflow.sklearn.autolog()` to log hyperparameters, training metrics, and the model artifact to `cupguard/mlruns/`. Test-set **accuracy** and **f1_macro** are logged in `evaluate()`. Open the **MLflow dashboard** to compare runs.

---

In a second terminal (from **Part-A**):

```bash
mlflow server --port 5000
```

Then open [http://localhost:5000](http://localhost:5000), select experiment **cupguard-worldcup**, and compare runs.

Each promoted run logs:

- **Autolog:** model params, training metrics, sklearn + pyfunc model at `runs:/<run_id>/model`
- **Manual:** test-set `accuracy`, `f1_macro`

---

## Your task

| Part | Goal |
|------|------|
| **1** | Run the full lifecycle end-to-end `ingest()` → `transform()` → `train()` → `evaluate()` → `package()` |
| **2** | Review metrics and validation results |
| **3** | Engineer features to improve F1 score |


### Project layout

```
Part-A/
├── pyproject.toml          # project dependencies (uv)
├── uv.lock
├── mlflow.db               # MLflow tracking store (created when you run the server)
└── cupguard/
    ├── data/
    │   ├── raw/            # goalscorers.csv, matches_enriched.csv, results.csv, shootouts.csv
    │   └── processed/      # team_stats_2025.csv
    └── src/
        ├── pipeline.py           # lifecycle, MLflow autolog, prediction helpers
        └── worldcup_agent_app.py # Streamlit UI
```
