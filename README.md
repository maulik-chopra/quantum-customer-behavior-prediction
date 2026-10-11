# Q-INSIGHT | Quantum Customer Intelligence

A presentation-ready Streamlit research prototype for customer churn analysis. It compares class-balanced Logistic Regression with an optional four-qubit Qiskit variational classifier, while clearly labeling synthetic demo data and simulator-only results.

## Start on Windows

Open PowerShell in this folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Python 3.11 is recommended for broad Qiskit wheel availability. The Qiskit model is optional at runtime: if a package install or simulator API fails, the classical app remains usable and the error is displayed.

## Dataset

Download the Kaggle [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). Save/extract `WA_Fn-UseC_-Telco-Customer-Churn.csv` anywhere on your computer, then upload it in the app. Select `Churn` as the target if it is not auto-selected. `customerID` is excluded from the model. The app accepts a binary target labeled Yes/No, True/False, or 1/0. No data is sent to a server by this app.

When no CSV is uploaded, enable **DEMO DATA**. These generated rows are synthetic and must not be described as real customer data or Kaggle experimental findings.

## What the experiment does

The app makes a stratified train/test split before fitting preprocessing. Numeric columns are median-imputed and standardized; categorical columns are most-frequent-imputed and one-hot encoded. Logistic Regression uses balanced class weights. The optional quantum branch fits four-component PCA on training rows, scales components to angles, and trains a ZZFeatureMap plus RealAmplitudes VQC on Qiskit Aer simulation. Both evaluate on the same held-out rows. Quantum training is deliberately bounded to 100 training records and 25 optimizer iterations for laptop practicality.

## Troubleshooting

- **PowerShell blocks activation:** run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.
- **Install errors:** confirm Python 3.11 and 64-bit Windows; create a fresh environment and reinstall requirements.
- **Qiskit model unavailable:** use the exact error shown by the app. The Qiskit stack is rapidly evolving; preserve the classical comparison and report the failed quantum run honestly. Do not write placeholder quantum metrics.
- **Target validation:** choose a target with exactly two recognized labels: Yes/No, True/False, or 1/0.
- **Unexpected columns:** CSV scoring expects the same feature columns used when training. The target and customer ID are not prediction features.
- **Slow quantum training:** the variational circuit is an educational simulator workload. Reduce the source CSV size for a live demo or skip quantum training and explain the limitation.

## Demo checklist

1. Start the app and point out the DEMO DATA label, or upload the real Telco CSV.
2. Open Analysis: explain cohort size, churn share, contract associations and PCA map. Emphasize association is not causation.
3. Train Logistic Regression and show held-out metrics and coefficient chart.
4. Attempt the quantum simulator run; show the circuit and measured results only if it completes.
5. Open Prediction, score an example, adjust the high-risk threshold, then show batch CSV scoring if prepared.
6. Download the experiment JSON report and close with limitations: one split, small data representation, simulator rather than hardware, no quantum-advantage claim.

See [CODE_EXPLANATION_GUIDE.md](CODE_EXPLANATION_GUIDE.md) and [VIVA_PREPARATION.md](VIVA_PREPARATION.md).

## Deploy online with minimal changes

This project uses Streamlit. The simplest deployment path is Streamlit Community Cloud:

1. Create a GitHub repository and upload the contents of this `q-insight` folder to the repository root. Include `app.py`, `requirements.txt`, and the `src` folder. Do not upload `.venv` or customer CSV files; `.gitignore` excludes them.
2. Sign in at [share.streamlit.io](https://share.streamlit.io/) with GitHub and choose **Create app**.
3. Select the repository, its `main` branch, and `app.py`, then choose **Deploy**.

The cloud service installs packages from `requirements.txt` in its own Linux environment. A Windows Application Control policy that blocks a local scikit-learn `.pyd` extension is specific to the local environment; it does not by itself mean the cloud deployment will fail. If cloud startup fails, use the final error lines in the app logs to diagnose that separate deployment environment.
