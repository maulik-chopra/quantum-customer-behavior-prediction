"""Q-INSIGHT: an explainable customer churn research dashboard."""
import json
import time
from io import StringIO
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.data_processing import detect_target, make_demo_data, prepare_data
from src.classical_model import train_classical
from src.quantum_model import train_quantum
from src.evaluation import evaluate_model

st.set_page_config(page_title="Q-INSIGHT | Customer Intelligence", page_icon="◈", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');
.stApp{background:radial-gradient(ellipse at 10% 0%,#11233c 0%,#080d18 45%,#060912 100%);color:#e7f0ff;font-family:'DM Sans',sans-serif}
h1,h2,h3{font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.03em}
.hero{padding:26px 30px;border:1px solid #223854;border-radius:20px;background:linear-gradient(120deg,rgba(22,45,72,.9),rgba(18,23,43,.78));box-shadow:0 20px 60px #02061188;margin-bottom:20px}
.eyebrow{color:#66d9f4;font-size:12px;letter-spacing:.18em;text-transform:uppercase;font-weight:700}
.metric{padding:18px 20px;border:1px solid #263850;border-radius:16px;background:linear-gradient(145deg,#121d2d,#0c1422);min-height:94px}
.metric small{color:#92a8c5;text-transform:uppercase;letter-spacing:.12em;font-size:10px}.metric strong{display:block;color:#eaf6ff;font:600 26px 'Space Grotesk';margin-top:7px}
.stButton>button{border-radius:10px;background:linear-gradient(100deg,#0e95b9,#5e63d9);border:0;color:white;font-weight:700}
[data-testid="stTabs"] button{color:#9fb5d2}.stCaption{color:#9fb2cd}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="hero"><div class="eyebrow">◈ Q-INSIGHT · RESEARCH COMMAND CENTER</div><h1>Quantum Customer Intelligence</h1><div>Behavior analytics · Churn risk · Classical vs quantum experiment</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### DATA SOURCE")
    uploaded = st.file_uploader("Upload a customer CSV", type="csv")
    use_demo = st.toggle("Use clearly labeled DEMO DATA", value=uploaded is None)
    st.caption("Recommended columns follow the Kaggle Telco Customer Churn dataset.")

if uploaded is not None:
    try:
        data = pd.read_csv(uploaded)
        demo_mode = False
    except Exception as error:
        st.error(f"Could not read CSV: {error}"); st.stop()
elif use_demo:
    data = make_demo_data(); demo_mode = True
else:
    st.info("Upload a CSV or enable DEMO DATA to begin."); st.stop()

if demo_mode:
    st.info("DEMO DATA — synthetic Telco-like records are used. Results are illustrative and are not Kaggle findings.")
target_guess = detect_target(data)
target = st.sidebar.selectbox("Churn label column", list(data.columns), index=list(data.columns).index(target_guess) if target_guess in data.columns else len(data.columns)-1)
try:
    prepared = prepare_data(data, target)
    X_train, X_test, y_train, y_test, preprocessor, feature_names, raw_train, raw_test = prepared
except Exception as error:
    st.error(f"Dataset validation failed: {error}"); st.stop()

missing = int(data.isna().sum().sum())
st.markdown('<div class="eyebrow">LIVE CUSTOMER COHORT</div>', unsafe_allow_html=True)
kpi = st.columns(4)
for col, label, value in zip(kpi, ["CUSTOMERS", "CHURN RATE", "FEATURES", "MISSING CELLS"], [f"{len(data):,}", f"{y_train.mean():.1%}", str(X_train.shape[1]), str(missing)]):
    col.markdown(f'<div class="metric"><small>{label}</small><strong>{value}</strong></div>', unsafe_allow_html=True)

tabs = st.tabs(["01  Analysis", "02  Prediction", "03  Quantum vs Classical"])
with tabs[0]:
    st.subheader("Behavior signals")
    c1, c2 = st.columns([1,1])
    if "Contract" in data:
        chart = data.groupby(["Contract", target], dropna=False).size().reset_index(name="Customers")
        c1.plotly_chart(px.bar(chart, x="Contract", y="Customers", color=target, barmode="group", title="Customer outcomes by contract", color_discrete_sequence=["#55d8f2", "#a187ff"]), use_container_width=True)
    else:
        c1.plotly_chart(px.histogram(data, x=target, color=target, title="Churn vs retention", color_discrete_sequence=["#55d8f2", "#a187ff"]), use_container_width=True)
    categorical_cols = [column for column in data.select_dtypes(exclude="number").columns if column != target]
    if categorical_cols:
        category = st.selectbox("Compare churn across a service or payment category", categorical_cols)
        grouped = data.groupby([category, target], dropna=False).size().reset_index(name="Customers")
        st.plotly_chart(px.bar(grouped, x=category, y="Customers", color=target, barmode="group", title=f"Customer outcomes by {category}", color_discrete_sequence=["#55d8f2", "#a187ff"]), use_container_width=True)
    numeric_cols = data.select_dtypes(include="number").columns.tolist()
    if numeric_cols:
        selected = st.selectbox("Explore numeric behavior", numeric_cols)
        c2.plotly_chart(px.histogram(data, x=selected, color=target, marginal="box", title=f"{selected} by churn label", color_discrete_sequence=["#5bdcf3", "#a187ff"]), use_container_width=True)
    else:
        c2.metric("Target balance", f"{data[target].value_counts(normalize=True).max():.1%} largest class")
    st.caption("These charts describe associations in the loaded cohort. They do not establish that a feature causes churn.")
    # PCA embedding is calculated on transformed rows for visual exploration only.
    from sklearn.decomposition import PCA
    embedding = PCA(n_components=2, random_state=17).fit_transform(preprocessor.transform(data.drop(columns=[target]).drop(columns=[c for c in data if str(c).lower() in {"customerid","id"}], errors="ignore")))
    plot = pd.DataFrame({"PC1": embedding[:,0], "PC2": embedding[:,1], "Label": data[target].astype(str)})
    st.plotly_chart(px.scatter(plot, x="PC1", y="PC2", color="Label", opacity=.7, title="Customer behavior map · PCA projection", color_discrete_sequence=["#55d8f2", "#a187ff"]), use_container_width=True)
    st.caption("PCA map is an exploratory projection. The supervised model's preprocessing remains fitted on training rows only.")
    if "Contract" in data:
        rates = data.assign(_churn=data[target].map({"Yes":1,"No":0,"1":1,"0":0})).groupby("Contract", dropna=False)["_churn"].mean().dropna()
        if len(rates): st.info(f"Observed churn share ranges from {rates.min():.0%} to {rates.max():.0%} across contracts. This is a data-supported association, not a causal effect.")

with tabs[1]:
    st.subheader("Single customer risk")
    classical = st.session_state.get("classical_model")
    if classical is None:
        st.warning("Train the baseline in Quantum vs Classical first to enable model predictions.")
    else:
        row = {}
        inputs = st.columns(3)
        for index, column in enumerate(raw_train.columns):
            values = raw_train[column].dropna()
            with inputs[index % 3]:
                if pd.api.types.is_numeric_dtype(values):
                    row[column] = st.number_input(column, value=float(values.median()), key=f"form_{column}")
                else:
                    options = values.astype(str).value_counts().index.tolist()
                    row[column] = st.selectbox(column, options, key=f"form_{column}")
        if st.button("Predict Customer Churn", type="primary"):
            try:
                sample = pd.DataFrame([row], columns=raw_train.columns)
                vector = preprocessor.transform(sample)
                probability = float(classical.predict_proba(vector)[0,1])
                threshold = st.session_state.get("risk_threshold", .67)
                level = "HIGH" if probability >= threshold else ("MEDIUM" if probability >= .35 else "LOW")
                st.session_state["last_prediction"] = (probability, level, sample)
            except Exception as error: st.error(f"Prediction failed: {error}")
        threshold = st.slider("High risk threshold", .5, .9, .67, .01)
        st.session_state["risk_threshold"] = threshold
        if "last_prediction" in st.session_state:
            probability, level, sample = st.session_state["last_prediction"]
            st.metric("Classical churn probability", f"{probability:.1%}", f"{level} RISK")
            st.progress(min(probability, 1.0), text="Estimated churn probability")
            quantum = st.session_state.get("quantum_result")
            if quantum:
                qprob = float(quantum["model"].predict_proba(quantum["scale"].transform(quantum["pca"].transform(preprocessor.transform(sample))))[0,1])
                st.write(f"Quantum simulator estimate: **{qprob:.1%}**")
                if (qprob >= .5) != (probability >= .5): st.warning("Models disagree on the 0.5 classification threshold; review this customer rather than treating either estimate as certain.")
            st.info("Suggested action: review account context and offer a relevant service or contract conversation. The score is a screening signal, not a decision.")
    st.markdown("#### Batch CSV scoring")
    batch_file = st.file_uploader("Upload feature rows (same columns as training data)", type="csv", key="batch")
    if batch_file and classical:
        try:
            batch = pd.read_csv(batch_file)
            batch_x = preprocessor.transform(batch[raw_train.columns])
            scores = classical.predict_proba(batch_x)[:,1]
            scored = batch.copy(); scored["churn_probability"] = scores
            scored["risk_category"] = np.where(scores >= st.session_state.get("risk_threshold", .67), "High", np.where(scores >= .35, "Medium", "Low"))
            st.dataframe(scored.head(20), use_container_width=True)
            st.download_button("Download scored CSV", scored.to_csv(index=False), "q_insight_predictions.csv", "text/csv")
        except Exception as error: st.error(f"Batch validation failed: {error}")

with tabs[2]:
    st.subheader("Reproducible model experiment")
    st.caption("Both models use the same stratified held-out customers. Quantum PCA and feature scaling are fitted on training rows only.")
    train_col, q_col = st.columns(2)
    if train_col.button("Train Classical Baseline", type="primary"):
        with st.spinner("Fitting Logistic Regression…"):
            model, elapsed = train_classical(X_train, y_train)
            measured = evaluate_model(model, X_test, y_test)
            st.session_state["classical_model"] = model
            st.session_state["classical_eval"] = measured
            st.session_state["classical_seconds"] = elapsed
    if q_col.button("Train Quantum Simulator"):
        try:
            with st.spinner("Running four-qubit variational classifier on Aer simulator…"):
                result = train_quantum(X_train, y_train, X_test)
                result["evaluation"] = evaluate_model(result["model"], result["test_x"], y_test)
                st.session_state["quantum_result"] = result
        except Exception as error:
            st.session_state["quantum_error"] = str(error)
    if "quantum_error" in st.session_state: st.error(f"Quantum model unavailable: {st.session_state['quantum_error']}")
    classical_eval = st.session_state.get("classical_eval")
    quantum_result = st.session_state.get("quantum_result")
    results = []
    if classical_eval:
        for metric in ["accuracy","precision","recall","f1","roc_auc"]: results.append({"Model":"Logistic Regression","Metric":metric,"Score":classical_eval[metric]})
    if quantum_result:
        for metric in ["accuracy","precision","recall","f1","roc_auc"]: results.append({"Model":"Quantum VQC","Metric":metric,"Score":quantum_result["evaluation"][metric]})
    if results:
        metrics = pd.DataFrame(results)
        st.plotly_chart(px.bar(metrics, x="Metric", y="Score", color="Model", barmode="group", range_y=[0,1], title="Measured held-out metrics", color_discrete_map={"Logistic Regression":"#55d8f2","Quantum VQC":"#a187ff"}), use_container_width=True)
        # Draw ROC curves from measured probabilities, including the random-ranking reference.
        roc_figure = go.Figure()
        if classical_eval:
            fpr, tpr, _ = classical_eval["roc"]
            roc_figure.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Logistic Regression", line={"color":"#55d8f2"}))
        if quantum_result:
            fpr, tpr, _ = quantum_result["evaluation"]["roc"]
            roc_figure.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="Quantum VQC", line={"color":"#a187ff"}))
        roc_figure.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Chance", line={"color":"#718096","dash":"dash"}))
        roc_figure.update_layout(title="ROC curves · held-out customers", xaxis_title="False positive rate", yaxis_title="True positive rate", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="#dce9fa")
        st.plotly_chart(roc_figure, use_container_width=True)
        timing = []
        if classical_eval:
            timing.append({"Model":"Logistic Regression","Training seconds":st.session_state.get("classical_seconds", 0),"Inference seconds":classical_eval["inference_seconds"]})
        if quantum_result:
            timing.append({"Model":"Quantum VQC","Training seconds":quantum_result["training_seconds"],"Inference seconds":quantum_result["evaluation"]["inference_seconds"]})
        st.markdown("#### Measured runtime")
        st.dataframe(pd.DataFrame(timing), use_container_width=True, hide_index=True)
        for name, result in [("Logistic Regression", classical_eval), ("Quantum VQC", quantum_result["evaluation"] if quantum_result else None)]:
            if result:
                st.write(f"**{name} · confusion matrix [true rows, predicted columns]**")
                st.dataframe(pd.DataFrame(result["confusion"], index=["Actual retained","Actual churn"], columns=["Pred retained","Pred churn"]))
        if classical_eval and quantum_result:
            st.info("Compare recall alongside accuracy: false negatives are customers who churned but were missed. Quantum scores are a small simulator experiment, not evidence of quantum advantage.")
        if quantum_result:
            st.markdown("#### Four-qubit circuit setup")
            st.write(f"Aer simulator · training rows {quantum_result['train_rows']} · training time {quantum_result['training_seconds']:.2f}s · PCA variance retained {quantum_result['explained_variance']:.1%}")
            st.code(str(quantum_result["feature_map"]) + "\n+\n" + str(quantum_result["ansatz"]), language="text")
        report = {"title":"Q-INSIGHT experiment report","synthetic_demo_data":demo_mode,"dataset_rows":len(data),"train_rows":len(y_train),"test_rows":len(y_test),"seed":17,"configuration":{"classical":"Balanced Logistic Regression","quantum":"4-qubit ZZFeatureMap + RealAmplitudes VQC","quantum_training_row_limit":100,"quantum_optimizer_iterations":25,"decision_threshold":0.5},"classical":({k:v for k,v in classical_eval.items() if k not in {"roc","confusion","predictions","probabilities"}} | {"training_seconds":st.session_state.get("classical_seconds")} if classical_eval else None),"quantum":(({k:v for k,v in quantum_result["evaluation"].items() if k not in {"roc","confusion","predictions","probabilities"}} | {"training_seconds":quantum_result["training_seconds"],"pca_explained_variance":quantum_result["explained_variance"],"training_rows":quantum_result["train_rows"]}) if quantum_result else None),"quantum_error":st.session_state.get("quantum_error"),"limitations":["Simulation only; no physical quantum hardware.","Small PCA representation and bounded training sample.","Metrics describe this dataset split, not guaranteed future performance."]}
        # Convert numpy values for a portable JSON experiment report.
        report_json = json.dumps(report, default=lambda value: np.asarray(value).tolist(), indent=2)
        st.download_button("Download experiment report (JSON)", report_json, "q_insight_experiment_report.json", "application/json")
    else:
        st.info("No measured model results yet. Train either model above to populate comparisons.")
    if "classical_model" in st.session_state:
        coefficients = st.session_state["classical_model"].coef_[0]
        top = pd.DataFrame({"Feature":feature_names,"Coefficient":coefficients}).assign(Magnitude=lambda x:x.Coefficient.abs()).nlargest(12,"Magnitude")
        st.plotly_chart(px.bar(top.sort_values("Coefficient"), x="Coefficient", y="Feature", orientation="h", title="Largest Logistic Regression coefficients (association, not causation)", color="Coefficient", color_continuous_scale="RdBu"), use_container_width=True)
