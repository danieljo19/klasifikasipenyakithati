import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(page_title="Liver Disease Prediction", layout="wide")

# SIMPAN TAB AKTIF
if "active_tab" not in st.session_state:
    st.session_state["active_tab"] = "📂 Upload CSV"

# =====================================================
# DARK UI STYLE
# =====================================================
st.markdown("""
<style>
.stApp {background-color: #0f172a; color: #f8fafc;}
h1, h2, h3, h4 {color: #38bdf8;}
.stButton>button {background-color: #2563eb;color: white;border-radius: 10px;font-weight: bold;}
.stFileUploader {background-color: #020617;border-radius: 14px;padding: 25px;border: 1px solid #334155;}
div[data-testid="stMetric"] {background-color: #020617;padding: 20px;border-radius: 12px;border: 1px solid #334155;}
.stDataFrame {background-color: #020617;}
.stSidebar {background-color: #020617;}
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL
# =====================================================
@st.cache_resource
def load_model(kernel):
    base = "Model"
    model = joblib.load(f"{base}/svm_{kernel}_model.pkl")
    scaler = joblib.load(f"{base}/scaler.pkl")
    encoder = joblib.load(f"{base}/encoder.pkl")
    pca = joblib.load(f"{base}/pca.pkl")
    info = joblib.load(f"{base}/svm_{kernel}_info.pkl")
    return model, scaler, encoder, pca, info


# =====================================================
# PREPROCESS
# =====================================================
def preprocess(df, encoder, scaler, pca):
    df.columns = df.columns.str.strip()

    rename_map = {
        'Age of the patient':'Age', 'Gender of the patient':'Gender',
        'Alkphos Alkaline Phosphotase':'Alkaline Phosphotase',
        'Sgpt Alamine Aminotransferase':'SGPT',
        'Sgot Aspartate Aminotransferase':'SGOT',
        'Total Protiens':'Total Proteins',
        'ALB Albumin':'Albumin',
        'A/G Ratio Albumin and Globulin Ratio':'A/G Ratio'
    }
    df.rename(columns=rename_map, inplace=True)

    required = ["Age","Gender","Total Bilirubin","Direct Bilirubin","Alkaline Phosphotase",
                "SGPT","SGOT","Total Proteins","Albumin","A/G Ratio"]

    before = len(df)
    df = df.dropna(subset=required)
    after = len(df)
    removed = before - after

    missing = set(required) - set(df.columns)
    if missing:
        st.error(f"❌ Missing columns: {missing}")
        return None, None, None

    X_num = df[[c for c in required if c != "Gender"]]
    X_cat = encoder.transform(df[["Gender"]])

    X = pd.concat([X_num.reset_index(drop=True),
                   pd.DataFrame(X_cat, columns=encoder.get_feature_names_out())], axis=1)

    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    return X_pca, df, (before, after, removed)


# =====================================================
# PREDICT
# =====================================================
def predict_single(data, model, scaler, encoder, pca):
    df = pd.DataFrame([data])
    X, _, _ = preprocess(df, encoder, scaler, pca)
    if X is None: return None, None
    return model.predict(X)[0], model.predict_proba(X)[0]


def predict_batch(df, model, scaler, encoder, pca):
    X, clean, meta = preprocess(df, encoder, scaler, pca)
    if X is None: return None, None, None, None
    return model.predict(X), model.predict_proba(X), clean, meta


# =====================================================
# GAUGE
# =====================================================
def gauge(prob, pred):
    title = "Liver Disease" if pred==1 else "Non-Liver Disease"
    value = prob[1]*100 if pred==1 else prob[0]*100
    return go.Figure(go.Indicator(mode="gauge+number",
        value=value, title={"text":title},
        gauge={"bar":{"color":"red" if pred==1 else "green"},"axis":{"range":[0,100]}}))


# =====================================================
# SIDEBAR MODEL CONFIG
# =====================================================
st.sidebar.title("⚙ Configuration Model")
kernel = st.sidebar.selectbox("Select Kernel",["rbf","linear","poly","sigmoid"])

if st.sidebar.button("Load Model"):
    model, scaler, encoder, pca, info = load_model(kernel)
    st.session_state.update({"model":model,"scaler":scaler,
                             "encoder":encoder,"pca":pca,"info":info})

if "info" in st.session_state:
    st.sidebar.metric("F1-Macro", f"{st.session_state['info']['valid_f1_macro']:.4f}")
    st.sidebar.metric("ROC-AUC", f"{st.session_state['info']['valid_roc_auc']:.4f}")


# =====================================================
# TABS
# =====================================================
tabs = ["✍ Manual Input","📂 Upload CSV","📊 PCA Analysis"]
tab1, tab2, tab3 = st.tabs(tabs)


# ======================================================================
# TAB 1 – Manual Input
# ======================================================================
with tab1:
    if "model" not in st.session_state:
        st.warning("Load model terlebih dahulu")
    else:
        Age = st.number_input("Age",1,120,45)
        Gender = st.selectbox("Gender",["Male","Female"])
        TotalB = st.number_input("Total Bilirubin",0.0,50.0,1.0)
        DirectB = st.number_input("Direct Bilirubin",0.0,50.0,0.3)
        Alk = st.number_input("Alkaline Phosphotase",0.0,1000.0,200.0)
        SGPT = st.number_input("SGPT",0.0,1000.0,30.0)
        SGOT = st.number_input("SGOT",0.0,1000.0,35.0)
        TP = st.number_input("Total Proteins",0.0,20.0,6.5)
        Alb = st.number_input("Albumin",0.0,10.0,3.2)
        AGR = st.number_input("A/G Ratio",0.0,5.0,1.0)

        if st.button("Predict Manual Input"):
            st.session_state["active_tab"] = "✍ Manual Input"

            pred,prob = predict_single(
                {"Age":Age,"Gender":Gender,"Total Bilirubin":TotalB,"Direct Bilirubin":DirectB,
                 "Alkaline Phosphotase":Alk,"SGPT":SGPT,"SGOT":SGOT,
                 "Total Proteins":TP,"Albumin":Alb,"A/G Ratio":AGR},
                st.session_state["model"],st.session_state["scaler"],
                st.session_state["encoder"],st.session_state["pca"])

            if pred is not None:
                st.success(f"Prediction Result: **{'Liver Disease' if pred==1 else 'Non-Liver Disease'}**")
                st.plotly_chart(gauge(prob,pred), use_container_width=True)


# ======================================================================
# TAB 2 – Upload CSV
# ======================================================================
with tab2:
    st.session_state["active_tab"] = "📂 Upload CSV"

    if "model" not in st.session_state:
        st.warning("Load model terlebih dahulu")
    else:
        option = st.radio("Select Test File",["Upload Manual","Use Testing Dataset"])

        file = None
        if option == "Upload Manual":
            file = st.file_uploader("Upload CSV", type=["csv","xlsx"], key="uploader")
        else:
            file = "test.csv.xlsx"
            st.success(f"Using Dataset Testing")

        if file:
            st.session_state["active_tab"] = "📂 Upload CSV"

            try:
                if str(file).endswith(".csv"):
                    df = pd.read_csv(file, encoding="latin1")
                else:
                    df = pd.read_excel(file)
            except:
                st.error("File error format")
                st.stop()

            st.subheader("Preview Data")
            st.dataframe(df.head())

            pred,prob,clean,meta = predict_batch(df,st.session_state["model"],
                                            st.session_state["scaler"],
                                            st.session_state["encoder"],
                                            st.session_state["pca"])

            if clean is not None:
                before,after,removed = meta
                st.info(f"📌 Total Data: **{before}**, Setelah Clean: **{after}**, Terhapus: **{removed}**")

                clean["Prediction"] = pred
                clean["Prob_Liver"] = prob[:,1]

                st.dataframe(clean)
                st.download_button("⬇ Download Result", clean.to_csv(index=False), "prediction.csv")

                # PIE CHART
                fig = px.pie(values=[(pred==0).sum(), (pred==1).sum()],
                             names=["Non-Liver","Liver"], hole=0.3)
                fig.update_traces(pull=[0,0.1])
                st.plotly_chart(fig, use_container_width=True)


# ======================================================================
# TAB 3 – PCA ANALYSIS
# ======================================================================
with tab3:
    if "model" in st.session_state:
        st.subheader("📊 PCA Cumulative Explained Variance")
        cum_var = np.cumsum(st.session_state["pca"].explained_variance_ratio_)
        st.dataframe(pd.DataFrame({"Component": np.arange(1,len(cum_var)+1),
                                   "Cumulative Variance": cum_var}))

        st.subheader("📄 PCA Feature Influence Ranking")
        feat_names = ["Age","Total Bilirubin","Direct Bilirubin","Alkaline Phosphotase",
                      "SGPT","SGOT","Total Proteins","Albumin","A/G Ratio"] + \
                     list(st.session_state["encoder"].get_feature_names_out())

        abs_loading = np.abs(st.session_state["pca"].components_[0])
        ranking = pd.DataFrame({"Feature": feat_names, "Influence Score": abs_loading})
        ranking = ranking.sort_values(by="Influence Score", ascending=False)

        st.dataframe(ranking.style.background_gradient())
    else:
        st.warning("Load model terlebih dahulu")
