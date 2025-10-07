# app.py — LPD Tester (SVM + PCA) — Opsi-1 (kartu hasil pastel) + Imputasi anti-NaN
# ==================================================================================
# Fitur:
# - Uji via File (CSV/XLSX) & Form (Realtime)
# - Kompatibel dengan pipeline:
#    (a) punya 'preprocess' (ColumnTransformer) → bisa terima raw (Age, Gender, ...)
#    (b) tanpa 'preprocess' → app bangun X_enc otomatis (impute + OHE Gender)
# - Imputasi otomatis agar PCA/SVM tidak error NaN
# - Model Inspector: explained variance, scree plot, heatmap loadings
# - UI: kartu hasil (Opsi-1), gauge semicircle, metrics, confusion matrix, download hasil
# ==================================================================================

import os, io, base64, warnings, sys, textwrap
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)

# -------------------- Page Setup --------------------
st.set_page_config(
    page_title="LPD • SVM+PCA — Pengujian",
    page_icon="🩺",
    layout="wide",
)

# -------------------- Pastel UI CSS --------------------
st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}

/* Badge status (pastel) */
.badge {
  display:inline-flex; align-items:center; gap:.45rem;
  padding:.32rem .70rem; border-radius:999px; font-weight:700;
  border:1.5px solid transparent; letter-spacing:.1px;
}
.badge-low  { background:#dcfce7; color:#14532d; border-color:#86efac; }
.badge-mod  { background:#fef9c3; color:#713f12; border-color:#fde68a; }
.badge-high { background:#fee2e2; color:#7f1d1d; border-color:#fecaca; }
.badge-info { background:#e0f2fe; color:#0c4a6e; border-color:#bae6fd; }

/* Kartu hasil (border pastel + inner glow) */
.card {
  background: rgba(147, 197, 253, .06);
  border: 2px solid #93c5fd;
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 0 0 3px rgba(147, 197, 253, .15) inset;
  color: #e5e7eb;
  margin:.25rem 0 1rem 0;
}
.section {
  background: rgba(187, 247, 208, .06);
  border: 2px solid #bbf7d0;
  border-radius: 12px;
  padding: 12px 14px;
  box-shadow: 0 0 0 3px rgba(187, 247, 208, .12) inset;
  color: #e5e7eb;
}

/* Teks hasil */
.result-title { font-size: 1.6rem; line-height: 1.25; font-weight: 800; color: #e5e7eb; margin: .1rem 0 .6rem 0; }
.result-sub   { color:#cbd5e1; font-size: .95rem; }
.icon-pill {
  display:inline-flex; align-items:center; justify-content:center;
  width:26px; height:26px; border-radius:6px; margin-right:.35rem;
  background:#22c55e20; border:1.5px solid #86efac; color:#86efac;
}

/* Table header center */
[data-testid="stTable"] th {text-align:center;}
</style>
""", unsafe_allow_html=True)

# -------------------- Schema (sesuai X_enc training) --------------------
TARGET = "Result"
NUMERIC_SHORT = [
    'Age', 'Total Bilirubin', 'Direct Bilirubin',
    'Alkaline Phosphotase', 'SGPT', 'SGOT',
    'Total Proteins', 'Albumin', 'A/G Ratio'
]
TRAIN_SCHEMA = NUMERIC_SHORT + ['Gender_Female', 'Gender_Male']

# Normalisasi berbagai variasi kolom → schema singkat
NORMALIZE_TO_SHORT = {
    'Age of the patient': 'Age',
    'Gender of the patient': 'Gender',
    'Total Bilirubin': 'Total Bilirubin',
    'Direct Bilirubin': 'Direct Bilirubin',
    'Alkaline Phosphatase': 'Alkaline Phosphotase',
    'Alkaline Phosphatase ': 'Alkaline Phosphotase',
    'Alkphos Alkaline Phosphatase': 'Alkaline Phosphotase',
    'Alkphos Alkaline Phosphatase\xa0': 'Alkaline Phosphotase',
    'SGPT': 'SGPT', 'Sgpt Alamine Aminotransferase': 'SGPT',
    'SGOT': 'SGOT', 'Sgot Aspartate Aminotransferase': 'SGOT',
    'Total Proteins': 'Total Proteins', 'Total Prot': 'Total Proteins',
    'Albumin': 'Albumin', 'Albi Albumin': 'Albumin',
    'A/G Ratio': 'A/G Ratio',
    'A/G Ratio Albumin and Globulin Ratio': 'A/G Ratio',
    'Result': 'Result'
}

# -------------------- Utils --------------------
def normalize_to_short_cols(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: NORMALIZE_TO_SHORT.get(c, c))

def load_any(uploaded_or_path):
    if uploaded_or_path is None:
        raise FileNotFoundError("File tidak ditemukan.")
    if isinstance(uploaded_or_path, str):
        if not os.path.exists(uploaded_or_path):
            raise FileNotFoundError(f"Path tidak valid: {uploaded_or_path}")
        if uploaded_or_path.lower().endswith(".csv"):
            return pd.read_csv(uploaded_or_path, encoding_errors="ignore")
        elif uploaded_or_path.lower().endswith(".xlsx"):
            return pd.read_excel(uploaded_or_path)
        else:
            raise ValueError("Gunakan format .csv atau .xlsx")
    else:
        name = uploaded_or_path.name.lower()
        if name.endswith(".csv"):
            return pd.read_csv(uploaded_or_path, encoding_errors="ignore")
        elif name.endswith(".xlsx"):
            return pd.read_excel(uploaded_or_path)
        else:
            raise ValueError("Gunakan format .csv atau .xlsx")

def pipeline_has_preprocess(pipeline) -> bool:
    try:
        return ('preprocess' in pipeline.named_steps)
    except Exception:
        return False

def impute_numeric_median(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Imputasi median utk kolom numerik; jika semua NaN → isi 0 sebagai fallback."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        med = s.median()
        if np.isnan(med):
            med = 0.0
        out[c] = s.fillna(med)
    return out

def impute_gender(df: pd.DataFrame) -> pd.DataFrame:
    """Isi Gender kosong → 'Male' dan normalisasi label."""
    out = df.copy()
    if 'Gender' in out.columns:
        g = out['Gender'].astype(str).str.strip().str.lower()
        g = g.map({'male':'Male','m':'Male','laki-laki':'Male','laki laki':'Male',
                   'female':'Female','f':'Female','perempuan':'Female'}).fillna('Male')
        out['Gender'] = g
    return out

def build_X_enc_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Bangun X_enc (impute median + OHE Gender) dan urutkan sesuai TRAIN_SCHEMA."""
    df = normalize_to_short_cols(df_raw.copy())
    df = impute_gender(df)
    for c in NUMERIC_SHORT:
        if c not in df.columns: df[c] = np.nan
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = impute_numeric_median(df, NUMERIC_SHORT)
    df['Age'] = np.floor(df['Age']).astype(int)

    g = df['Gender'] if 'Gender' in df.columns else pd.Series(['Male']*len(df))
    df['Gender_Female'] = (g == 'Female').astype(int)
    df['Gender_Male']   = (g == 'Male').astype(int)

    for c in TRAIN_SCHEMA:
        if c not in df.columns: df[c] = 0
    X_enc = df[TRAIN_SCHEMA].copy()
    X_enc = impute_numeric_median(X_enc, TRAIN_SCHEMA)
    return X_enc

def risk_level(prob):
    if prob < 0.33: return "Low", "#16a34a"
    if prob < 0.66: return "Moderate", "#f59e0b"
    return "High", "#dc2626"

def near_boundary(prob, window=0.05):
    return abs(prob - 0.5) <= window

# --- Gauge visual ---
def render_gauge_html(prob: float):
    pct = max(0.0, min(prob*100.0, 100.0))
    angle = -90 + (pct * 1.8)  # 0..100% -> -90..+90
    if prob < 0.33:
        label, emoji = "Low", "🟢"
    elif prob < 0.66:
        label, emoji = "Moderate", "🟡"
    else:
        label, emoji = "High", "🔴"

    html = f"""
<div style="width:100%;display:flex;justify-content:center;margin-top:.5rem;">
  <div style="position:relative;width:620px;height:340px;font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto;">
    <div style="
      position:absolute;left:0;bottom:0;width:100%;height:100%;
      border-top-left-radius:620px;border-top-right-radius:620px;
      background:
        conic-gradient(from 180deg,
          #DCFCE7 0deg 110deg,
          #FEF3C7 110deg 230deg,
          #FEE2E2 230deg 360deg
        );
      -webkit-mask: radial-gradient(circle at 50% 100%, transparent 225px, #000 226px);
              mask: radial-gradient(circle at 50% 100%, transparent 225px, #000 226px);
      box-shadow: inset 0 12px 24px rgba(0,0,0,.18);
    "></div>

    <div style="position:absolute;left:0;bottom:0;width:100%;height:100%;
                -webkit-mask: radial-gradient(circle at 50% 100%, transparent 214px, #000 215px);
                        mask: radial-gradient(circle at 50% 100%, transparent 214px, #000 215px);">
      {"".join([
        f'<div style="position:absolute;left:50%;bottom:0;transform-origin:50% 100%;' +
        f'transform:translateX(-50%) rotate({-90 + i*18}deg);width:2px;height:34px;' +
        'background:rgba(255,255,255,.65);border-radius:2px;opacity:.8;"></div>'
        for i in range(0,11)
      ])}
    </div>

    <div style="
      position:absolute;left:50%;bottom:0;transform-origin:50% 100%;
      transform: translateX(-50%) rotate(""" + f"{angle}" + """deg);
      width:5px;height:210px;background:linear-gradient(#e5e7eb,#9ca3af);
      border-radius:3px; z-index:3; box-shadow:0 0 6px rgba(0,0,0,.35);
      transition: transform .9s cubic-bezier(.2,.9,.2,1.05);
    "></div>
    <div style="
      position:absolute;left:50%;bottom:0;transform:translate(-50%, 8px);
      width:18px;height:18px;background:#e5e7eb;border-radius:50%;z-index:4;
      box-shadow:0 0 8px rgba(0,0,0,.45);
    "></div>

    <div style="position:absolute;left:6%;bottom:48px;color:#9CA3AF;font-size:12px;">0%</div>
    <div style="position:absolute;left:50%;transform:translateX(-50%);bottom:8px;color:#9CA3AF;font-size:12px;">50%</div>
    <div style="position:absolute;right:6%;bottom:48px;color:#9CA3AF;font-size:12px;">100%</div>

    <div style="position:absolute;left:12%;bottom:115px;color:#16a34a;font-weight:700;">Low</div>
    <div style="position:absolute;left:50%;transform:translateX(-50%);bottom:160px;color:#f59e0b;font-weight:700;">Moderate</div>
    <div style="position:absolute;right:12%;bottom:115px;color:#dc2626;font-weight:700;">High</div>

    <div style="
      position:absolute;left:0;right:0;bottom:84px;text-align:center;
      color:#e5e7eb;font-size:48px;font-weight:800;letter-spacing:.5px;">
      """ + f"{pct:.1f}" + """% <span style="font-size:34px;vertical-align:2px;">""" + f"{emoji}" + """</span>
    </div>
    <div style="
      position:absolute;left:0;right:0;bottom:52px;text-align:center;
      color:#cbd5e1;font-size:14px;letter-spacing:.4px;text-transform:uppercase;">
      Risk: """ + f"{label}" + """
    </div>
  </div>
</div>
"""
    components.html(html, height=360)

# --- Kartu hasil Opsi-1 (pastel) ---
def render_output_option1(pred: int, ppos: float):
    is_positive = (pred == 1)
    prob_pct = f"{ppos*100:.1f}%"

    # badge risiko
    if ppos < 0.33:
        risk_html = '<span class="badge badge-low">Low risk</span>'
    elif ppos < 0.66:
        risk_html = '<span class="badge badge-mod">Moderate risk</span>'
    else:
        risk_html = '<span class="badge badge-high">High risk</span>'

    # label positif/negatif
    label_html = (
        '<span class="icon-pill">✅</span> Positive (terindikasi penyakit liver)'
        if is_positive else
        '<span class="icon-pill" style="background:#60a5fa20;border-color:#93c5fd;color:#93c5fd;">✖</span> '
        'Negative (tidak terindikasi penyakit liver)'
    )

    html = f"""
<div class="card">
  <div class="result-title">Hasil Prediksi: {label_html}</div>
  <div class="result-sub" style="display:flex;gap:.6rem;align-items:center;flex-wrap:wrap;">
    {risk_html}
    <span>Probabilitas positif: <b style="color:#e5e7eb">{prob_pct}</b></span>
    <span style="opacity:.8">• Ambang keputusan = 50%</span>
  </div>
</div>
"""
    st.markdown(html, unsafe_allow_html=True)

# --- Prediksi aman (imputasi kalau ketemu NaN) ---
def safe_predict(pipeline, X, has_preprocess: bool):
    def _try_infer(_X):
        try:
            yp = pipeline.predict(_X)
            pp = None
            try:
                pp = pipeline.predict_proba(_X)[:,1]
            except Exception:
                try:
                    dv = pipeline.decision_function(_X)
                    pp = 1/(1+np.exp(-dv))
                except Exception:
                    pp = None
            return yp, pp, None
        except Exception as e:
            return None, None, e

    y_pred, p_pos, err = _try_infer(X)
    if err is None:
        return y_pred, p_pos

    # jika error terkait NaN → imputasi, lalu coba lagi
    msg = str(err).lower()
    if "nan" in msg or "contains nan" in msg:
        X2 = X.copy()
        if has_preprocess:
            num_cols = [c for c in X2.columns if pd.api.types.is_numeric_dtype(X2[c])]
            if num_cols:
                X2 = impute_numeric_median(X2, num_cols)
        else:
            X2 = impute_numeric_median(X2, X2.columns.tolist())
        y_pred, p_pos, err2 = _try_infer(X2)
        if err2 is None:
            return y_pred, p_pos
        X3 = X2.dropna()
        y_pred, p_pos, err3 = _try_infer(X3)
        if err3 is None:
            return y_pred, p_pos

    raise err

# -------------------- Sidebar: Model Loader --------------------
st.title("🩺 LPD • Pengujian SVM+PCA")
st.caption("Pipeline: Scaler → PCA (0.95) → SVC")

st.sidebar.header("Model")
uploaded_pkl = st.sidebar.file_uploader("Upload pipeline .pkl (opsional)", type=["pkl"])

@st.cache_resource(show_spinner=False)
def load_pipeline_from_bytes(pkl_bytes: bytes):
    return joblib.load(io.BytesIO(pkl_bytes))

pipeline = None
model_label = None

if uploaded_pkl is not None:
    try:
        pipeline = load_pipeline_from_bytes(uploaded_pkl.getvalue())
        model_label = uploaded_pkl.name
        st.sidebar.success("Model berhasil dimuat ✅")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}")

if pipeline is None:
    default_path = "svm_pca_rbf_new.pkl"  # ubah jika nama file berbeda
    if os.path.exists(default_path):
        try:
            with open(default_path, "rb") as f:
                pipeline = load_pipeline_from_bytes(f.read())
            model_label = os.path.basename(default_path)
            st.sidebar.info("Model dimuat dari file lokal.")
        except Exception as e:
            st.sidebar.error(f"Gagal memuat {default_path}: {e}")
    else:
        st.sidebar.error("Tidak menemukan model. Upload .pkl atau letakkan file .pkl di folder yang sama lalu refresh.")
        st.stop()

st.sidebar.markdown(f"**Model:** `{model_label}`")
HAS_PREPROCESS = pipeline_has_preprocess(pipeline)

menu = st.sidebar.radio("Pilih menu", ["Uji via File (CSV/XLSX)", "Uji via Form (Realtime)", "Model Inspector"])

# -------------------- Menu 1: Uji via File --------------------
if menu == "Uji via File (CSV/XLSX)":
    st.subheader("📦 Uji via File")
    st.write("Unggah file **CSV/XLSX**. Jika kolom **Result** ada, metrik evaluasi akan dihitung.")
    up_data = st.file_uploader("Upload data", type=["csv","xlsx"])

    if up_data is not None:
        try:
            df = load_any(up_data)
            df = normalize_to_short_cols(df)

            if 'Result' in df.columns:
                uniq = set(pd.Series(df['Result']).dropna().unique().tolist())
                if uniq == {1, 2}:
                    df['Result'] = df['Result'].replace({1:1, 2:0})

            st.markdown("**Preview (setelah normalisasi nama kolom):**")
            st.dataframe(df.head(25), use_container_width=True)

            # Siapkan X untuk model
            if HAS_PREPROCESS:
                X = df.drop(columns=['Result']) if 'Result' in df.columns else df.copy()
                num_like = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
                if num_like:
                    X = impute_numeric_median(X, num_like)
                X = impute_gender(X)
            else:
                if set(['Gender_Female','Gender_Male']).issubset(df.columns):
                    X = df.drop(columns=['Result']) if 'Result' in df.columns else df.copy()
                    for c in TRAIN_SCHEMA:
                        if c not in X.columns: X[c] = 0
                    X = X[TRAIN_SCHEMA]
                    X = impute_numeric_median(X, TRAIN_SCHEMA)
                else:
                    X = build_X_enc_from_raw(df.drop(columns=['Result']) if 'Result' in df.columns else df)

            # Prediksi
            y_pred, prob = safe_predict(pipeline, X, HAS_PREPROCESS)
            out = df.copy()
            out['Pred'] = y_pred
            if prob is not None:
                out['Prob_Pos'] = prob

            st.markdown("**Hasil Prediksi (head):**")
            st.dataframe(out.head(50), use_container_width=True)

            # Metrik jika ada target
            if 'Result' in df.columns:
                y_true = df['Result'].astype(int)
                acc = accuracy_score(y_true, y_pred)
                prec, rec, f1m, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
                roc = roc_auc_score(y_true, prob) if prob is not None else None

                c1,c2,c3,c4,c5 = st.columns(5)
                c1.metric("Accuracy", f"{acc:.3f}")
                c2.metric("Precision (macro)", f"{prec:.3f}")
                c3.metric("Recall (macro)", f"{rec:.3f}")
                c4.metric("F1 (macro)", f"{f1m:.3f}")
                if roc is not None:
                    c5.metric("ROC-AUC", f"{roc:.3f}")

                cm = confusion_matrix(y_true, y_pred, labels=[0,1])
                fig, ax = plt.subplots(figsize=(4,3))
                im = ax.imshow(cm)
                ax.set_title("Confusion Matrix")
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xticklabels(["0","1"]); ax.set_yticklabels(["0","1"])
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, cm[i,j], ha="center", va="center", color="w")
                fig.colorbar(im)
                st.pyplot(fig)

                with st.expander("Classification Report (teks)"):
                    st.text(classification_report(y_true, y_pred, digits=4))

            # Download hasil
            data = out.to_csv(index=False).encode()
            b64 = base64.b64encode(data).decode()
            st.markdown(f'<a download="lpd_predictions.csv" href="data:file/csv;base64,{b64}">⬇️ Download lpd_predictions.csv</a>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan memproses file: {e}")
    else:
        st.info("Silakan unggah file data.")

# -------------------- Menu 2: Uji via Form (Realtime) --------------------
elif menu == "Uji via Form (Realtime)":
    st.subheader("📝 Uji via Form (Realtime)")

    colA, colB = st.columns(2)
    with colA:
        age = st.number_input("Age (tahun)", min_value=0, max_value=120, value=45, step=1)
        total_bil = st.number_input("Total Bilirubin (mg/dL)", min_value=0.0, value=0.8, step=0.1)
        direct_bil = st.number_input("Direct Bilirubin (mg/dL)", min_value=0.0, value=0.2, step=0.1)
        alp = st.number_input("Alkaline Phosphotase (U/L)", min_value=0.0, value=110.0, step=1.0)
        alt = st.number_input("SGPT / ALT (U/L)", min_value=0.0, value=30.0, step=1.0)
    with colB:
        gender_id = st.selectbox("Gender", ["Male", "Female"])
        ast = st.number_input("SGOT / AST (U/L)", min_value=0.0, value=28.0, step=1.0)
        tprot = st.number_input("Total Proteins (g/dL)", min_value=0.0, value=7.0, step=0.1)
        alb = st.number_input("Albumin (g/dL)", min_value=0.0, value=4.2, step=0.1)
        agr = st.number_input("A/G Ratio", min_value=0.0, value=1.4, step=0.1)

    row = {
        "Age": age, "Gender": gender_id,
        "Total Bilirubin": total_bil, "Direct Bilirubin": direct_bil,
        "Alkaline Phosphotase": alp, "SGPT": alt, "SGOT": ast,
        "Total Proteins": tprot, "Albumin": alb, "A/G Ratio": agr,
    }
    one = pd.DataFrame([row])
    st.markdown("**Data yang akan diuji:**")
    st.dataframe(one, use_container_width=True)

    if st.button("🔮 Prediksi"):
        try:
            if HAS_PREPROCESS:
                X_form = impute_gender(one.copy())
                num_like = [c for c in X_form.columns if pd.api.types.is_numeric_dtype(X_form[c])]
                if num_like: X_form = impute_numeric_median(X_form, num_like)
            else:
                X_form = build_X_enc_from_raw(one)

            y_pred, prob = safe_predict(pipeline, X_form, HAS_PREPROCESS)
            pred = int(y_pred[0])
            ppos = float(prob[0]) if prob is not None else 0.5

            # === KARTU HASIL OPSI-1 ===
            render_output_option1(pred, ppos)

            # Near boundary hint
            if near_boundary(ppos):
                st.markdown(
                    '<div class="badge badge-info">Near decision boundary</div>'
                    '<div style="opacity:.85;margin-top:.25rem;">'
                    'Probabilitas mendekati 50% → hasil bisa berubah dengan sedikit perubahan nilai.'
                    '</div>',
                    unsafe_allow_html=True
                )

            # Gauge
            st.markdown("#### 🎯 Visualisasi Risiko")
            render_gauge_html(ppos)

        except Exception as e:
            st.error(f"Gagal memprediksi: {e}")

# -------------------- Menu 3: Model Inspector --------------------
elif menu=="Model Inspector":
    st.subheader("🧩 Model Inspector")
    pca = pipeline.named_steps.get("pca", None)
    if pca and hasattr(pca,"explained_variance_ratio_"):
        evr = pca.explained_variance_ratio_
        cum = np.cumsum(evr)
        df_pca = pd.DataFrame({
            "PC":[f"PC{i+1}" for i in range(len(evr))],
            "ExplainedVar":evr,
            "Cumulative":cum
        })
        st.dataframe(df_pca, use_container_width=True)

        # Scree plot
        fig1, ax1 = plt.subplots()
        ax1.plot(range(1,len(evr)+1),evr,marker='o',label='Explained Var Ratio')
        ax1.plot(range(1,len(cum)+1),cum,marker='s',linestyle='--',label='Cumulative')
        ax1.set_xlabel("PC")
        ax1.set_ylabel("Proporsi Varians")
        ax1.legend()
        st.pyplot(fig1)

        # Heatmap loadings bila tersedia
        if hasattr(pca,"components_"):
            # Gunakan TRAIN_SCHEMA untuk nama fitur (pipeline tanpa preprocess)
            loadings = pca.components_.T
            fig2, ax2 = plt.subplots()
            im = ax2.imshow(loadings, aspect='auto')
            ax2.set_xticks(range(loadings.shape[1]))
            ax2.set_xticklabels([f"PC{i+1}" for i in range(loadings.shape[1])])
            ax2.set_yticks(range(len(TRAIN_SCHEMA)))
            ax2.set_yticklabels(TRAIN_SCHEMA)
            ax2.set_title("PCA Loadings")
            fig2.colorbar(im, ax=ax2)
            st.pyplot(fig2)
    else:
        st.info("PCA tidak tersedia pada model.")
