# app.py
# =========================================================
# Streamlit — Uji Model SVM+PCA (RBF) untuk LPD
# - Sidebar Model: tampilkan hanya nama model yang dipakai (tanpa radio)
# - Model Inspector: tampilkan jumlah fitur setelah scaling & setelah PCA
# - Form: rentang nilai normal muncul di bawah masing-masing input
# =========================================================

import os, io, base64, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, confusion_matrix, classification_report
)
import plotly.graph_objects as go

# -------------------- Konfigurasi Halaman --------------------
st.set_page_config(
    page_title="LPD • SVM+PCA (RBF) — Pengujian",
    page_icon="🩺",
    layout="wide",
)

# -------------------- Konstanta Kolom --------------------
TARGET = "Result"
NUM_COLS = [
    "Age of the patient", "Total Bilirubin", "Direct Bilirubin",
    "Alkphos Alkaline Phosphatase", "Sgpt Alamine Aminotransferase",
    "Sgot Aspartate Aminotransferase", "Total Prot", "Albi Albumin", "A/G Ratio"
]
CAT_COLS = ["Gender of the patient"]
EXPECTED_COLUMNS = NUM_COLS + CAT_COLS + [TARGET]

COLUMN_NORMALIZATION_MAP = {
    'Age of the patient': 'Age of the patient',
    'Gender of the patient': 'Gender of the patient',
    'Total Bilirubin': 'Total Bilirubin',
    'Direct Bilirubin': 'Direct Bilirubin',
    '\xa0Alkphos Alkaline Phosphotase': 'Alkphos Alkaline Phosphatase',
    'Alkphos Alkaline Phosphatase': 'Alkphos Alkaline Phosphatase',
    'Alkphos Alkaline Phosphatase ': 'Alkphos Alkaline Phosphatase',
    'Alkphos Alkaline Phosphatase\xa0': 'Alkphos Alkaline Phosphatase',
    'Alkphos Alkaline Phosphotase': 'Alkphos Alkaline Phosphatase',
    '\xa0Sgpt Alamine Aminotransferase': 'Sgpt Alamine Aminotransferase',
    'Sgpt Alamine Aminotransferase': 'Sgpt Alamine Aminotransferase',
    'Sgpt Alami': 'Sgpt Alamine Aminotransferase',
    'Sgot Aspartate Aminotransferase': 'Sgot Aspartate Aminotransferase',
    'Total Protiens': 'Total Prot',
    'Total Prot': 'Total Prot',
    '\xa0ALB Albumin': 'Albi Albumin',
    'ALB Albumin': 'Albi Albumin',
    'Albi Albumin': 'Albi Albumin',
    'A/G Ratio Albumin and Globulin Ratio': 'A/G Ratio',
    'A/G Ratio': 'A/G Ratio',
    'Result': 'Result',
}

# -------------------- Rentang Nilai Normal (Panduan) --------------------
NORMAL_RANGES_ADULT = {
    "Total Bilirubin": (0.1, 1.2, "mg/dL"),
    "Direct Bilirubin": (0.0, 0.3, "mg/dL"),
    "Alkphos Alkaline Phosphatase": (44, 147, "U/L"),   # dewasa
    "Sgpt Alamine Aminotransferase": (7, 56, "U/L"),
    "Sgot Aspartate Aminotransferase": (10, 40, "U/L"),
    "Total Prot": (6.0, 8.3, "g/dL"),
    "Albi Albumin": (3.5, 5.0, "g/dL"),
    "A/G Ratio": (1.0, 2.1, "rasio"),
}

def normal_ranges(age:int, gender_id:str):
    ranges = {k:(lo,hi,unit) for k,(lo,hi,unit) in NORMAL_RANGES_ADULT.items()}
    note = None
    if age < 18:
        note = "Perhatian: usia < 18 tahun — beberapa nilai (mis. ALP) dapat lebih tinggi dari rentang dewasa."
    return ranges, note

# -------------------- Utils --------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: COLUMN_NORMALIZATION_MAP.get(c, c) for c in df.columns})

def coerce_numeric(df: pd.DataFrame, numeric_cols=NUM_COLS):
    df2 = df.copy()
    for c in numeric_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2

def fix_target_binary(df: pd.DataFrame):
    if TARGET in df.columns:
        uniq = set(pd.Series(df[TARGET]).dropna().unique().tolist())
        if uniq == {1, 2}:  # map {1,2}->{1,0} seperti training
            df[TARGET] = df[TARGET].replace({1:1, 2:0})
    return df

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
        
def risk_level(prob):
    if prob < 0.33: return "Low", "#16a34a"
    if prob < 0.66: return "Moderate", "#f59e0b"
    return "High", "#dc2626"

def list_anomalies(one_row: pd.Series, ranges: dict):
    notes = []
    for k,(lo,hi,unit) in ranges.items():
        v = float(one_row[k])
        if v < lo:
            notes.append(f"⬇️ **{k}** {v} {unit} (min {lo})")
        elif v > hi:
            notes.append(f"⬆️ **{k}** {v} {unit} (max {hi})")
    return notes

def tornado_sensitivity(pipeline, base_df: pd.DataFrame, num_cols, rel=0.2):
    """One-way sensitivity ±rel (default 20%) untuk prob kelas positif."""
    try:
        base_prob = float(pipeline.predict_proba(base_df)[:,1][0])
    except Exception:
        # fallback pakai decision_function → sigmoid
        dv = float(pipeline.decision_function(base_df)[0])
        base_prob = 1/(1+np.exp(-dv))

    rows = []
    for col in num_cols:
        v = float(base_df[col].iloc[0])
        v_minus = max(0.0, v*(1-rel))
        v_plus  = max(0.0, v*(1+rel))

        df_minus = base_df.copy(); df_minus.loc[:, col] = v_minus
        df_plus  = base_df.copy(); df_plus.loc[:, col]  = v_plus

        p_minus = float(pipeline.predict_proba(df_minus)[:,1][0])
        p_plus  = float(pipeline.predict_proba(df_plus)[:,1][0])

        delta = p_plus - p_minus  # rentang pengaruh
        rows.append((col, delta, p_minus, p_plus))

    # urutkan berdasarkan |delta|, ambil Top-5
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return rows[:5], base_prob

def b64_download(df: pd.DataFrame, filename="predictions.csv"):
    data = df.to_csv(index=False).encode()
    b64 = base64.b64encode(data).decode()
    return f'<a download="{filename}" href="data:file/csv;base64,{b64}">⬇️ Download {filename}</a>'

@st.cache_resource(show_spinner=False)
def load_pipeline_from_bytes(pkl_bytes: bytes):
    return joblib.load(io.BytesIO(pkl_bytes))

# -------------------- Sidebar (Model hanya teks nama) --------------------
import sys, sklearn, numpy as np, scipy
st.sidebar.caption(f"Py {sys.version.split()[0]} • sklearn {sklearn.__version__} • numpy {np.__version__} • scipy {scipy.__version__}")
st.title("🩺 LPD • Pengujian SVM+PCA (RBF)")
st.caption("Pipeline: preprocess → scaler → PCA → SVC (rbf)")

st.sidebar.header("Model")
uploaded_pkl = st.sidebar.file_uploader("Upload pipeline .pkl (opsional)", type=["pkl"])
pipeline = None
model_label = None

if uploaded_pkl is not None:
    try:
        pipeline = load_pipeline_from_bytes(uploaded_pkl.getvalue())
        model_label = uploaded_pkl.name
        st.sidebar.success("Model berhasil dimuat ✅")
    except Exception as e:
        st.sidebar.error(f"Gagal memuat model: {e}")

# Jika tidak upload, gunakan default path
if pipeline is None:
    default_path = "svm_pca_rbf.pkl"
    if os.path.exists(default_path):
        try:
            with open(default_path, "rb") as f:
                pipeline = load_pipeline_from_bytes(f.read())
            model_label = os.path.basename(default_path)
            st.sidebar.info(f"Model dimuat dari file lokal.")
        except Exception as e:
            st.sidebar.error(f"Gagal memuat {default_path}: {e}")
    else:
        st.sidebar.error(f"Tidak menemukan model. Upload .pkl atau simpan {default_path} di folder yang sama.")
        st.stop()

# Tampilkan hanya teks nama model yang digunakan
st.sidebar.markdown(f"**Model yang digunakan:** `{model_label}`")

# Menu utama (tetap untuk navigasi fitur)
menu = st.sidebar.radio("Pilih menu", ["Uji via File (CSV/XLSX)", "Uji via Form (Realtime)", "Model Inspector"])

# -------------------- Guard: model wajib ada --------------------
if pipeline is None:
    st.info("Silakan muat model terlebih dahulu dari sidebar.")
    st.stop()

# -------------------- Menu 1: Uji via File --------------------
if menu == "Uji via File (CSV/XLSX)":
    st.subheader("📦 Uji via File")
    st.write("Unggah file **CSV/XLSX** (mis. `test.csv.xlsx`). Jika kolom **Result** ada, metrik evaluasi akan dihitung.")

    up_data = st.file_uploader("Upload data", type=["csv","xlsx"])
    if up_data is not None:
        try:
            df = load_any(up_data)
            df = normalize_columns(df)
            df = fix_target_binary(df)
            df = coerce_numeric(df)
            st.markdown("**Preview data:**")
            st.dataframe(df.head(25), use_container_width=True)

            X = df.drop(columns=[TARGET]) if TARGET in df.columns else df.copy()
            y_true = df[TARGET].astype(int) if TARGET in df.columns else None

            y_pred = pipeline.predict(X)
            out = df.copy()
            out["Pred"] = y_pred

            # probabilitas jika tersedia
            try:
                prob = pipeline.predict_proba(X)[:,1]
                out["Prob_Pos"] = prob
            except Exception:
                prob = None

            st.markdown("**Hasil Prediksi (head):**")
            st.dataframe(out.head(50), use_container_width=True)

            # Metrik (bila target ada)
            if y_true is not None:
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

                # Confusion matrix
                cm = confusion_matrix(y_true, y_pred, labels=[0,1])
                fig, ax = plt.subplots(figsize=(4,3))
                im = ax.imshow(cm, cmap="Blues")
                ax.set_title("Confusion Matrix")
                ax.set_xticks([0,1]); ax.set_yticks([0,1])
                ax.set_xticklabels(["0","1"]); ax.set_yticklabels(["0","1"])
                ax.set_xlabel("Predicted"); ax.set_ylabel("True")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, cm[i,j], ha="center", va="center")
                fig.colorbar(im)
                st.pyplot(fig)

                with st.expander("Classification Report (teks)"):
                    st.text(classification_report(y_true, y_pred, digits=4))

            # Download
            st.markdown(b64_download(out, "lpd_predictions.csv"), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Terjadi kesalahan memproses file: {e}")
    else:
        st.info("Silakan unggah file data.")

# -------------------- Menu 2: Uji via Form (Realtime) --------------------
elif menu == "Uji via Form (Realtime)":
    st.subheader("📝 Uji via Form (Realtime)")
    st.write("Pilih **Umur** & **Gender** terlebih dahulu. Panduan **rentang nilai normal** akan tampil **di bawah masing-masing input**.")

    # --- Identitas dasar (tentukan rentang normal) ---
    colA, colB = st.columns([1,1])
    with colA:
        age = st.number_input("Umur (tahun)", min_value=1, max_value=120, value=45, step=1)
    with colB:
        gender_id = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
    gender_mapped = "Male" if gender_id == "Laki-laki" else "Female"

    ranges, note = normal_ranges(age, gender_mapped)
    if note:
        st.caption(f"ℹ️ {note}")

    # Helper: default ke tengah rentang normal bila memungkinkan
    def mid(lo, hi, decimals=1):
        return round((lo+hi)/2, decimals)

    st.markdown("---")
    st.markdown("**Isi Data Klinis:**")

    c1,c2 = st.columns(2)
    with c1:
        total_bil = st.number_input(
            "Total Bilirubin (mg/dL)",
            min_value=0.0, value=mid(*ranges["Total Bilirubin"][:2], 1), step=0.1
        )
        st.caption(f"Rentang normal dewasa: {ranges['Total Bilirubin'][0]}–{ranges['Total Bilirubin'][1]} {ranges['Total Bilirubin'][2]}")

        direct_bil = st.number_input(
            "Direct Bilirubin (mg/dL)",
            min_value=0.0, value=mid(*ranges["Direct Bilirubin"][:2], 1), step=0.1
        )
        st.caption(f"Rentang normal dewasa: {ranges['Direct Bilirubin'][0]}–{ranges['Direct Bilirubin'][1]} {ranges['Direct Bilirubin'][2]}")

        alp = st.number_input(
            "Alkphos Alkaline Phosphatase (U/L)",
            min_value=0.0, value=float(mid(*ranges["Alkphos Alkaline Phosphatase"][:2], 0)), step=1.0
        )
        st.caption(f"Rentang normal dewasa: {ranges['Alkphos Alkaline Phosphatase'][0]}–{ranges['Alkphos Alkaline Phosphatase'][1]} {ranges['Alkphos Alkaline Phosphatase'][2]}")

        alt = st.number_input(
            "Sgpt Alamine Aminotransferase (U/L)",
            min_value=0.0, value=float(mid(*ranges["Sgpt Alamine Aminotransferase"][:2], 0)), step=1.0
        )
        st.caption(f"Rentang normal dewasa: {ranges['Sgpt Alamine Aminotransferase'][0]}–{ranges['Sgpt Alamine Aminotransferase'][1]} {ranges['Sgpt Alamine Aminotransferase'][2]}")

        ast = st.number_input(
            "Sgot Aspartate Aminotransferase (U/L)",
            min_value=0.0, value=float(mid(*ranges["Sgot Aspartate Aminotransferase"][:2], 0)), step=1.0
        )
        st.caption(f"Rentang normal dewasa: {ranges['Sgot Aspartate Aminotransferase'][0]}–{ranges['Sgot Aspartate Aminotransferase'][1]} {ranges['Sgot Aspartate Aminotransferase'][2]}")

    with c2:
        tprot = st.number_input(
            "Total Protein (g/dL)",
            min_value=0.0, value=mid(*ranges["Total Prot"][:2], 1), step=0.1
        )
        st.caption(f"Rentang normal dewasa: {ranges['Total Prot'][0]}–{ranges['Total Prot'][1]} {ranges['Total Prot'][2]}")

        alb = st.number_input(
            "Albumin (g/dL)",
            min_value=0.0, value=mid(*ranges["Albi Albumin"][:2], 1), step=0.1
        )
        st.caption(f"Rentang normal dewasa: {ranges['Albi Albumin'][0]}–{ranges['Albi Albumin'][1]} {ranges['Albi Albumin'][2]}")

        agr = st.number_input(
            "A/G Ratio",
            min_value=0.0, value=mid(*ranges["A/G Ratio"][:2], 1), step=0.1
        )
        st.caption(f"Rentang normal dewasa: {ranges['A/G Ratio'][0]}–{ranges['A/G Ratio'][1]} {ranges['A/G Ratio'][2]}")

    # Bangun 1 baris fitur sesuai skema training
    row = {
        "Age of the patient": age,
        "Gender of the patient": gender_mapped,
        "Total Bilirubin": total_bil,
        "Direct Bilirubin": direct_bil,
        "Alkphos Alkaline Phosphatase": alp,
        "Sgpt Alamine Aminotransferase": alt,
        "Sgot Aspartate Aminotransferase": ast,
        "Total Prot": tprot,
        "Albi Albumin": alb,
        "A/G Ratio": agr,
    }
    one = pd.DataFrame([row])

    st.markdown("**Data yang akan diuji:**")
    st.dataframe(one, use_container_width=True)

    if st.button("🔮 Prediksi"):
        try:
            # --- Probabilitas dasar ---
            pred = int(pipeline.predict(one)[0])
            try:
                ppos = float(pipeline.predict_proba(one)[:,1][0])
            except Exception:
                dv = float(pipeline.decision_function(one)[0]); ppos = 1/(1+np.exp(-dv))

            # --- Threshold slider (ubah ambang keputusan) ---
            st.markdown("**Decision threshold**")
            thr = st.slider("Ambang positif", 0.05, 0.95, 0.50, 0.01)
            pred_thr = 1 if ppos >= thr else 0

            # --- Badge + ringkasan risiko ---
            level, color = risk_level(ppos)
            st.markdown(
                f"""
                <div style="display:flex;gap:12px;align-items:center;margin:.4rem 0;">
                <div style="background:{color};color:white;padding:.35rem .6rem;border-radius:999px;font-weight:700;">
                    {level} risk
                </div>
                <div style="opacity:.8">Probabilitas positif: <b>{ppos:.1%}</b> • Threshold: <b>{thr:.2f}</b> → Prediksi: <b>{pred_thr}</b></div>
                </div>
                """, unsafe_allow_html=True
            )

            # --- Risk gauge (Plotly) ---
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ppos*100,
                number={'suffix': "%"},
                title={'text': "Risk (Positive class)"},
                gauge={
                    'axis': {'range': [0,100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0,33], 'color': '#E6F4EA'},
                        {'range': [33,66], 'color': '#FEF3C7'},
                        {'range': [66,100], 'color': '#FEE2E2'},
                    ],
                    'threshold': {'line': {'color': '#111', 'width': 3}, 'thickness': 0.8, 'value': thr*100},
                }
            ))
            st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar": False})

            # --- Anomali klinis (dibanding rentang normal) ---
            notes = list_anomalies(one.iloc[0], ranges)
            if notes:
                st.markdown("**🧪 Nilai di luar rentang normal:**")
                st.markdown("\n".join([f"- {n}" for n in notes]))
            else:
                st.markdown("**🧪 Semua nilai berada dalam rentang normal dewasa.**")

            # --- Tornado chart: one-way sensitivity ±20% ---
            top_rows, base_prob = tornado_sensitivity(pipeline, one, NUM_COLS, rel=0.2)
            if top_rows:
                st.markdown("**📈 Sensitivitas (±20%) — TOP 5 fitur paling berpengaruh**")
                # visual pakai matplotlib (biar ringan)
                labels = [r[0] for r in top_rows][::-1]
                deltas = [r[1]*100 for r in top_rows][::-1]   # ke persen
                fig, ax = plt.subplots(figsize=(6, 3.8))
                ax.barh(labels, deltas)
                ax.axvline(0, color="#999", linewidth=1)
                ax.set_xlabel("Δ Probabilitas positif (p.p.) untuk +20% vs -20%")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Gagal memprediksi: {e}")

# -------------------- Menu 3: Model Inspector --------------------
elif menu == "Model Inspector":
    st.subheader("🧩 Model Inspector")

    # ---- Jumlah fitur setelah scaling (pre-PCA) ----
    n_after_scaling = None
    try:
        pre = pipeline.named_steps.get("preprocess", None)
        if pre is not None:
            try:
                n_after_scaling = len(pre.get_feature_names_out())
            except Exception:
                # fallback: transform satu baris dummy
                template = {c: np.nan for c in NUM_COLS}
                # ambil kategori gender pertama jika tersedia
                try:
                    ohe = pre.named_transformers_["cat"].named_steps["ohe"]
                    template["Gender of the patient"] = ohe.categories_[0][0]
                except Exception:
                    template["Gender of the patient"] = "Male"
                Xtmp = pd.DataFrame([template])
                n_after_scaling = pre.transform(Xtmp).shape[1]
    except Exception as e:
        st.info(f"Tidak bisa menghitung fitur pre-PCA: {e}")

    # ---- Jumlah fitur setelah PCA ----
    n_after_pca = None
    var_sum = None
    try:
        pca = pipeline.named_steps.get("pca", None)
        if pca is not None and hasattr(pca, "n_components_"):
            n_after_pca = int(pca.n_components_)
            var_sum = float(np.sum(pca.explained_variance_ratio_))
    except Exception as e:
        st.info(f"Tidak bisa membaca PCA: {e}")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Fitur setelah scaling (pre-PCA)", f"{n_after_scaling if n_after_scaling is not None else '-'}")
    with c2:
        st.metric("Fitur setelah PCA", f"{n_after_pca if n_after_pca is not None else '-'}")
    if var_sum is not None:
        st.caption(f"Total explained variance PCA: {var_sum:.3f}")

    # ---- Detail PCA (opsional) ----
    try:
        if pca is not None and hasattr(pca, "n_components_"):
            df_pca = pd.DataFrame({
                "PC": [f"PC{i+1}" for i in range(n_after_pca)],
                "ExplainedVariance": pca.explained_variance_ratio_,
            })
            st.markdown("**Rincian Komponen PCA:**")
            st.dataframe(df_pca, use_container_width=True)
    except Exception:
        pass

    # ---- Nama fitur setelah encoding (sebelum scaling & PCA) ----
    try:
        if pre is not None:
            try:
                features = pre.get_feature_names_out()
                st.markdown("**Nama Fitur (sebelum scaling & PCA):**")
                st.dataframe(pd.DataFrame({"Feature": features}), use_container_width=True, height=350)
            except Exception:
                st.info("get_feature_names_out() tidak tersedia (sklearn lawas).")
        else:
            st.info("Preprocess (ColumnTransformer) tidak ditemukan.")
    except Exception as e:
        st.error(f"Gagal membaca fitur: {e}")