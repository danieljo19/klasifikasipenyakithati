# app.py
# =========================================================
# Streamlit — Uji Model SVM+PCA (RBF) untuk LPD
# - Nilai normal dewasa (Mayo Clinic)
# - ALT/AST/ALP/Albumin pediatrik (age & sex specific)
# - Caption sumber publik di bawah input
# - Visual risiko: Gauge setengah lingkaran (HTML/CSS)
# - Risk explanation + near decision boundary + UI cards (sublist tanpa bullet ganda)
# - Model Inspector: Ringkasan PCA otomatis (top-k fitur dominan per PC)
# =========================================================

import os, io, base64, warnings, sys, textwrap, re
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

# -------------------- Konfigurasi Halaman --------------------
st.set_page_config(
    page_title="LPD • SVM+PCA (RBF) — Pengujian",
    page_icon="🩺",
    layout="wide",
)

# -------------------- Global CSS --------------------
st.markdown("""
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 2rem;}
.badge {display:inline-block;padding:.30rem .60rem;border-radius:999px;font-weight:700;color:#fff;}
.badge-low {background:#16a34a;}
.badge-mod {background:#f59e0b;}
.badge-high{background:#dc2626;}
.badge-info{background:#0ea5e9;}
.card{
  background:#0f172a; border:1px solid #1f2937;
  border-radius:12px; padding:14px 16px; margin:10px 0;
}
.card h4{margin:.1rem 0 .4rem 0;}
ul.clean{margin:.2rem 0 .2rem 1.25rem;}
ul.clean li{margin:.25rem 0;}
/* sublist tanpa bullet kedua (hanya indent/tab) */
.sublist{list-style:none; margin-left:1.25rem; padding-left:.25rem;}
.sublist li{margin:.18rem 0;}
.legend {display:flex;gap:18px; align-items:center; flex-wrap:wrap; margin-top:.35rem;}
.dot{display:inline-block;width:12px;height:12px;border-radius:999px;margin-right:6px;}
.dot-green{background:#22c55e;}
.dot-yellow{background:#fbbf24;}
.dot-red{background:#ef4444;}
[data-testid="stTable"] th {text-align:center;}
</style>
""", unsafe_allow_html=True)

# -------------------- Skema Kolom --------------------
TARGET = "Result"
NUM_COLS = [
    "Age of the patient", "Total Bilirubin", "Direct Bilirubin",
    "Alkphos Alkaline Phosphatase", "Sgpt Alamine Aminotransferase",
    "Sgot Aspartate Aminotransferase", "Total Prot",
    "Albi Albumin", "A/G Ratio"
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

# -------------------- Sumber Publik --------------------
RANGE_SOURCES = {
    "Total Bilirubin": ("Mayo Clinic", "https://www.mayoclinic.org/tests-procedures/liver-function-tests/about/pac-20394595"),
    "Direct Bilirubin": ("Mayo Clinic", "https://www.mayoclinic.org/tests-procedures/liver-function-tests/about/pac-20394595"),
    "Alkphos Alkaline Phosphatase": ("Mayo Clinic (dewasa) / Mayo Pediatric & CHOP (anak)", "https://www.mayoclinic.org/tests-procedures/liver-function-tests/about/pac-20394595"),
    "Sgpt Alamine Aminotransferase": ("Mayo (dewasa) / CHOP-Medscape (anak)", "https://www.mayoclinic.org/tests-procedures/liver-function-tests/about/pac-20394595"),
    "Sgot Aspartate Aminotransferase": ("Mayo (dewasa) / CHOP-Medscape (anak)", "https://www.mayoclinic.org/tests-procedures/liver-function-tests/about/pac-20394595"),
    "Total Prot": ("Mayo Clinic", "https://www.mayoclinic.org/tests-procedures/liver-function-tests/about/pac-20394595"),
    "Albi Albumin": ("Mayo (dewasa) / CHOP (anak)", "https://www.chop.edu/sites/default/files/2024-06/chop-labs-reference-ranges.pdf"),
    "A/G Ratio": ("Cleveland Clinic", "https://my.clevelandclinic.org/health/diagnostics/17662-liver-function-tests"),
}

# -------------------- Rentang Dewasa (Mayo Clinic) --------------------
NORMAL_RANGES_ADULT = {
    "Sgpt Alamine Aminotransferase": (7, 55, "U/L"),     # ALT
    "Sgot Aspartate Aminotransferase": (8, 48, "U/L"),   # AST
    "Alkphos Alkaline Phosphatase": (40, 129, "U/L"),    # ALP
    "Albi Albumin": (3.5, 5.0, "g/dL"),
    "Total Prot": (6.3, 7.9, "g/dL"),
    "Total Bilirubin": (0.1, 1.2, "mg/dL"),
    "Direct Bilirubin": (0.0, 0.3, "mg/dL"),
    "A/G Ratio": (1.0, 2.1, "rasio"),
}

# -------------------- Pediatrik (age & sex specific) --------------------
_ALP_BANDS = [
    (0,    0.08, (70, 300), (70, 300)),   # 1–29 hari
    (0.08, 0.92, (70, 345), (70, 345)),   # 1–11 bulan
    (1,    3,    (145,320), (145,320)),
    (4,    6,    (150,380), (150,380)),
    (7,    9,    (175,420), (175,420)),
    (10,   11,   (135,530), (130,560)),
    (12,   13,   (200,495), (105,420)),
    (14,   15,   (130,525), (70, 230)),
    (16,   20,   (65, 260), (50, 130)),
]
_ALP_ADULT = (40, 129)

def _alp_range(age_years: float, gender_id: str):
    if age_years >= 21:
        lo, hi = _ALP_ADULT
        return (lo, hi, "U/L")
    sex = "male" if str(gender_id).lower().startswith("male") else "female"
    for amin, amax, male_rng, fem_rng in _ALP_BANDS:
        if amin <= age_years <= amax:
            lo, hi = male_rng if sex == "male" else fem_rng
            return (lo, hi, "U/L")
    lo, hi = (65,260) if sex=="male" else (50,130)
    return (lo, hi, "U/L")

def _alb_range(age_years: float):
    if age_years < 1:       return (3.5, 5.4, "g/dL")
    if 1 <= age_years <= 3: return (3.5, 4.6, "g/dL")
    if 4 <= age_years <= 6: return (3.5, 5.2, "g/dL")
    if 7 <= age_years < 20: return (3.7, 5.6, "g/dL")
    return (3.5, 5.0, "g/dL")

def _alt_range(age_years: float):
    if age_years < 1:          return (5, 60, "U/L")
    if 1 <= age_years <= 3:    return (5, 55, "U/L")
    if 4 <= age_years <= 6:    return (5, 50, "U/L")
    if 7 <= age_years <= 12:   return (5, 45, "U/L")
    if 13 <= age_years <= 18:  return (5, 40, "U/L")
    return (7, 55, "U/L")

def _ast_range(age_years: float):
    if age_years < 1:          return (20, 75, "U/L")
    if 1 <= age_years <= 3:    return (15, 60, "U/L")
    if 4 <= age_years <= 6:    return (15, 55, "U/L")
    if 7 <= age_years <= 12:   return (15, 50, "U/L")
    if 13 <= age_years <= 18:  return (10, 45, "U/L")
    return (8, 48, "U/L")

# -------------------- Builder Rentang Normal --------------------
def normal_ranges(age: int, gender_id: str):
    age_years = float(age)
    ranges = dict(NORMAL_RANGES_ADULT)
    ranges["Alkphos Alkaline Phosphatase"] = _alp_range(age_years, gender_id)
    ranges["Albi Albumin"] = _alb_range(age_years)
    ranges["Sgpt Alamine Aminotransferase"] = _alt_range(age_years)
    ranges["Sgot Aspartate Aminotransferase"] = _ast_range(age_years)
    note = None
    if age_years < 18:
        note = "Mode pediatrik aktif: ALT, AST, ALP, dan Albumin sudah disesuaikan usia (ALP juga jenis kelamin)."
    return ranges, note

def _caption_for(param: str, rng: tuple, age: int, gender_id: str):
    lo, hi, unit = rng
    src_name, src_url = RANGE_SOURCES.get(param, ("Sumber klinis", "#"))
    extra = ""
    if param in ("Alkphos Alkaline Phosphatase", "Albi Albumin", "Sgpt Alamine Aminotransferase", "Sgot Aspartate Aminotransferase"):
        if age < 18:
            extra = " (pediatrik; disesuaikan usia"
            if param == "Alkphos Alkaline Phosphatase":
                extra += "/jenis kelamin"
            extra += ")"
    return f"Rentang normal: {lo}–{hi} {unit}{extra}. Sumber: [{src_name}]({src_url})."

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
        if uniq == {1, 2}:
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

def near_boundary(prob, window=0.05):
    return abs(prob - 0.5) <= window

def list_anomalies(one_row: pd.Series, ranges: dict):
    notes = []
    for k,(lo,hi,unit) in ranges.items():
        if k not in one_row: 
            continue
        v = float(one_row[k])
        if v < lo:
            notes.append(f"⬇️ <b>{k}</b> {v} {unit} (min {lo})")
        elif v > hi:
            notes.append(f"⬆️ <b>{k}</b> {v} {unit} (max {hi})")
    return notes

def near_normal_limits(one_row: pd.Series, ranges: dict, tol=0.10):
    findings = []
    for k,(lo,hi,unit) in ranges.items():
        if k not in one_row:
            continue
        v = float(one_row[k])
        if not (lo <= v <= hi):
            continue
        width = hi - lo
        if width <= 0: 
            continue
        if (v - lo) <= tol*width:
            findings.append(f"{k} {v} {unit} (dekat batas bawah {lo})")
        elif (hi - v) <= tol*width:
            findings.append(f"{k} {v} {unit} (dekat batas atas {hi})")
    return findings

def tornado_sensitivity(pipeline, base_df: pd.DataFrame, num_cols, rel=0.2):
    try:
        base_prob = float(pipeline.predict_proba(base_df)[:,1][0])
    except Exception:
        dv = float(pipeline.decision_function(base_df)[0])
        base_prob = 1/(1+np.exp(-dv))
    rows = []
    for col in num_cols:
        if col not in base_df.columns: 
            continue
        v = float(base_df[col].iloc[0])
        v_minus = max(0.0, v*(1-rel))
        v_plus  = max(0.0, v*(1+rel))
        df_minus = base_df.copy(); df_minus.loc[:, col] = v_minus
        df_plus  = base_df.copy(); df_plus.loc[:, col]  = v_plus
        try:
            p_minus = float(pipeline.predict_proba(df_minus)[:,1][0])
            p_plus  = float(pipeline.predict_proba(df_plus)[:,1][0])
        except Exception:
            dv_m = float(pipeline.decision_function(df_minus)[0])
            dv_p = float(pipeline.decision_function(df_plus)[0])
            p_minus = 1/(1+np.exp(-dv_m))
            p_plus  = 1/(1+np.exp(-dv_p))
        delta = p_plus - p_minus
        rows.append((col, delta, p_minus, p_plus))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return rows[:5], base_prob

@st.cache_resource(show_spinner=False)
def load_pipeline_from_bytes(pkl_bytes: bytes):
    return joblib.load(io.BytesIO(pkl_bytes))

def md_bold_to_html(s: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)

# ---------- Visual helper (HTML/CSS) ----------
def render_gauge_html(prob: float):
    """Semicircle gauge (low→left, high→right) dengan jarum & label."""
    pct = max(0.0, min(prob*100.0, 100.0))
    angle = -90 + (pct * 1.8)  # 0–100% -> -90..+90 deg
    html = f"""
<div style="width:100%;display:flex;justify-content:center;">
  <div style="position:relative;width:560px;height:280px;">
    <div style="
      position:absolute;left:0;bottom:0;width:100%;height:100%;
      border-top-left-radius:560px;border-top-right-radius:560px;
      background:
        conic-gradient(from 180deg,
          #E6F4EA 0deg 120deg,
          #FEF3C7 120deg 240deg,
          #FEE2E2 240deg 360deg
        );
      -webkit-mask: radial-gradient(circle at 50% 100%, transparent 190px, #000 191px);
              mask: radial-gradient(circle at 50% 100%, transparent 190px, #000 191px);
    "></div>

    <div style="
      position:absolute;left:50%;bottom:0;transform-origin:50% 100%;
      transform: translateX(-50%) rotate({angle}deg);
      width:4px;height:185px;background:#e5e7eb;border-radius:2px;z-index:2;
      box-shadow:0 0 2px rgba(0,0,0,.45);
    "></div>

    <div style="
      position:absolute;left:50%;bottom:0;transform:translate(-50%, 10px);
      width:16px;height:16px;background:#e5e7eb;border-radius:50%;z-index:3;
      box-shadow:0 0 2px rgba(0,0,0,.45);
    "></div>

    <div style="
      position:absolute;left:0;right:0;bottom:-6px;text-align:center;
      color:#e5e7eb;font-size:44px;font-weight:700;">
      {pct:.1f}%
    </div>
  </div>
</div>
"""
    components.html(textwrap.dedent(html), height=300)

def render_near_boundary_badge(ppos: float):
    if near_boundary(ppos):
        st.markdown(
            """
            <div class="badge badge-info">Near decision boundary</div>
            <div style="opacity:.8;margin-top:.25rem;">
                Probabilitas mendekati 50% → prediksi lebih mudah berubah bila ada sedikit perubahan nilai.
            </div>
            """,
            unsafe_allow_html=True
        )

# ---------- PCA Inspector Helpers ----------
def prettify_feat_name(name: str) -> str:
    s = name.split("__")[-1]
    if "Gender of the patient_" in s:
        return "Gender=" + s.split("Gender of the patient_")[-1]
    return s

def pca_summary(preprocess, pca, top_k=3):
    """Kembalikan DataFrame: PC | ExplainedVariance(%) | Top-k fitur dominan (+/-loading)."""
    try:
        feat_names = [prettify_feat_name(n) for n in preprocess.get_feature_names_out()]
    except Exception:
        feat_names = [f"f{i}" for i in range(pca.components_.shape[1])]

    comps = pca.components_
    evr   = pca.explained_variance_ratio_
    rows = []
    for i, (pc_vec, ev) in enumerate(zip(comps, evr), start=1):
        idx = np.argsort(np.abs(pc_vec))[::-1][:top_k]
        items = []
        for j in idx:
            sign = "+" if pc_vec[j] >= 0 else "−"
            items.append(f"{feat_names[j]} ({sign}{abs(pc_vec[j]):.3f})")
        rows.append({
            "PC": f"PC{i}",
            "ExplainedVariance(%)": round(ev*100, 2),
            f"Top {top_k} features (|loading| terbesar)": ", ".join(items)
        })
    return pd.DataFrame(rows)

# ---------- Risk explanation (sub-list tanpa bullet ganda) ----------
def explain_risk_structured(ppos, top_rows, anomalies, one_row=None, ranges=None):
    """Kembalikan list item terstruktur: [{'text': str, 'sub': [str, ...]}]."""
    items = []
    level, _ = risk_level(ppos)

    items.append({"text": f"Probabilitas kelas positif: **{ppos:.1%}** → kategori **{level} risk**."})

    if near_boundary(ppos):
        items.append({"text": "Hasil berada **dekat decision boundary** (≈50%). Interpretasi perlu ekstra hati-hati."})

    if top_rows:
        points = []
        for col, _, p_minus, p_plus in top_rows[:3]:
            direction = "naik" if p_plus > p_minus else "turun"
            points.append(f"**{col}** (prob {direction} saat nilai +20%).")
        items.append({"text": "Fitur paling memengaruhi (one-way sensitivity ±20%): " + "; ".join(points)})

    if anomalies:
        items.append({"text": "Nilai di luar rentang normal: " + "; ".join(anomalies)})
    else:
        items.append({"text": "Semua parameter berada dalam rentang normal untuk usia/gender ini."})

    if (ppos >= 0.5) and not anomalies:
        sub = [
            "**Pola multivariat**: kombinasi beberapa nilai normal dapat membentuk **pola yang mirip** pasien positif pada data latih.",
            "**Model non-linier & PCA**: interaksi non-linier antar parameter tetap relevan walau tiap nilai tampak normal.",
            "**Distribusi populasi**: nilai **serempak dekat batas** bisa menaikkan risiko agregat.",
            "**Usia/Gender**: pola spesifik pada data latih dapat memengaruhi keputusan.",
            "**Ambang keputusan**: bila probabilitas mendekati ambang, sedikit perubahan nilai/fitur dapat mengubah hasil.",
            "**Variasi & artefak**: puasa, obat, alkohol, olahraga berat, hemolisis sampel, dll.",
        ]
        if one_row is not None and ranges is not None:
            near_edges = near_normal_limits(one_row, ranges, tol=0.10)
            if near_edges:
                sub.insert(0, "Beberapa parameter **masih normal tetapi mendekati batas**: " + "; ".join(near_edges))
        items.append({"text": "Mengapa model tetap positif meskipun semua nilai normal?", "sub": sub})
        items.append({"text": "Langkah: pertimbangkan **ulang cek lab**, telaah obat/alkohol/aktivitas, tambah panel (GGT, INR, USG), & konsultasi klinis."})

    items.append({"text": "Catatan: hasil model bersifat estimasi; **konfirmasi klinis** tetap diperlukan."})
    return items

# -------------------- Sidebar (Model) --------------------
import sklearn, scipy
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

if pipeline is None:
    default_path = "svm_pca_rbf.pkl"
    if os.path.exists(default_path):
        try:
            with open(default_path, "rb") as f:
                pipeline = load_pipeline_from_bytes(f.read())
            model_label = os.path.basename(default_path)
            st.sidebar.info("Model dimuat dari file lokal.")
        except Exception as e:
            st.sidebar.error(f"Gagal memuat {default_path}: {e}")
    else:
        st.sidebar.error("Tidak menemukan model. Upload .pkl atau simpan svm_pca_rbf.pkl di folder yang sama.")
        st.stop()

st.sidebar.markdown(f"**Model yang digunakan:** `{model_label}`")
menu = st.sidebar.radio("Pilih menu", ["Uji via File (CSV/XLSX)", "Uji via Form (Realtime)", "Model Inspector"])

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

            prob = None
            try:
                prob = pipeline.predict_proba(X)[:,1]
                out["Prob_Pos"] = prob
            except Exception:
                pass

            st.markdown("**Hasil Prediksi (head):**")
            st.dataframe(out.head(50), use_container_width=True)

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
    st.write("Pilih **Umur** & **Gender**. Panduan **rentang normal** tampil di bawah masing-masing input.")

    colA, colB = st.columns([1,1])
    with colA:
        age = st.number_input("Umur (tahun)", min_value=0, max_value=120, value=45, step=1)
    with colB:
        gender_id = st.selectbox("Gender", ["Laki-laki", "Perempuan"])
    gender_mapped = "Male" if gender_id == "Laki-laki" else "Female"

    ranges, note = normal_ranges(age, gender_mapped)
    if note:
        st.caption(f"ℹ️ {note}")

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
        st.caption(_caption_for("Total Bilirubin", ranges["Total Bilirubin"], age, gender_mapped))

        direct_bil = st.number_input(
            "Direct Bilirubin (mg/dL)",
            min_value=0.0, value=mid(*ranges["Direct Bilirubin"][:2], 1), step=0.1
        )
        st.caption(_caption_for("Direct Bilirubin", ranges["Direct Bilirubin"], age, gender_mapped))

        alp = st.number_input(
            "Alkphos Alkaline Phosphatase (U/L)",
            min_value=0.0, value=float(mid(*ranges["Alkphos Alkaline Phosphatase"][:2], 0)), step=1.0
        )
        st.caption(_caption_for("Alkphos Alkaline Phosphatase", ranges["Alkphos Alkaline Phosphatase"], age, gender_mapped))

        alt = st.number_input(
            "Sgpt Alamine Aminotransferase (U/L)",
            min_value=0.0, value=float(mid(*ranges["Sgpt Alamine Aminotransferase"][:2], 0)), step=1.0
        )
        st.caption(_caption_for("Sgpt Alamine Aminotransferase", ranges["Sgpt Alamine Aminotransferase"], age, gender_mapped))

        ast = st.number_input(
            "Sgot Aspartate Aminotransferase (U/L)",
            min_value=0.0, value=float(mid(*ranges["Sgot Aspartate Aminotransferase"][:2], 0)), step=1.0
        )
        st.caption(_caption_for("Sgot Aspartate Aminotransferase", ranges["Sgot Aspartate Aminotransferase"], age, gender_mapped))

    with c2:
        tprot = st.number_input(
            "Total Protein (g/dL)",
            min_value=0.0, value=mid(*ranges["Total Prot"][:2], 1), step=0.1
        )
        st.caption(_caption_for("Total Prot", ranges["Total Prot"], age, gender_mapped))

        alb = st.number_input(
            "Albumin (g/dL)",
            min_value=0.0, value=mid(*ranges["Albi Albumin"][:2], 1), step=0.1
        )
        st.caption(_caption_for("Albi Albumin", ranges["Albi Albumin"], age, gender_mapped))

        agr = st.number_input(
            "A/G Ratio",
            min_value=0.0, value=mid(*ranges["A/G Ratio"][:2], 1), step=0.1
        )
        st.caption(_caption_for("A/G Ratio", ranges["A/G Ratio"], age, gender_mapped))

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
            pred = int(pipeline.predict(one)[0])
            try:
                ppos = float(pipeline.predict_proba(one)[:,1][0])
            except Exception:
                dv = float(pipeline.decision_function(one)[0]); ppos = 1/(1+np.exp(-dv))

            level, color = risk_level(ppos)
            st.markdown(
                f"""
                <div style="display:flex;gap:12px;align-items:center;margin:.4rem 0;">
                  <div class="badge" style="background:{color}">{level} risk</div>
                  <div style="opacity:.85">Probabilitas positif: <b>{ppos:.1%}</b></div>
                </div>
                """, unsafe_allow_html=True
            )
            render_near_boundary_badge(ppos)

            # Gauge & legenda
            st.markdown("#### Visualisasi Risiko")
            render_gauge_html(ppos)
            st.markdown(
                """
                <div class="legend">
                    <span><span class="dot dot-green"></span>Hijau = risiko rendah (0–33%)</span>
                    <span>•</span>
                    <span><span class="dot dot-yellow"></span>Kuning = risiko sedang (33–66%)</span>
                    <span>•</span>
                    <span><span class="dot dot-red"></span>Merah = risiko tinggi (&gt;66%)</span>
                </div>
                """, unsafe_allow_html=True
            )

            # Nilai di luar rentang normal
            notes = list_anomalies(one.iloc[0], ranges)
            if notes:
                st.markdown('<div class="card"><h4>🧪 Nilai di luar rentang normal</h4>' +
                            "<ul class='clean'>" + "".join([f"<li>{n}</li>" for n in notes]) +
                            "</ul></div>", unsafe_allow_html=True)
            else:
                st.markdown('<div class="card">✅ Semua nilai berada dalam rentang normal untuk usia/gender ini.</div>',
                            unsafe_allow_html=True)

            # Sensitivitas — TOP 5 fitur paling berpengaruh
            with st.expander("📈 Analisis sensitivitas (opsional) — TOP 5 fitur paling berpengaruh"):
                top_rows, base_prob = tornado_sensitivity(pipeline, one, NUM_COLS, rel=0.2)
                if top_rows:
                    labels = [r[0] for r in top_rows][::-1]
                    deltas = [r[1]*100 for r in top_rows][::-1]
                    fig, ax = plt.subplots(figsize=(6, 3.8))
                    ax.barh(labels, deltas)
                    ax.set_xlabel("Δ Probabilitas positif (p.p.) untuk +20% vs -20%")
                    st.pyplot(fig)
                else:
                    st.caption("Tidak tersedia.")

            # --- Penjelasan yang rapi (sub-list tanpa bullet ganda) ---
            top_rows, _ = tornado_sensitivity(pipeline, one, NUM_COLS, rel=0.2)
            exp_items = explain_risk_structured(ppos, top_rows, notes, one_row=one.iloc[0], ranges=ranges)
            html = ['<div class="card"><h4>🧠 Mengapa hasilnya seperti ini?</h4><ul class="clean">']
            for it in exp_items:
                html.append(f"<li>{md_bold_to_html(it['text'])}</li>")
                if 'sub' in it and it['sub']:
                    html.append("<ul class='sublist'>")
                    for sub in it['sub']:
                        html.append(f"<li>{md_bold_to_html(sub)}</li>")
                    html.append("</ul>")
            html.append("</ul></div>")
            st.markdown("".join(html), unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Gagal memprediksi: {e}")

# -------------------- Menu 3: Model Inspector --------------------
elif menu == "Model Inspector":
    st.subheader("🧩 Model Inspector")

    n_after_scaling = None
    pre = None
    try:
        pre = pipeline.named_steps.get("preprocess", None)
        if pre is not None:
            try:
                n_after_scaling = len(pre.get_feature_names_out())
            except Exception:
                template = {c: np.nan for c in NUM_COLS}
                try:
                    ohe = pre.named_transformers_["cat"].named_steps["ohe"]
                    template["Gender of the patient"] = ohe.categories_[0][0]
                except Exception:
                    template["Gender of the patient"] = "Male"
                Xtmp = pd.DataFrame([template])
                n_after_scaling = pre.transform(Xtmp).shape[1]
    except Exception as e:
        st.info(f"Tidak bisa menghitung fitur pre-PCA: {e}")

    n_after_pca = None
    var_sum = None
    pca = None
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

    # Rincian komponen PCA (explained variance)
    if pca is not None and hasattr(pca, "explained_variance_ratio_"):
        df_pca = pd.DataFrame({
            "PC": [f"PC{i+1}" for i in range(pca.n_components_)],
            "ExplainedVariance": pca.explained_variance_ratio_,
        })
        st.markdown("**Rincian Komponen PCA:**")
        st.dataframe(df_pca, use_container_width=True)

    # ---- Ringkasan PCA (otomatis) ----
    if (pre is not None) and (pca is not None) and hasattr(pca, "components_"):
        st.markdown("**Ringkasan PCA (otomatis)**")
        top_k = st.slider("Tampilkan berapa fitur dominan per PC?", 2, 7, 3, 1, key="pca_topk")
        df_summary = pca_summary(pre, pca, top_k=top_k)
        st.dataframe(df_summary, use_container_width=True)

        with st.expander("Lihat matriks loading lengkap (semua fitur × semua PC)"):
            try:
                feat_names_full = [prettify_feat_name(n) for n in pre.get_feature_names_out()]
            except Exception:
                feat_names_full = [f"f{i}" for i in range(pca.components_.shape[1])]
            loadings_full = pd.DataFrame(
                pca.components_.T, index=feat_names_full,
                columns=[f"PC{i+1}" for i in range(pca.components_.shape[0])]
            )
            st.dataframe(loadings_full, use_container_width=True, height=420)
    else:
        st.info("Preprocess/PCA tidak tersedia atau belum dilatih, sehingga ringkasan PCA tidak bisa ditampilkan.")