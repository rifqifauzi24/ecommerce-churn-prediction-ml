# 📌 Project Context — Churn Prediction

## Tentang Saya
- Background: Data Science / Analytics
- Tujuan project: Portofolio untuk cari kerja
- Tools: VS Code + Python + Jupyter Notebook
- OS: macOS Monterey 12.0.1 (Claude Code tidak kompatibel, gunakan Claude.ai)

---

## Project yang Sedang Dikerjakan
**Customer Churn Prediction — E-Commerce**

Membangun model ML untuk memprediksi pelanggan e-commerce yang kemungkinan besar akan berhenti belanja (churn), sehingga bisnis bisa mengambil tindakan retensi lebih awal.

---

## Dataset
- **Nama:** E-Commerce Customer Churn Dataset
- **Sumber:** Kaggle — ankitverma2010/ecommerce-customer-churn-analysis-and-prediction
- **File:** `E Commerce Dataset.xlsx`  ← nama file yang benar
- **Lokasi di project:** `data/raw/E Commerce Dataset.xlsx`
- **Sheet yang dipakai:** `E Comm` (sheet kedua)
- **Sheet lainnya:** `Data Dict` (kamus variabel, tidak dipakai untuk modeling)

### Daftar Kolom Dataset (dari sheet E Comm)
| Kolom | Deskripsi |
|---|---|
| CustomerID | Unique customer ID |
| Churn | Churn Flag (target variable) |
| Tenure | Tenure of customer in organization |
| PreferredLoginDevice | Preferred login device of customer |
| CityTier | City tier |
| WarehouseToHome | Distance between warehouse to home |
| PreferredPaymentMode | Preferred payment method of customer |
| Gender | Gender of customer |
| HourSpendOnApp | Number of hours spend on mobile app or website |
| NumberOfDeviceRegistered | Total number of devices registered |
| PreferedOrderCat | Preferred order category in last month |
| SatisfactionScore | Satisfactory score of customer on service |
| MaritalStatus | Marital status of customer |
| NumberOfAddress | Total number of addresses added |
| Complain | Any complaint raised in last month |
| OrderAmountHikeFromLastYear | Percentage increase in order from last year |
| CouponUsed | Total number of coupons used in last month |
| OrderCount | Total number of orders placed in last month |
| DaySinceLastOrder | Day since last order by customer |
| CashbackAmount | Average cashback in last month |

---

## Struktur Folder Project
```
churn-prediction/
├── data/
│   ├── raw/
│   │   └── E Commerce Dataset.xlsx   ← nama file yang benar
│   └── processed/                    ← data hasil preprocessing
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   ├── 03_Modeling.ipynb
│   └── 04_Explainability.ipynb
├── models/
│   └── xgboost_churn.pkl             ← model tersimpan
├── src/
│   ├── preprocess.py
│   └── predict.py
├── app/
│   └── streamlit_app.py
├── PROJECT_CONTEXT.md                ← file ini
└── README.md
```

---

## Cara Load Dataset
```python
import pandas as pd
df = pd.read_excel('../data/raw/E Commerce Dataset.xlsx', sheet_name='E Comm')
```

---

## Alur Project (Step by Step)
1. ✅ **Business Understanding** — definisi churn, tujuan bisnis
2. 🔄 **EDA** — sedang dikerjakan di `01_EDA.ipynb`
3. ⬜ **Preprocessing** — missing values, encoding, scaling, SMOTE
4. ⬜ **Feature Engineering** — RFM features (recency, frequency, monetary)
5. ⬜ **Modeling** — Logistic Regression, Random Forest, XGBoost/LightGBM
6. ⬜ **Evaluasi** — Recall, Precision, F1, AUC-ROC, Confusion Matrix
7. ⬜ **Explainability** — SHAP values
8. ⬜ **Business Recommendation** — insight actionable dari model
9. ⬜ **Deployment** — simpan model .pkl + Streamlit dashboard

---

## Tech Stack
| Kebutuhan | Library |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisasi | `matplotlib`, `seaborn` |
| Modeling | `scikit-learn`, `xgboost`, `lightgbm` |
| Class imbalance | `imbalanced-learn` (SMOTE) |
| Explainability | `shap` |
| Deployment | `streamlit` |
| Save model | `pickle` / `joblib` |
| Baca Excel | `openpyxl` |

---

## Keputusan Teknis yang Sudah Disepakati
- **Definisi churn:** ikuti kolom `Churn` di dataset (sudah ada labelnya)
- **Metrik utama:** Recall (karena cost of missing churner lebih tinggi)
- **Model utama:** XGBoost (performa terbaik), dibandingkan dengan Logistic Regression (baseline) dan Random Forest
- **Handle imbalance:** SMOTE atau `class_weight='balanced'`
- **Explainability:** SHAP summary plot

---

## Status Terakhir
- [x] Brainstorming ide project selesai
- [x] Dataset dipilih & didownload
- [x] Struktur folder dibuat
- [x] Library diinstall
- [x] Nama file & sheet name dataset dikonfirmasi
- [ ] 01_EDA.ipynb — Cell 1 (import library) ✅, Cell 2 (load data) sedang diperbaiki

---

## Cara Pakai Dokumen Ini
Setiap mulai sesi baru dengan Claude, paste isi dokumen ini dan tambahkan:
> *"Lanjutkan project saya. Status terakhir: [isi status kamu]"*

Claude akan langsung paham konteks tanpa perlu dijelaskan ulang dari awal.