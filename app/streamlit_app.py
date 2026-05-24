import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🛒",
    layout="wide"
)

# ── Load Model ───────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load('../models/xgboost_churn.pkl')

model = load_model()

# ── Header ───────────────────────────────────────────
st.title("🛒 Customer Churn Prediction")
st.markdown("**E-Commerce Churn Prediction Dashboard** — masukkan data pelanggan untuk memprediksi kemungkinan churn.")
st.divider()

# ── Input Form ───────────────────────────────────────
st.subheader("📋 Data Pelanggan")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=61, value=10)
    warehouse_to_home = st.number_input("Jarak Warehouse ke Rumah (km)", min_value=5, max_value=35, value=15)
    hour_spend_on_app = st.number_input("Jam di App per Hari", min_value=0, max_value=5, value=3)
    number_of_device = st.number_input("Jumlah Device Terdaftar", min_value=1, max_value=6, value=3)
    satisfaction_score = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
    city_tier = st.selectbox("City Tier", [1, 2, 3])

with col2:
    number_of_address = st.number_input("Jumlah Alamat", min_value=1, max_value=10, value=3)
    complain = st.selectbox("Ada Komplain?", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    order_amount_hike = st.number_input("Kenaikan Order dari Tahun Lalu (%)", min_value=11, max_value=26, value=15)
    coupon_used = st.number_input("Kupon Digunakan", min_value=0, max_value=16, value=1)
    order_count = st.number_input("Jumlah Order Bulan Ini", min_value=1, max_value=16, value=2)
    day_since_last_order = st.number_input("Hari Sejak Order Terakhir", min_value=0, max_value=46, value=3)

with col3:
    cashback_amount = st.number_input("Cashback (Rp)", min_value=0, max_value=300, value=160)
    preferred_login = st.selectbox("Login Device", ["Mobile Phone", "Phone", "Computer"],)
    preferred_payment = st.selectbox("Metode Pembayaran", ["Debit Card", "UPI", "CC", "Cash on Delivery", "E wallet", "COD", "Credit Card"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    order_cat = st.selectbox("Kategori Order Favorit", ["Laptop & Accessory", "Mobile", "Mobile Phone", "Fashion", "Grocery", "Others"])
    marital_status = st.selectbox("Status Pernikahan", ["Single", "Married", "Divorced"])

# ── Encoding ─────────────────────────────────────────
login_map = {"Mobile Phone": 1, "Phone": 2, "Computer": 0}
payment_map = {"Debit Card": 4, "UPI": 6, "CC": 0, "Cash on Delivery": 2, "E wallet": 5, "COD": 1, "Credit Card": 3}
gender_map = {"Male": 1, "Female": 0}
order_cat_map = {"Laptop & Accessory": 2, "Mobile": 3, "Mobile Phone": 4, "Fashion": 1, "Grocery": 0, "Others": 5}
marital_map = {"Single": 2, "Married": 1, "Divorced": 0}

input_data = pd.DataFrame([{
    'Tenure': tenure,
    'PreferredLoginDevice': login_map[preferred_login],
    'CityTier': city_tier,
    'WarehouseToHome': warehouse_to_home,
    'PreferredPaymentMode': payment_map[preferred_payment],
    'Gender': gender_map[gender],
    'HourSpendOnApp': hour_spend_on_app,
    'NumberOfDeviceRegistered': number_of_device,
    'PreferedOrderCat': order_cat_map[order_cat],
    'SatisfactionScore': satisfaction_score,
    'MaritalStatus': marital_map[marital_status],
    'NumberOfAddress': number_of_address,
    'Complain': complain,
    'OrderAmountHikeFromlastYear': order_amount_hike,
    'CouponUsed': coupon_used,
    'OrderCount': order_count,
    'DaySinceLastOrder': day_since_last_order,
    'CashbackAmount': cashback_amount
}])

# ── Predict ──────────────────────────────────────────
st.divider()

if st.button("🔍 Prediksi Churn", type="primary", use_container_width=True):
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]

    col_result1, col_result2, col_result3 = st.columns(3)

    with col_result1:
        st.metric("Hasil Prediksi", "🔴 CHURN" if pred == 1 else "🟢 TIDAK CHURN")
    with col_result2:
        st.metric("Probabilitas Churn", f"{prob*100:.1f}%")
    with col_result3:
        risk = "🔴 Tinggi" if prob > 0.7 else ("🟡 Sedang" if prob > 0.4 else "🟢 Rendah")
        st.metric("Tingkat Risiko", risk)

    st.divider()

    # Rekomendasi bisnis
    st.subheader("💡 Rekomendasi")
    if pred == 1:
        st.error("⚠️ Pelanggan ini berisiko churn! Tindakan yang disarankan:")
        if tenure < 3:
            st.write("- 🎁 Berikan **welcome bonus** atau diskon khusus pelanggan baru")
        if complain == 1:
            st.write("- 📞 **Hubungi pelanggan** untuk menyelesaikan komplain segera")
        if cashback_amount < 150:
            st.write("- 💰 Tingkatkan **program cashback** untuk pelanggan ini")
        if day_since_last_order > 5:
            st.write("- 📧 Kirim **email re-engagement** dengan penawaran spesial")
        st.write("- 🎯 Masukkan ke **program loyalitas** prioritas")
    else:
        st.success("✅ Pelanggan ini kemungkinan tidak churn.")
        st.write("- Pertahankan kualitas layanan saat ini")
        st.write("- Tetap monitor jika ada perubahan perilaku belanja")