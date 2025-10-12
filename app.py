import streamlit as st
from src.pipeline import main_predict
from src.utils import plot_map  # đã chỉnh sửa để trả về figure
import torch
import geopandas as gpd
import sys, os

# Thêm src vào sys.path nếu cần
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

# ================================================================
# ⚙️ Load mô hình, dữ liệu, scaler
# ================================================================
@st.cache_resource
def load_resources():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Bản đồ hành chính Việt Nam
    vn_adm1 = gpd.read_file('model/gadm41_VNM_1.json')
    mekong_provinces = [
        "LongAn", "TiềnGiang", "BếnTre", "TràVinh", "VĩnhLong",
        "ĐồngTháp", "AnGiang", "CầnThơ", "HậuGiang",
        "SócTrăng", "BạcLiêu", "CàMau", "KiênGiang"
    ]
    dbscl = vn_adm1[vn_adm1["NAME_1"].isin(mekong_provinces)]

    return vn_adm1, dbscl, device

# ================================================================
# 🚀 Streamlit App
# ================================================================
st.set_page_config(page_title="Rainfall Prediction ĐBSCL", page_icon="🌧️", layout="centered")

st.title("🌦️ Dự đoán lượng mưa ĐBSCL bằng Mô hình Học sâu")
st.caption("ConvLSTM + U-Net dựa trên dữ liệu ERA5 & GLDAS (2019–2024)")

# Chọn ngày dự đoán
col1, col2, col3 = st.columns(3)
with col1:
    day = st.number_input("Ngày", 1, 31, 1)
with col2:
    month = st.number_input("Tháng", 1, 12, 1)
with col3:
    year = st.number_input("Năm", min_value=2025, max_value=2027, value=2025)

if st.button("🔮 Dự đoán lượng mưa"):
    st.info("⏳ Đang tải mô hình và dự đoán...")

    # Load bản đồ & device
    vn_adm1, dbscl, device = load_resources()

    # Chạy pipeline dự đoán
    pred = main_predict(day, month, year)

    st.success(f"✅ Dự đoán lượng mưa ngày {day}/{month}/{year} thành công!")

    # Vẽ bản đồ lượng mưa
    fig = plot_map(
        pred, 
        title=f"Predicted Rainfall ({day}/{month}/{year})",
        vn_adm1=vn_adm1, 
        dbscl=dbscl
    )
    st.pyplot(fig)  # ✅ truyền figure riêng, không còn cảnh báo deprecated

    st.caption("Dự đoán lượng mưa Đồng bằng Sông Cửu Long bằng mô hình ConvLSTM + UNet")
