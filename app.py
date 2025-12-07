import streamlit as st
from src.pipeline import main_predict
from src.utils import plot_map
import torch
import geopandas as gpd
import sys, os
from src import config

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(
    page_title="Rainfall Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

@st.cache_resource
def load_css():
    return """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Poppins', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0a1929 0%, #1a2332 25%, #1e3a5f 50%, #2a4a6f 75%, #1a3d5c 100%);
        }
        
        .main-wow {
            font-size: 4rem;
            font-weight: 700;
            text-align: center;
            background: linear-gradient(120deg, #2980b9 0%, #3498db 50%, #5dade2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin: 40px 0 20px 0;
            text-shadow: 0 0 30px rgba(52, 152, 219, 0.5);
            letter-spacing: -2px;
        }
        
        .custom-divider {
            height: 4px;
            background: linear-gradient(90deg, transparent, #3498db, transparent);
            margin: 30px auto 50px auto;
            width: 60%;
            border-radius: 2px;
            box-shadow: 0 0 20px rgba(52, 152, 219, 0.4);
        }
        
        .slider-container {
            background: linear-gradient(135deg, #1c2e3e 0%, #243447 100%);
            padding: 25px;
            border-radius: 20px;
            margin: 15px 0;
            border: 2px solid rgba(52, 152, 219, 0.3);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        
        .slider-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: #5DADE2;
            margin-bottom: 15px;
            text-align: center;
        }
        
        .stButton > button {
            background: linear-gradient(120deg, #2980b9 0%, #3498db 100%);
            color: white;
            font-size: 1.4rem;
            font-weight: 600;
            padding: 20px 60px;
            border-radius: 50px;
            border: none;
            box-shadow: 0 10px 30px rgba(52, 152, 219, 0.4);
            transition: all 0.3s ease;
            width: 100%;
            margin: 30px 0;
            letter-spacing: 1px;
        }
        
        .stButton > button:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(52, 152, 219, 0.6);
            background: linear-gradient(120deg, #3498db 0%, #5dade2 100%);
        }
        
        .model-card {
            background: linear-gradient(135deg, #1c2e3e 0%, #243447 100%);
            padding: 25px;
            border-radius: 20px;
            margin: 15px 0;
            border: 2px solid rgba(52, 152, 219, 0.3);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
        }
        
        .model-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .model-name {
            font-size: 1.5rem;
            font-weight: 600;
            color: #5DADE2;
        }
        
        .model-time {
            font-size: 1.1rem;
            color: #85C1E9;
            font-weight: 500;
        }
        
        .status-text {
            font-size: 1.3rem;
            font-weight: 500;
            color: #5DADE2;
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #1a4d6f 0%, #2874A6 100%);
            border-radius: 15px;
            margin: 20px 0;
            border-left: 5px solid #3498db;
        }
        
        .preview-container {
            margin: 30px 0;
        }
        
        .preview-label {
            font-size: 1.5rem;
            font-weight: 600;
            color: #5DADE2;
            text-align: center;
            margin-bottom: 20px;
        }
        
        .stNumberInput > div > div > input {
            background: #2c3e50;
            border: 2px solid #3498db;
            border-radius: 15px;
            font-size: 1.2rem;
            font-weight: 600;
            color: #00D9FF;
            padding: 12px;
        }
        
        label {
            font-weight: 700;
            color: #00D9FF !important;
            font-size: 1.15rem;
            text-shadow: 0 0 10px rgba(0, 217, 255, 0.3);
        }
        
        .stSelectbox > div > div {
            background: #2c3e50;
            border: 2px solid #3498db;
            border-radius: 15px;
            font-size: 1.1rem;
            color: #FFFFFF;
            font-weight: 500;
        }
        
        div[data-baseweb="select"] span {
            color: #FFFFFF !important;
            font-weight: 500;
        }
        
        div[data-testid="column"] {
            padding: 0 10px;
        }
        
        .stPlotlyChart, .stPyplot {
            background: #1c2e3e;
            padding: 25px;
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            margin-top: 30px;
            border: 2px solid rgba(52, 152, 219, 0.2);
        }
    </style>
    """

st.markdown(load_css(), unsafe_allow_html=True)

st.markdown('<div class="main-wow">Mekong Delta Rainfall Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    vn_adm1 = gpd.read_file(config.GADM_PATH)
    mekong_provinces = [
        "LongAn", "TiềnGiang", "BếnTre", "TràVinh", "VĩnhLong",
        "ĐồngTháp", "AnGiang", "CầnThơ", "HậuGiang",
        "SócTrăng", "BạcLiêu", "CàMau", "KiênGiang"
    ]
    dbscl = vn_adm1[vn_adm1["NAME_1"].isin(mekong_provinces)]

    return vn_adm1, dbscl, device

st.markdown("<br>", unsafe_allow_html=True)

colA, colB = st.columns(2)

with colA:
    st.markdown("""
        <div class="slider-container">
            <div class="slider-title">Stage 1 Model</div>
        </div>
        """, unsafe_allow_html=True)
    
    model_stage1 = st.selectbox(
        "Stage 1 Model",
        ["ConvLSTM", "ConvGRU", "TrajGRU"],
        label_visibility="collapsed"
    )

with colB:
    st.markdown("""
        <div class="slider-container">
            <div class="slider-title">Stage 2 Model</div>
        </div>
        """, unsafe_allow_html=True)
    
    model_stage2 = st.selectbox(
        "Stage 2 Model",
        ["UNet", "FPN", "UNet3Plus"],
        label_visibility="collapsed"
    )

st.markdown("<br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    day = st.number_input("Day", 1, 31, 1)
with col2:
    month = st.number_input("Month", 1, 12, 1)
with col3:
    year = st.number_input("Year", min_value=2025, max_value=2027, value=2025)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("Start Prediction", key="predict_btn"):
    st.markdown("<br>", unsafe_allow_html=True)
    
    status_text = st.empty()
    
    status_text.markdown('<div class="status-text">Loading resources and models...</div>', unsafe_allow_html=True)
    vn_adm1, dbscl, device = load_resources()
    
    status_text.markdown(f'<div class="status-text">Running prediction for {day}/{month}/{year}...</div>', unsafe_allow_html=True)
    pred = main_predict(model_stage1, model_stage2, day, month, year)
    
    status_text.markdown('<div class="status-text">Prediction completed successfully!</div>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class='model-card'>
        <div class='model-header'>
            <span class='model-name'>Rainfall Prediction</span>
            <span class='model-time'>{day}/{month}/{year}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col_center = st.columns([0.5, 2, 0.5])[1]
    with col_center:
        fig = plot_map(
            pred, 
            title=f"Predicted Rainfall ({day}/{month}/{year})",
            vn_adm1=vn_adm1, 
            dbscl=dbscl
        )
        st.pyplot(fig, use_container_width=True)
    
    status_text.empty()
    
    del pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()