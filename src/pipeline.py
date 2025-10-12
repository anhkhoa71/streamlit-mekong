from src.dataset import FolderTimeSeriesDataset, TensorTimeSeriesDataset
from src.convlstm import ConvLSTMForecaster
from src.unet import UNet
from src.utils import autoregressive_forecast_last, predict_rainfall, plot_map
import joblib, geopandas as gpd, torch
from datetime import datetime
# ================================================================
# 🚀 Main Pipeline
# ================================================================
def main_predict(day, month, year):
    # Load model, dataset, scaler...
    test_dataset = FolderTimeSeriesDataset("data/test")
    test_dataset_last = TensorTimeSeriesDataset("data/test")
    # ----- Thông tin thời gian -----
    start_date = datetime(2024, 12, 31)
    end_date = datetime(year, month, day)
    n_days = (end_date - start_date).days
    print(f"⏳ Số ngày cần dự đoán từ 31/12/2024: {n_days} ngày")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----- Load mô hình UNet -----
    in_channels = test_dataset_last[0][0].shape[0]
    out_channels = test_dataset_last[1][1].shape[0]
    model_unet = UNet(in_channels=in_channels, out_channels=out_channels,
                      features=[32, 64, 128, 256]).to(device)

    checkpoint = torch.load('model/best_unet_model.pth', map_location=device)
    model_unet.load_state_dict(checkpoint['model_state_dict'])

    # ----- Load mô hình ConvLSTM -----
    input_channels = test_dataset[0][0].shape[1]
    conv2d_lstm_model = ConvLSTMForecaster(
        input_channels=input_channels,
        hidden_dims=[64, 32, 16],
        kernel_size=3,
        dropout_rate=0.2
    ).to(device)

    conv_lstm = torch.load("model/conv_lstm_best.pth", map_location="cpu")

    conv2d_lstm_model.load_state_dict(conv_lstm['model_state_dict'])


    # ----- Load scaler -----
    minmax = joblib.load('model/minmax_scaler_target_train.pkl')
    robust = joblib.load('model/robust_scaler_target_train.pkl')


    # ----- Chuẩn bị dữ liệu đầu vào -----
    pre_data, _ = test_dataset[-1]
    pre_data = pre_data.unsqueeze(0)  # [1, time_in, C, H, W]

    features = autoregressive_forecast_last(
        conv2d_lstm_model, pre_data, n_days=n_days, device=device
    )

    features = torch.from_numpy(features).unsqueeze(0).float().to(device)

    # ----- Dự đoán lượng mưa -----
    pred = predict_rainfall(model_unet, features, minmax, robust, device=device)
    return pred