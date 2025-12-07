import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GADM_PATH = "models/gadm41_VNM_1.json"
MINMAX_PATH = "models/minmax_scaler_target_train.pkl"
ROBUST_PATH = "models/robust_scaler_target_train.pkl"
class ConvLSTM:
    PATH = "models/convlstm.pth"
    INPUT_CHANNELS = 24
    HIDDEN_DIMS = [64, 32, 16]                  
    KERNEL_SIZE = 3

class ConvGRU:
    PATH = "models/convgru.pth"
    INPUT_CHANNELS = 24
    HIDDEN_DIMS = [64, 32, 16]
    KERNEL_SIZE = 3
    
class TrajGRU:
    PATH = "models/trajgru.pth"
    TIME_OUT = 1
    INPUT_CHANNELS = 24
    HIDDEN_DIMS = [16, 16, 8]
    OUTPUT_CHANNELS = 24
    L = 5
    
class Unet:
    PATH = "models/unet.pth"
    INPUT_CHANNELS = 24
    OUT_CHANNELS = 1
    FEATURES = [32, 64, 128, 256]
    
class FPN:
    PATH = "models/fpn.pth"
    INPUT_CHANNELS = 24
    BASE_CHANNELS = 32
    FPN_CHANNELS = 64
    
class UNet3Plus:
    PATH = "models/unet3plus.pth"
    INPUT_CHANNELS = 24
    OUT_CHANNELS = 1
    BASE_CHANNELS = 8