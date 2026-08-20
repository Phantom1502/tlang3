import pandas as pd
import numpy as np
import os
import glob

class Preprocess:
    @staticmethod
    def calculate_atr(df: pd.DataFrame, period=100) -> pd.Series:
        """Tính chỉ báo ATR chuẩn kỹ thuật (Wilder's Smoothing)"""
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift(1))
        low_cp = np.abs(df['Low'] - df['Close'].shift(1))
        
        # True Range
        tr = np.maximum(high_low, np.maximum(high_cp, low_cp))
        
        # ATR chuẩn dùng Wilder's Smoothing (alpha = 1 / period)
        atr = tr.ewm(alpha=1/period, adjust=False).mean()
        return atr
    
    @staticmethod
    def preprocess(csv_path: str, output_path: str, window: int = 100, atr_period: int = 100) -> None:
        df = pd.read_csv(csv_path)
        df = df[['Datetime', 'Open', 'High', 'Low', 'Close']]
        
        # 1. Tính ATR
        df['ATR_100'] = Preprocess.calculate_atr(df, atr_period)
        df = df.dropna().reset_index(drop=True)
        
        # 2. Vectorization: Dùng rolling() thay cho vòng lặp for i in range(...)
        # High max và Low min trong window=100 nến
        rolling_high = df['High'].rolling(window=window, min_periods=1).max()
        rolling_low = df['Low'].rolling(window=window, min_periods=1).min()
        
        # Range với 20% padding
        window_range = (rolling_high - rolling_low)
        
        # Tính tỉ lệ ratio
        ratios = window_range / df['ATR_100']
        
        # Bỏ dòng đầu tiên (nếu cần khớp với range(1, len(df)) cũ)
        all_max_ratios = ratios.iloc[1:].values
        
        # 3. Tính FINAL_SCALE_FACTOR
        FINAL_SCALE_FACTOR = np.percentile(all_max_ratios, 99.9)
        
        # 4. Xuất file CSV
        os.makedirs(output_path, exist_ok=True)
        base_name = os.path.basename(csv_path)
        filename = os.path.splitext(base_name)[0]
        output_file = os.path.join(output_path, f"{filename}.csv")
        
        df.to_csv(output_file, index=False)
        
        # 5. Ghi scale factor
        scale_factor_file = os.path.join(output_path, "scale_factor.txt")
        with open(scale_factor_file, 'a') as f:
            f.write(f"{filename}: {FINAL_SCALE_FACTOR:.2f}\n")
            
def preprocess_folder(
    folder_path: str, 
    output_path: str,
    window: int = 100, 
    atr_period: int = 100
):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    for csv_file in csv_files:
        Preprocess.preprocess(csv_file, output_path, window, atr_period)
        
if __name__ == "__main__":
    from app.config import AppConfig, load_config
    cfg: AppConfig = load_config("configs")
    # preprocess train data
    preprocess_folder(
        "./data/raw/train", 
        "./data/preprocessed/train",
        window=cfg.window.input_candles,
        atr_period=100
    )
    # preprocess val data
    preprocess_folder(
        "./data/raw/val", 
        "./data/preprocessed/val",
        window=cfg.window.input_candles,
        atr_period=100
    )