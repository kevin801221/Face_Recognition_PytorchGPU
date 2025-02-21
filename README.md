# PyTorch 版本人臉辨識系統

這是使用 PyTorch 和 InsightFace 實現的即時人臉辨識系統。

## 系統需求

- Python 3.8+
- CUDA（可選，有 GPU 加速會更快）
- 網路攝像頭

## 安裝步驟

1. 建立虛擬環境（建議）：
```bash
python -m venv torch_env
torch_env\Scripts\activate
```

2. 安裝依賴：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 準備訓練數據：
   - 在專案目錄下創建 `training_images` 資料夾
   - 在 `training_images` 中為每個人創建一個子資料夾，資料夾名稱為該人的名字
   - 在每個人的資料夾中放入該人的人臉照片（建議 5-10 張不同角度的照片）

2. 訓練人臉特徵：
```bash
python train_face.py
```

3. 運行即時識別：
```bash
python realtime_recognition.py
```

## 功能特點

- 使用 InsightFace 進行高精度人臉檢測和特徵提取
- 支援 GPU 加速（如果有 CUDA）
- 實時追蹤多個人臉
- 人臉質量評估
- 穩定性評分系統

## 按鍵操作

- 按 'q' 退出程式
