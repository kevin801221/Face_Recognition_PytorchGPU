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

## 2025/02/26 重構
# YCM 智能門禁系統 - 專案重構

## 專案結構

```
face_recognition_torch/
│
├── main.py                     # 主程式入口點
├── config.py                   # 配置和設置
│
├── models/                     # 模型目錄
│   ├── __init__.py
│   └── face_recognition.py     # 人臉識別模型
│
├── utils/                      # 工具函數目錄
│   ├── __init__.py
│   ├── conversation_memory.py  # 對話記憶資料庫
│   ├── face_utils.py           # 人臉處理工具
│   ├── resource_monitor.py     # 系統資源監控
│   └── speech_input.py         # 語音輸入識別
│
├── services/                   # 服務目錄
│   ├── __init__.py
│   ├── api_client.py           # API 通訊
│   ├── llm_service.py          # LLM 集成 (OpenAI/Ollama)
│   └── face_service.py         # 人臉識別服務
│
└── gui/                        # 圖形介面目錄
    ├── __init__.py
    └── chat_window.py          # 聊天窗口介面
```

## 主要功能模塊

1. **人臉識別核心功能**
   - 使用 InsightFace 實現高精度人臉檢測和識別
   - 支持 GPU 加速處理
   - 特徵提取和比對優化

2. **對話系統**
   - 支持 OpenAI 和 Ollama 兩種 LLM 模型
   - 記憶對話歷史，提供上下文感知回應
   - 支持對話摘要生成

3. **系統優化**
   - 資源監控和自動調整處理頻率
   - 多線程並行處理
   - 緩存機制減少 API 請求

4. **使用者體驗**
   - 圖形化聊天介面
   - 語音輸入支持
   - 豐富的人臉識別顯示

## 使用方法

1. **基本啟動**
   ```bash
   python main.py
   ```

2. **帶參數啟動**
   ```bash
   python main.py --model gpt4o --resolution 720p --cpu-limit 70
   ```

3. **訓練新人臉**
   ```bash
   python -c "from services.face_service import FaceService; FaceService().train_face()"
   ```

## 運行環境要求

- Python 3.8+
- PyTorch 2.0+
- InsightFace
- PySide6
- OpenAI API Key (如使用 OpenAI 模型)
- Ollama (如使用 Ollama 模型)

## 配置項

主要配置參數在 `config.py` 中定義：

- 模型選擇
- CPU 使用限制
- 攝像頭解析度
- 人臉檢測參數
- API 相關設定