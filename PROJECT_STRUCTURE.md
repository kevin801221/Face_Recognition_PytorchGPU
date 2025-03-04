# 專案結構說明

本文檔說明 `face_recognition_torch` 專案的目錄結構。

## 目錄結構

```
face_recognition_torch/
├── audio/                # 音頻文件目錄
├── backup/               # 備份文件目錄
├── data/                 # 數據文件目錄
│   ├── face_features.json       # 人臉特徵數據
│   ├── detection_log.json       # 人臉檢測日誌
│   ├── current_session_log.json # 當前會話日誌
│   └── conversation_memory.db   # 對話記憶數據庫
├── docs/                 # 文檔目錄
│   ├── README.md                # 專案說明文檔
│   └── IMPROVEMENTS.md          # 改進建議文檔
├── employee_data/        # 員工數據目錄
├── gui/                  # 圖形界面目錄
│   ├── __init__.py              # 模組初始化文件
│   ├── chat_window.py           # 聊天窗口界面
│   └── face_recognition_gui.py  # 人臉識別界面
├── logs/                 # 日誌目錄
├── models/               # 模型目錄
│   ├── __init__.py              # 模組初始化文件
│   └── face_recognition.py      # 人臉識別模型
├── scripts/              # 腳本目錄
│   ├── list_voices.py           # 列出可用語音
│   ├── train_face.py            # 訓練人臉特徵
│   └── fix_main.py              # 修復主程式
├── services/             # 服務目錄
│   ├── __init__.py              # 模組初始化文件
│   ├── api_client.py            # API 客戶端
│   ├── face_service.py          # 人臉識別服務
│   ├── llm_service.py           # 大型語言模型服務
│   └── ...                      # 其他服務
├── tests/                # 測試目錄
│   ├── test_chatgpt_tts.py      # ChatGPT TTS 測試
│   ├── test_elevenlabs_tts.py   # ElevenLabs TTS 測試
│   └── ...                      # 其他測試
├── training_images/      # 訓練圖像目錄
│   └── [人名]/                  # 每個人的訓練圖像
├── utils/                # 工具目錄
│   ├── __init__.py              # 模組初始化文件
│   ├── conversation_memory.py   # 對話記憶管理
│   ├── face_utils.py            # 人臉處理工具
│   └── ...                      # 其他工具
├── main.py               # 主程式
├── config.py             # 配置文件
└── requirements.txt      # 依賴列表
```

## 目錄說明

- **audio/**: 存儲系統生成的音頻文件
- **backup/**: 存儲重要文件的備份版本
- **data/**: 存儲系統運行所需的數據文件
- **docs/**: 存儲項目文檔和說明
- **employee_data/**: 存儲員工相關數據
- **gui/**: 存儲圖形界面相關代碼
- **logs/**: 存儲系統運行日誌
- **models/**: 存儲模型定義和實現
- **scripts/**: 存儲獨立運行的腳本
- **services/**: 存儲服務實現
- **tests/**: 存儲測試代碼
- **training_images/**: 存儲人臉訓練圖像
- **utils/**: 存儲工具函數和輔助類

## 使用注意

1. 數據文件存放在 `data/` 目錄下
2. 腳本文件在 `scripts/` 目錄下，可以獨立運行
3. 測試文件在 `tests/` 目錄下，用於測試各個組件
4. 備份文件在 `backup/` 目錄下，可以在需要時恢復
