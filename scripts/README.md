# 腳本目錄

本目錄包含各種獨立運行的腳本，用於執行特定任務。

## 文件說明

- `list_voices.py`: 列出 ElevenLabs 可用的語音列表
- `train_face.py`: 訓練人臉特徵，用於人臉識別
- `fix_main.py`: 修復主程式的工具腳本

## 使用方法

這些腳本可以獨立運行，例如：

```bash
python scripts/list_voices.py
python scripts/train_face.py
```

### train_face.py

此腳本用於訓練人臉識別系統，會透過攝像頭捕獲人臉圖像並提取特徵。

使用步驟：
1. 運行腳本
2. 輸入姓名
3. 面對攝像頭，按空格鍵捕獲不同角度的人臉（共5張）
4. 系統會自動保存人臉特徵到 `data/face_features.json`

```bash
python scripts/train_face.py
```
