# 服務模組

本目錄包含所有服務實現，負責與外部 API 交互和提供核心功能。

## 文件說明

- `__init__.py`: 模組初始化文件
- `api_client.py`: API 客戶端，用於與外部 API 通信
- `llm_service.py`: 大型語言模型服務，支持 OpenAI 和 Ollama
- `face_service.py`: 人臉識別服務，提供人臉檢測和識別功能
- `chatgpt_tts_service.py`: ChatGPT 文字轉語音服務
- `inai_tts_service.py`: Inai 文字轉語音服務
- `tavily_service.py`: Tavily 搜索服務

## 使用方法

這些服務由 main.py 調用，提供系統的核心功能。
