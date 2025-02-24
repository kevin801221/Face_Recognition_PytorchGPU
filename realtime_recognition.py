import os
import cv2
import json
import numpy as np
from datetime import datetime
import time
import math
import torch
import requests
import ollama
import argparse
from dotenv import load_dotenv
from openai import OpenAI
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PySide6.QtWidgets import QApplication
from gui.chat_window import ChatWindow
import sys

# 載入環境變數
load_dotenv()

# 解析命令行參數
parser = argparse.ArgumentParser(description='YCM 智能門禁系統')
parser.add_argument('--model', type=str, default='gpt4o',
                    help='選擇要使用的 LLM 模型 (預設: gpt4o)')
args = parser.parse_args()

# 初始化 LLM
if args.model == 'gpt4o':
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    print("使用 OpenAI GPT-4 模型")
    USE_OPENAI = True
else:
    print(f"使用 Ollama 模型: {args.model}")
    USE_OPENAI = False

# API 設置
API_BASE_URL = "https://inai-hr.jp.ngrok.io/api/employees/search/by-name"

# 初始化 Qt 應用
qt_app = QApplication.instance()
if not qt_app:
    qt_app = QApplication(sys.argv)
chat_window = ChatWindow()
print("聊天窗口已初始化")

# 顯示一條測試消息
chat_window.show_message(f"系統啟動：YCM 館長已準備就緒！(使用 {args.model} 模型)")
print("已發送測試消息")

# 檢查 CUDA 狀態
print("CUDA 狀態檢查:")
print("============")
print("CUDA 是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA 設備:", torch.cuda.get_device_name(0))
    print("CUDA 版本:", torch.version.cuda)
print("============")

# 初始化 InsightFace
face_app = FaceAnalysis(
    name='buffalo_l',
    allowed_modules=['detection', 'recognition'],
    providers=['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
)

# 確保使用 GPU
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    print("正在使用 GPU 設備 ID:", torch.cuda.current_device())

face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(320, 320))

def generate_prompt(employee_data, is_first_chat=True):
    """根據員工資料生成 prompt"""
    if is_first_chat:
        prompt = f"""你現在是 YCM 館長，一個友善、專業的智能助手。你正在與 {employee_data['name']} 進行對話。

你應該：
1. 以 "哈囉 {employee_data['name']}，今天過得還好嗎？我是 YCM 館長" 開始對話
2. 根據以下資訊，從中選擇一個有趣的點來延續對話
3. 保持專業但友善的態度
"""
    else:
        prompt = f"""你現在是 YCM 館長，一個友善、專業的智能助手。你正在與 {employee_data['name']} 進行對話。

你應該：
1. 根據以下資訊，從中選擇一個有趣的點來延續對話
2. 不要重複之前的開場白
3. 直接問一個有趣的問題
4. 保持專業但友善的態度
"""

    prompt += f"""
以下是關於 {employee_data['name']} 的資訊：

基本資料：
- 中文名字：{employee_data['chinese_name']}
- 部門：{employee_data['department']}
- 職位：{employee_data['position']}
- 工作年資：{employee_data['total_years_experience']} 年

專業技能：
{', '.join(employee_data['technical_skills'])}

興趣愛好：
{', '.join(employee_data['interests'])}

證書：
{chr(10).join([f"- {cert['name']} (由 {cert['issuing_organization']} 頒發)" for cert in employee_data['certificates']])}

工作經驗：
{chr(10).join([f"- {exp['company_name']}: {exp['position']} ({exp['description']})" for exp in employee_data['work_experiences']])}
"""
    return prompt

def chat_with_employee(employee_data, is_first_chat=True):
    """使用選定的 LLM 與員工對話"""
    try:
        # 生成初始 prompt
        system_prompt = generate_prompt(employee_data, is_first_chat)
        
        if USE_OPENAI:
            # 使用 OpenAI API
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            return response.choices[0].message.content
        else:
            # 使用 Ollama
            response = ollama.chat(
                model='deepseek-r1:8b',
                messages=[
                    {
                        'role': 'system',
                        'content': system_prompt
                    }
                ]
            )
            return response['message']['content']
            
    except Exception as e:
        print(f"與 LLM 通信時發生錯誤: {e}")
        return None

def get_employee_data(name):
    """從 API 獲取員工資料"""
    try:
        import requests
        
        # 發送 API 請求
        response = requests.get(f"{API_BASE_URL}/{name}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and len(data['data']) > 0:
                return data['data'][0]
        return None
        
    except Exception as e:
        print(f"獲取員工資料時發生錯誤: {e}")
        return None

def cosine_similarity(a, b):
    """計算兩個向量的餘弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def realtime_face_recognition():
    """即時人臉識別主函數"""
    print("啟動即時人臉識別...")
    
    # 載入已知人臉特徵
    with open('face_features.json', 'r', encoding='utf-8') as f:
        known_face_data = json.load(f)
    print(f"已載入人臉特徵，共 {len(known_face_data)} 人")
    
    # 初始化攝像頭並設置更高的分辨率
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        print("無法開啟攝像頭")
        return
        
    # 初始化當前執行期間的檢測記錄
    current_session_log = {}
    current_date = datetime.now().strftime("%Y/%m/%d")
    current_session_log[current_date] = []
    
    # 用於追蹤最近檢測到的人臉和其完整資料
    recent_detections = {}
    employee_cache = {}  # 快取員工資料
    chat_cooldown = {}  # 用於控制對話頻率
    active_conversations = set()  # 追踪正在進行的對話
    
    # 性能優化參數
    frame_skip = 2  # 每隔幾幀處理一次
    frame_count = 0
    process_this_frame = True
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("無法讀取攝像頭畫面")
                break
            
            frame_count += 1
            process_this_frame = frame_count % frame_skip == 0
            
            if process_this_frame:
                # 縮小圖像以加快處理速度
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                
                # 檢測人臉
                faces = face_app.get(small_frame)
                current_time = datetime.now().strftime("%H:%M:%S")
                current_time_obj = datetime.strptime(current_time, "%H:%M:%S")
                
                # 清理舊的檢測記錄
                for person_id in list(recent_detections.keys()):
                    last_detection_time = datetime.strptime(recent_detections[person_id], "%H:%M:%S")
                    time_diff = (current_time_obj - last_detection_time).total_seconds()
                    if time_diff > 5:  # 如果超過5秒沒有檢測到，則移除記錄
                        del recent_detections[person_id]
                        if person_id in active_conversations:
                            active_conversations.remove(person_id)
                
                # 處理每個檢測到的人臉
                for face in faces:
                    # 調整邊界框以匹配原始圖像大小
                    bbox = face.bbox.astype(int) * 2
                    embedding = face.embedding
                    
                    # 使用向量化操作尋找最相似的人臉
                    max_similarity = -1
                    identity = "Unknown"
                    
                    for person_name, features in known_face_data.items():
                        similarities = [cosine_similarity(embedding, feature) for feature in features]
                        max_person_similarity = max(similarities)
                        if max_person_similarity > max_similarity:
                            max_similarity = max_person_similarity
                            identity = person_name
                    
                    # 計算置信度（將相似度轉換為百分比）
                    confidence = (max_similarity + 1) * 50  # 轉換範圍從[-1,1]到[0,100]
                    
                    # 根據置信度決定邊框顏色
                    if confidence > 70:
                        color = (0, 255, 0)  # 綠色
                    elif confidence > 50:
                        color = (0, 255, 255)  # 黃色
                    else:
                        color = (0, 0, 255)  # 紅色
                        identity = "Unknown"
                    
                    # 如果是已知人臉且置信度足夠
                    if identity != "Unknown" and confidence > 60:
                        # 更新檢測時間
                        recent_detections[identity] = current_time
                        
                        # 如果這個人不在活動對話中且已經過了冷卻時間
                        if identity not in active_conversations and (
                            identity not in chat_cooldown or 
                            (current_time_obj - datetime.strptime(chat_cooldown[identity], "%H:%M:%S")).total_seconds() > 120
                        ):
                            # 獲取員工完整資料
                            if identity not in employee_cache:
                                employee_data = get_employee_data(identity)
                                if employee_data:
                                    employee_cache[identity] = employee_data
                                    print(f"已獲取 {identity} 的完整資料")
                            
                            if identity in employee_cache:
                                # 標記為正在對話中
                                active_conversations.add(identity)
                                
                                # 檢查是否是首次對話
                                is_first_chat = identity not in chat_cooldown
                                response = chat_with_employee(employee_cache[identity], is_first_chat)
                                
                                if response:
                                    # 更新對話冷卻時間
                                    chat_cooldown[identity] = current_time
                                    # 顯示回應
                                    chat_window.show_message(f"YCM館長: {response}")
                                    # 完成對話後移除活動對話標記
                                    active_conversations.remove(identity)
                                    
                                    # 記錄到日誌
                                    detection_record = {
                                        "time": current_time,
                                        "identity": identity,
                                        "confidence": confidence,
                                        "response": response
                                    }
                                    current_session_log[current_date].append(detection_record)
                    
                    # 在畫面上標註
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, f"{identity} | {confidence:.1f}%", 
                              (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, color, 2)
            
            # 顯示畫面
            cv2.imshow('Face Recognition', frame)
            
            # 處理 Qt 事件
            qt_app.processEvents()
            
            # 按下 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_face_recognition()
