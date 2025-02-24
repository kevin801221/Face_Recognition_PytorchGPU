import os
import cv2
import numpy as np
import torch
import insightface
from datetime import datetime
import json
from dotenv import load_dotenv
import requests
import time
import asyncio
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from gui.chat_window import ChatWindow
import sys
import sqlite3
import threading
import math
from utils.speech_input import SpeechRecognizer
import ollama
from openai import OpenAI
from insightface.app import FaceAnalysis
import argparse

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

class ConversationMemory:
    def __init__(self):
        self.conn = sqlite3.connect('conversation_memory.db')
        self.setup_database()
        
    def setup_database(self):
        """創建對話記憶資料庫"""
        cursor = self.conn.cursor()
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT,
            role TEXT,
            message TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.conn.commit()
    
    def add_message(self, person_name, message, role='user'):
        """添加新的對話記錄"""
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO conversations (person_name, role, message) VALUES (?, ?, ?)',
            (person_name, role, message)
        )
        self.conn.commit()
    
    def get_recent_messages(self, person_name, limit=5):
        """獲取最近的對話記錄"""
        cursor = self.conn.cursor()
        cursor.execute(
            'SELECT role, message FROM conversations WHERE person_name = ? ORDER BY timestamp DESC LIMIT ?',
            (person_name, limit)
        )
        return cursor.fetchall()

def generate_prompt(employee_data, recent_messages=None, is_first_chat=True):
    """根據員工資料和對話記錄生成 prompt"""
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
1. 根據用戶的輸入和對話記錄，給出合適的回應
2. 利用員工資料中的資訊來豐富對話
3. 保持專業但友善的態度
"""

    # 添加對話記錄
    if recent_messages:
        prompt += "\n最近的對話記錄：\n"
        for role, message in recent_messages:
            prompt += f"{role}: {message}\n"

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

def handle_user_message(employee_data, user_message, conversation_memory):
    """處理用戶的文字輸入"""
    try:
        # 記錄用戶訊息
        conversation_memory.add_message(employee_data['name'], user_message, 'user')
        
        # 獲取最近的對話記錄
        recent_messages = conversation_memory.get_recent_messages(employee_data['name'])
        
        # 生成回應
        system_prompt = generate_prompt(employee_data, recent_messages, is_first_chat=False)
        
        if USE_OPENAI:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=300
            )
            ai_response = response.choices[0].message.content
        else:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ]
            response = ollama.chat(
                model='deepseek-r1:8b',
                messages=messages
            )
            ai_response = response['message']['content']
        
        # 記錄 AI 回應
        conversation_memory.add_message(employee_data['name'], ai_response, 'assistant')
        
        return ai_response
        
    except Exception as e:
        print(f"處理用戶訊息時發生錯誤: {e}")
        return "抱歉，我現在無法正常回應。請稍後再試。"

def chat_with_employee(employee_data, is_first_chat=True):
    """使用選定的 LLM 與員工對話"""
    try:
        # 生成初始 prompt
        system_prompt = generate_prompt(employee_data, is_first_chat=is_first_chat)
        
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

def cosine_distance(a, b):
    """計算兩個向量的餘弦距離"""
    return 1 - cosine_similarity(a, b)

def realtime_face_recognition():
    """即時人臉識別主函數"""
    print("啟動即時人臉識別...")
    
    # 載入已知人臉特徵
    with open('face_features.json', 'r', encoding='utf-8') as f:
        known_face_data = json.load(f)
    print(f"已載入人臉特徵，共 {len(known_face_data)} 人")
    
    # 初始化對話記憶
    conversation_memory = ConversationMemory()
    
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
    
    # 休眠模式相關變量
    sleep_mode = False
    last_face_position = None
    no_face_counter = 0
    POSITION_THRESHOLD = 50  # 人臉移動超過這個像素值就重新辨識
    NO_FACE_THRESHOLD = 30  # 沒有檢測到人臉的幀數閾值
    
    # 語音識別相關
    speech_recognizer = SpeechRecognizer()
    speech_text_buffer = ""
    last_speech_time = time.time()
    SPEECH_TIMEOUT = 2.0  # 語音輸入超時時間（秒）
    
    def process_speech_input():
        nonlocal speech_text_buffer, last_speech_time
        
        current_time = time.time()
        if speech_text_buffer and (current_time - last_speech_time) >= SPEECH_TIMEOUT:
            # 發送累積的文字
            if current_person and current_person in employee_cache:
                chat_window.input_field.setText(speech_text_buffer)
                chat_window.send_message()
            speech_text_buffer = ""
    
    def on_speech_detected(text):
        nonlocal speech_text_buffer, last_speech_time
        if text:
            speech_text_buffer = text
            last_speech_time = time.time()
            # 更新輸入框顯示
            chat_window.input_field.setText(speech_text_buffer)
    
    # 設置語音識別回調
    speech_recognizer.on_speech_detected = on_speech_detected
    
    # 啟動語音識別線程
    speech_thread = threading.Thread(target=speech_recognizer.start_listening, daemon=True)
    speech_thread.start()
    
    # 設置聊天窗口的訊息處理
    def on_user_message(message):
        nonlocal current_person
        if current_person and current_person in employee_cache:
            response = handle_user_message(
                employee_cache[current_person], 
                message,
                conversation_memory
            )
            chat_window.show_message(response)
        else:
            chat_window.show_message("抱歉，我現在無法確定你是誰。請讓我看清你的臉。")
    
    # 連接聊天窗口的訊息信號
    chat_window.message_sent.connect(on_user_message)
    
    # 主循環
    frame_count = 0
    current_person = None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 每隔幾幀處理一次
        frame_count += 1
        if frame_count % 2 != 0:
            continue
            
        # 檢測人臉
        faces = face_app.get(frame)
        
        # 更新無人臉計數器
        if not faces:
            no_face_counter += 1
            if no_face_counter >= NO_FACE_THRESHOLD:
                sleep_mode = False  # 重置休眠模式
                last_face_position = None
            continue
        else:
            no_face_counter = 0
        
        current_time = datetime.now().strftime("%H:%M:%S")
        current_person = None
        
        # 在休眠模式下只進行位置檢查
        if sleep_mode and faces:
            face = faces[0]
            current_pos = (face.bbox[0], face.bbox[1])
            
            if last_face_position:
                dx = current_pos[0] - last_face_position[0]
                dy = current_pos[1] - last_face_position[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > POSITION_THRESHOLD:
                    sleep_mode = False  # 退出休眠模式
            
            last_face_position = current_pos
            
            # 在休眠模式下使用最後識別的人
            for person_id, last_time in recent_detections.items():
                if (datetime.strptime(current_time, "%H:%M:%S") - 
                    datetime.strptime(last_time, "%H:%M:%S")).total_seconds() < 30:
                    current_person = person_id
                    break
        
        # 非休眠模式下進行完整的人臉辨識
        if not sleep_mode and faces:
            face = faces[0]
            current_pos = (face.bbox[0], face.bbox[1])
            last_face_position = current_pos
            
            # 提取人臉特徵
            face_feature = face.normed_embedding.tolist()
            
            # 尋找最匹配的人臉
            best_match = None
            min_distance = float('inf')
            
            for person_id, features in known_face_data.items():
                for feature in features:
                    distance = cosine_distance(face_feature, feature)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = person_id
            
            # 如果找到匹配的人臉
            if best_match and min_distance < 0.3:
                current_person = best_match
                recent_detections[current_person] = current_time
                
                # 如果這個人還沒有被快取
                if current_person not in employee_cache:
                    try:
                        employee_data = get_employee_data(current_person)
                        if employee_data:
                            employee_cache[current_person] = employee_data
                            
                            # 如果這是新的對話
                            if current_person not in active_conversations:
                                response = chat_with_employee(employee_data, is_first_chat=True)
                                chat_window.show_message(response)
                                active_conversations.add(current_person)
                                
                                # 進入休眠模式
                                sleep_mode = True
                    except Exception as e:
                        print(f"獲取員工資料時發生錯誤: {e}")
        
        # 處理語音輸入
        process_speech_input()
        
        # 顯示框架
        if faces:
            face = faces[0]
            bbox = face.bbox.astype(int)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            
            if current_person:
                cv2.putText(frame, current_person, (bbox[0], bbox[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow('Face Recognition', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_face_recognition()
