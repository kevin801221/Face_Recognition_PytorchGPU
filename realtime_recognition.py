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

# GPU 加速的特徵比對
def gpu_cosine_similarity(a, b):
    """使用 GPU 加速的餘弦相似度計算"""
    try:
        if torch.cuda.is_available():
            a = torch.tensor(a, dtype=torch.float32).cuda()
            b = torch.tensor(b, dtype=torch.float32).cuda()
            # 計算相似度
            similarity = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
            return float(similarity.cpu().numpy())  # 轉回 CPU 並轉為 Python float
        else:
            return cosine_similarity(a, b)
    except Exception as e:
        print(f"GPU 計算出錯，切換到 CPU: {e}")
        return cosine_similarity(a, b)

def cosine_similarity(a, b):
    """CPU 版本的餘弦相似度計算"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_distance(a, b):
    """計算餘弦距離"""
    return 1 - gpu_cosine_similarity(a, b)

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
    try:
        if is_first_chat:
            prompt = f"""你現在是 YCM 館長，一個友善、專業的智能助手。你正在與 {employee_data.get('name', '訪客')} 進行對話。

你應該：
1. 以簡短的問候開始對話
2. 保持專業但友善的態度
3. 給出簡短的回應，不要太長
"""
        else:
            prompt = f"""你現在是 YCM 館長，一個友善、專業的智能助手。你正在與 {employee_data.get('name', '訪客')} 進行對話。

你應該：
1. 根據用戶的輸入給出合適的回應
2. 保持專業但友善的態度
3. 給出簡短的回應，不要太長
"""

        # 添加對話記錄
        if recent_messages:
            prompt += "\n最近的對話記錄：\n"
            for role, message in recent_messages:
                prompt += f"{role}: {message}\n"

        # 只有當有完整的員工資料時才添加詳細信息
        if all(key in employee_data for key in ['name', 'chinese_name', 'department', 'position']):
            prompt += f"""
以下是關於 {employee_data['name']} 的資訊：

基本資料：
- 中文名字：{employee_data.get('chinese_name', '未提供')}
- 部門：{employee_data.get('department', '未提供')}
- 職位：{employee_data.get('position', '未提供')}
- 工作年資：{employee_data.get('total_years_experience', '未提供')} 年
"""

            if 'technical_skills' in employee_data and employee_data['technical_skills']:
                prompt += f"\n專業技能：\n{', '.join(employee_data['technical_skills'])}"

            if 'interests' in employee_data and employee_data['interests']:
                prompt += f"\n\n興趣愛好：\n{', '.join(employee_data['interests'])}"

            if 'certificates' in employee_data and employee_data['certificates']:
                prompt += "\n\n證書：\n"
                prompt += "\n".join([f"- {cert['name']} (由 {cert['issuing_organization']} 頒發)" 
                                   for cert in employee_data['certificates']])

            if 'work_experiences' in employee_data and employee_data['work_experiences']:
                prompt += "\n\n工作經驗：\n"
                prompt += "\n".join([f"- {exp['company_name']}: {exp['position']} ({exp['description']})" 
                                   for exp in employee_data['work_experiences']])

        print(f"生成的提示詞長度: {len(prompt)}")
        return prompt
    except Exception as e:
        print(f"生成提示詞時發生錯誤: {e}")
        return f"你是 YCM 館長，請友善地與用戶對話。"

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
        print(f"開始與員工對話，使用模型: {'GPT-4' if USE_OPENAI else 'Ollama'}")
        # 生成初始 prompt
        system_prompt = generate_prompt(employee_data, is_first_chat=is_first_chat)
        print(f"生成的系統提示: {system_prompt}")
        
        if USE_OPENAI:
            print("使用 OpenAI API")
            try:
                # 使用 OpenAI API
                response = openai_client.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": system_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=300
                )
                result = response.choices[0].message.content
                print(f"OpenAI 回應: {result}")
                return result
            except Exception as e:
                print(f"OpenAI API 錯誤: {e}")
                return "抱歉，AI 服務暫時無法使用。"
        else:
            print("使用 Ollama")
            try:
                # 使用 Ollama
                response = ollama.chat(
                    model='deepseek-r1:8b',
                    messages=[
                        {"role": "system", "content": system_prompt}
                    ]
                )
                result = response['message']['content']
                print(f"Ollama 回應: {result}")
                return result
            except Exception as e:
                print(f"Ollama 錯誤: {e}")
                return "抱歉，AI 服務暫時無法使用。"
    except Exception as e:
        print(f"對話系統錯誤: {e}")
        return "系統錯誤，請稍後再試。"

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

def realtime_face_recognition():
    """即時人臉識別主函數"""
    print("啟動即時人臉識別...")
    
    # 載入已知人臉特徵
    try:
        with open('face_features.json', 'r', encoding='utf-8') as f:
            known_face_data = json.load(f)
        print(f"已載入人臉特徵，共 {len(known_face_data)} 人")
        print("已知用戶列表:")
        for person_id, features in known_face_data.items():
            print(f"- {person_id}: {len(features)} 個特徵")
    except Exception as e:
        print(f"載入人臉特徵時發生錯誤: {e}")
        return
    
    # 初始化 InsightFace
    providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    face_app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection', 'recognition'],
        providers=providers
    )
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(320, 320))
    
    # 確保使用 GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("正在使用 GPU 設備 ID:", torch.cuda.current_device())
    
    # 初始化對話記憶
    conversation_memory = ConversationMemory()
    
    # 初始化攝像頭
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 使用較低的分辨率
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
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
    active_conversations = set()  # 追踨正在進行的對話
    
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
            # 自動發送累積的文字
            if current_person:
                print(f"自動發送語音輸入: {speech_text_buffer}")
                chat_window.input_field.setText(speech_text_buffer)
                chat_window.send_message()
                speech_text_buffer = ""
            else:
                print("未檢測到用戶，無法發送語音輸入")
    
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
            print(f"提取到人臉特徵，維度: {len(face_feature)}")
            
            # 尋找最匹配的人臉
            best_match = None
            min_distance = float('inf')
            
            # 使用 GPU 加速的特徵比對
            for person_id, features in known_face_data.items():
                for feature in features:
                    distance = cosine_distance(face_feature, feature)
                    print(f"比對 {person_id}: 距離 = {distance}")
                    if distance < min_distance:
                        min_distance = distance
                        best_match = person_id
            
            print(f"最佳匹配: {best_match}, 距離: {min_distance}")
            
            # 如果找到匹配的人臉
            if best_match and min_distance < 0.3:
                print(f"識別到用戶: {best_match}")
                current_person = best_match
                recent_detections[current_person] = current_time
                
                # 如果這個人還沒有被快取
                if current_person not in employee_cache:
                    try:
                        print(f"嘗試獲取員工資料: {current_person}")
                        employee_data = get_employee_data(current_person)
                        if employee_data:
                            print(f"成功獲取員工資料: {employee_data}")
                            employee_cache[current_person] = employee_data
                            
                            # 如果這是新的對話
                            if current_person not in active_conversations:
                                print("開始新對話")
                                response = chat_with_employee(employee_data, is_first_chat=True)
                                print(f"AI 回應: {response}")
                                if response:
                                    chat_window.show_message(response)
                                    print("顯示回應完成")
                                active_conversations.add(current_person)
                    except Exception as e:
                        print(f"獲取員工資料時發生錯誤: {e}")
            else:
                print(f"無法識別用戶，最小距離: {min_distance}")
                current_person = None  # 重置當前用戶
        
        # 在休眠模式下，如果檢測到人臉移動，就退出休眠模式
        elif sleep_mode and faces:
            face = faces[0]
            current_pos = (face.bbox[0], face.bbox[1])
            
            if last_face_position:
                dx = current_pos[0] - last_face_position[0]
                dy = current_pos[1] - last_face_position[1]
                distance = math.sqrt(dx*dx + dy*dy)
                
                if distance > POSITION_THRESHOLD:
                    print("檢測到人臉移動，退出休眠模式")
                    sleep_mode = False  # 退出休眠模式
                    current_person = None  # 重置當前用戶
            
            last_face_position = current_pos
        
        # 如果沒有檢測到人臉，重置狀態
        else:
            current_person = None
            if no_face_counter >= NO_FACE_THRESHOLD:
                sleep_mode = False
                last_face_position = None
        
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
