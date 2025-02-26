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
import psutil
from queue import Queue
# 載入環境變數
load_dotenv()

# 解析命令行參數
parser = argparse.ArgumentParser(description='YCM 智能門禁系統')
parser.add_argument('--model', type=str, default='gpt4o',
                    help='選擇要使用的 LLM 模型 (預設: gpt4o)')
parser.add_argument('--cpu-limit', type=float, default=80.0,
                    help='CPU 使用率限制 (預設: 80%)')
parser.add_argument('--resolution', type=str, default='480p',
                    help='攝像頭解析度 (360p, 480p, 720p)')
parser.add_argument('--skip-frames', type=int, default=2,
                    help='跳過幀數量 (預設: 每3幀處理1幀)')
args = parser.parse_args()

# 根據解析度參數設置攝像頭大小
RESOLUTION_MAP = {
    '360p': (640, 360),
    '480p': (640, 480),
    '720p': (1280, 720)
}
CAMERA_WIDTH, CAMERA_HEIGHT = RESOLUTION_MAP.get(args.resolution, (640, 480))

# 初始化 LLM
if args.model == 'gpt4o':
    openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    print("使用 OpenAI GPT-4o 模型")
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

# 資源監視器類
class ResourceMonitor:
    """監控和管理系統資源使用"""
    def __init__(self, target_cpu_percent=70.0):
        self.target_cpu_percent = target_cpu_percent
        self.current_cpu_percent = 0.0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """開始監控線程"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """監控CPU使用率的循環"""
        while self.monitoring:
            self.current_cpu_percent = psutil.cpu_percent(interval=1.0)
            time.sleep(0.5)  # 更新頻率
            
    def get_processing_delay(self):
        """根據CPU使用率計算處理延遲"""
        # 當CPU使用率超過目標時，增加延遲
        if self.current_cpu_percent > self.target_cpu_percent:
            # 越接近 100%，延遲越多
            excess = self.current_cpu_percent - self.target_cpu_percent
            # 將過量轉換為 0-1 範圍
            factor = min(excess / (100.0 - self.target_cpu_percent), 1.0)
            # 延遲範圍從 0 到 0.5 秒
            return factor * 0.5
        return 0.0
        
    def get_frame_skip_rate(self, base_skip=2):
        """計算應跳過的幀數"""
        # 基本跳過率(base_skip)通常為2，表示處理每3幀數據
        
        if self.current_cpu_percent > 90.0:
            # CPU 負載非常高，大幅增加跳過率
            return base_skip + 5  # 處理每 8 幀
        elif self.current_cpu_percent > 80.0:
            # CPU 負載很高，增加跳過率
            return base_skip + 3  # 處理每 6 幀
        elif self.current_cpu_percent > 70.0:
            return base_skip + 2  # 處理每 5 幀
        elif self.current_cpu_percent > 60.0:
            return base_skip + 1  # 處理每 4 幀
        return base_skip  # 默認處理每 3 幀
        
    def should_process_frame(self, frame_count):
        """決定是否處理當前幀"""
        skip_rate = self.get_frame_skip_rate()
        return frame_count % (skip_rate + 1) == 0
        
    def __str__(self):
        return f"CPU: {self.current_cpu_percent:.1f}% | Target: {self.target_cpu_percent}%"

# GPU 加速的特徵比對
def gpu_cosine_similarity(a, b):
    """使用 GPU 加速的餘弦相似度計算"""
    try:
        if torch.cuda.is_available():
            # 批量處理以減少 CPU-GPU 數據傳輸
            if isinstance(a, list) and isinstance(b, list):
                # 如果輸入是多個特徵向量，一次性處理
                a_tensor = torch.tensor(a, dtype=torch.float32).cuda()
                b_tensor = torch.tensor(b, dtype=torch.float32).cuda()
                
                # 計算歸一化
                a_norm = torch.norm(a_tensor, dim=1, keepdim=True)
                b_norm = torch.norm(b_tensor, dim=1, keepdim=True)
                
                # 計算相似度矩陣
                similarity = torch.matmul(a_tensor / a_norm, (b_tensor / b_norm).t())
                return similarity.cpu().numpy()
            else:
                # 單個向量處理
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

# 優化的批量特徵比對
def batch_feature_matching(query_feature, known_features_dict, top_k=3):
    """批量計算特徵相似度並返回最佳匹配"""
    start_time = time.time()
    
    # 準備批量處理數據
    all_features = []
    feature_mapping = []  # 用於追踪特徵所屬的人
    
    for person_id, features in known_features_dict.items():
        for feature in features:
            all_features.append(feature)
            feature_mapping.append(person_id)
    
    # 檢查是否有特徵可比對
    if not all_features:
        return None, 1.0
    
    # 使用GPU批量計算相似度
    if torch.cuda.is_available():
        # 轉換為張量
        query_tensor = torch.tensor([query_feature], dtype=torch.float32).cuda()
        features_tensor = torch.tensor(all_features, dtype=torch.float32).cuda()
        
        # 計算歸一化
        query_norm = torch.norm(query_tensor, dim=1, keepdim=True)
        features_norm = torch.norm(features_tensor, dim=1, keepdim=True)
        
        # 計算相似度
        similarity = torch.matmul(
            query_tensor / query_norm, 
            (features_tensor / features_norm).t()
        )
        
        # 獲取最佳匹配
        similarity_np = similarity.cpu().numpy()[0]
        
        # 找出前k個最大值的索引
        top_indices = np.argsort(similarity_np)[-top_k:][::-1]
        
        # 獲取結果
        results = []
        for idx in top_indices:
            person_id = feature_mapping[idx]
            distance = 1.0 - similarity_np[idx]  # 轉換為距離
            results.append((person_id, distance))
        
        # 找出距離最小的結果
        best_match = min(results, key=lambda x: x[1])
        
        print(f"批量特徵比對耗時: {time.time() - start_time:.4f}秒")
        return best_match
    else:
        # CPU 版本的比對
        min_distance = float('inf')
        best_match = None
        
        for person_id, features in known_features_dict.items():
            for feature in features:
                distance = cosine_distance(query_feature, feature)
                if distance < min_distance:
                    min_distance = distance
                    best_match = person_id
        
        print(f"CPU 特徵比對耗時: {time.time() - start_time:.4f}秒")
        return (best_match, min_distance) if best_match else (None, 1.0)

class EnhancedConversationMemory:
    """增強版對話記憶系統，改善上下文處理"""
    def __init__(self):
        self.conn = sqlite3.connect('conversation_memory.db')
        self.setup_database()
        # 快取機制
        self.message_cache = {}  # 用戶ID -> 最近消息列表
        self.cache_size = 20     # 每個用戶快取的消息數量
        
    def setup_database(self):
        """創建或升級對話記憶資料庫"""
        cursor = self.conn.cursor()
        
        # 檢查現有表結構
        cursor.execute("PRAGMA table_info(conversations)")
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        
        # 如果表不存在，創建新表
        if not columns:
            cursor.execute('''
            CREATE TABLE conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT,
                role TEXT,
                message TEXT,
                importance REAL DEFAULT 1.0,
                conversation_id TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            print("創建新的對話記憶表")
        else:
            # 檢查是否需要添加新列
            if 'importance' not in column_names:
                cursor.execute('ALTER TABLE conversations ADD COLUMN importance REAL DEFAULT 1.0')
                print("添加 importance 列")
                
            if 'conversation_id' not in column_names:
                cursor.execute('ALTER TABLE conversations ADD COLUMN conversation_id TEXT')
                print("添加 conversation_id 列")
                
                # 更新現有記錄的 conversation_id
                cursor.execute('''
                UPDATE conversations 
                SET conversation_id = person_name || '_' || strftime('%Y%m%d', timestamp)
                WHERE conversation_id IS NULL
                ''')
                print("更新現有記錄的 conversation_id")
        
        # 檢查是否需要創建索引
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_person_name'")
        if not cursor.fetchone():
            cursor.execute('CREATE INDEX idx_person_name ON conversations(person_name)')
            print("創建 person_name 索引")
            
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_conv_id'")
        if not cursor.fetchone():
            cursor.execute('CREATE INDEX idx_conv_id ON conversations(conversation_id)')
            print("創建 conversation_id 索引")
        
        # 檢查摘要表是否存在
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_summaries'")
        if not cursor.fetchone():
            cursor.execute('''
            CREATE TABLE conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT,
                summary TEXT,
                start_time DATETIME,
                end_time DATETIME,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            print("創建對話摘要表")
        
        self.conn.commit()
    
    def add_message(self, person_name, message, role='user', importance=1.0):
        """添加新的對話記錄，自動管理快取"""
        # 生成對話ID
        current_date = datetime.now().strftime("%Y%m%d")
        conversation_id = f"{person_name}_{current_date}"
        
        cursor = self.conn.cursor()
        cursor.execute(
            'INSERT INTO conversations (person_name, role, message, importance, conversation_id) VALUES (?, ?, ?, ?, ?)',
            (person_name, role, message, importance, conversation_id)
        )
        self.conn.commit()
        
        # 更新快取
        if person_name not in self.message_cache:
            self.message_cache[person_name] = []
        
        self.message_cache[person_name].append({
            'role': role,
            'message': message,
            'timestamp': datetime.now(),
            'importance': importance
        })
        
        # 維護快取大小
        if len(self.message_cache[person_name]) > self.cache_size:
            self.message_cache[person_name].pop(0)
    
    def calculate_message_importance(self, message):
        """計算消息重要性分數"""
        importance = 1.0  # 默認分數
        
        # 根據消息長度增加重要性
        if len(message) > 100:
            importance += 0.2
        
        # 關鍵詞檢測
        important_keywords = ["記住", "重要", "不要忘記", "請注意", "記得"]
        if any(keyword in message for keyword in important_keywords):
            importance += 0.5
            
        # 問題通常更重要
        if "?" in message or "？" in message:
            importance += 0.3
            
        return min(importance, 2.0)  # 上限為2.0
    
    def get_recent_messages(self, person_name, limit=8):
        """獲取最近的對話記錄，優先使用快取"""
        # 先檢查快取
        if person_name in self.message_cache and len(self.message_cache[person_name]) >= limit:
            # 直接從快取返回
            recent = self.message_cache[person_name][-limit:]
            return [(msg['message'], msg['role']) for msg in recent]
        
        # 快取不足，從數據庫查詢
        cursor = self.conn.cursor()
        cursor.execute(
            '''
            SELECT message, role FROM conversations 
            WHERE person_name = ? 
            ORDER BY timestamp DESC LIMIT ?
            ''',
            (person_name, limit)
        )
        result = cursor.fetchall()
        
        # 如果消息較少，嘗試添加對話摘要
        if len(result) < limit // 2:
            # 檢查摘要表是否存在
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_summaries'")
            if cursor.fetchone():
                # 獲取之前的對話摘要
                cursor.execute(
                    '''
                    SELECT summary FROM conversation_summaries
                    WHERE person_name = ?
                    ORDER BY timestamp DESC LIMIT 1
                    ''',
                    (person_name,)
                )
                summary = cursor.fetchone()
                if summary:
                    # 將摘要添加為系統消息
                    result.append((f"前次對話摘要: {summary[0]}", "system"))
                
        # 返回結果，注意返回順序從舊到新
        return [(msg[0], msg[1]) for msg in reversed(result)]
    
    def generate_conversation_summary(self, person_name, ai_client=None):
        """生成對話摘要並存儲"""
        # 檢查摘要表是否存在
        cursor = self.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='conversation_summaries'")
        if not cursor.fetchone():
            print("摘要表不存在，創建中...")
            cursor.execute('''
            CREATE TABLE conversation_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_name TEXT,
                summary TEXT,
                start_time DATETIME,
                end_time DATETIME,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            ''')
            self.conn.commit()
        
        # 獲取需要摘要的對話
        cursor.execute(
            '''
            SELECT message, role FROM conversations 
            WHERE person_name = ?
            ORDER BY timestamp DESC LIMIT 20
            ''', 
            (person_name,)
        )
        messages = cursor.fetchall()
        
        if not messages or len(messages) < 5:
            return None  # 消息太少，不生成摘要
            
        # 構建對話文本
        conversation_text = "\n".join([f"{role}: {message}" for message, role in messages])
        
        try:
            if not ai_client:  # 如果沒有提供AI客戶端，使用簡單規則摘要
                # 提取關鍵詞和句子的簡單摘要
                important_sentences = []
                for message, role in messages:
                    if role == "user" and len(message) > 20:
                        important_sentences.append(message)
                
                summary = "對話涉及: " + ", ".join(important_sentences[:3])
            else:
                # 使用AI生成摘要
                if USE_OPENAI and isinstance(ai_client, OpenAI):
                    response = ai_client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "請為以下對話生成簡短摘要，提取關鍵資訊。"},
                            {"role": "user", "content": conversation_text}
                        ],
                        max_tokens=100
                    )
                    summary = response.choices[0].message.content
                else:
                    # 使用 Ollama
                    response = ollama.chat(
                        model='deepseek-r1:8b',
                        messages=[
                            {"role": "system", "content": "請為以下對話生成簡短摘要，提取關鍵資訊。"},
                            {"role": "user", "content": conversation_text}
                        ]
                    )
                    summary = response['message']['content']
            
            # 存儲摘要
            current_time = datetime.now()
            cursor.execute(
                '''
                INSERT INTO conversation_summaries 
                (person_name, summary, start_time, end_time) 
                VALUES (?, ?, ?, ?)
                ''',
                (person_name, summary, current_time, current_time)
            )
            self.conn.commit()
            
            print(f"已生成並存儲對話摘要: {summary}")
            return summary
            
        except Exception as e:
            print(f"生成摘要時出錯: {e}")
            return None

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
            for message, role in recent_messages:
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
        # 計算消息重要性
        importance = conversation_memory.calculate_message_importance(user_message)
        
        # 記錄用戶訊息
        conversation_memory.add_message(
            employee_data['name'], 
            user_message, 
            'user',
            importance
        )
        
        # 獲取最近的對話記錄
        recent_messages = conversation_memory.get_recent_messages(employee_data['name'])
        
        # 生成回應
        system_prompt = generate_prompt(employee_data, recent_messages, is_first_chat=False)
        
        start_time = time.time()
        if USE_OPENAI:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
            response = openai_client.chat.completions.create(
                model="gpt-4o",
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
        
        print(f"AI 回應耗時: {time.time() - start_time:.2f}秒")
        
        # 記錄 AI 回應
        conversation_memory.add_message(employee_data['name'], ai_response, 'assistant')
        
        # 如果對話較長，生成摘要
        if len(recent_messages) >= 10:
            threading.Thread(
                target=conversation_memory.generate_conversation_summary,
                args=(employee_data['name'], openai_client if USE_OPENAI else None),
                daemon=True
            ).start()
        
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
        
        if USE_OPENAI:
            print("使用 OpenAI API")
            try:
                # 使用 OpenAI API
                response = openai_client.chat.completions.create(
                    model="gpt-4o",
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

# 員工資料緩存，避免頻繁 API 調用
employee_cache = {}

def get_employee_data(name, force_refresh=False):
    """從 API 獲取員工資料，帶緩存"""
    # 檢查緩存
    if not force_refresh and name in employee_cache:
        print(f"從緩存獲取 {name} 的資料")
        return employee_cache[name]
    
    try:
        import requests
        
        # 發送 API 請求
        response = requests.get(f"{API_BASE_URL}/{name}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and len(data['data']) > 0:
                employee_data = data['data'][0]
                # 更新緩存
                employee_cache[name] = employee_data
                return employee_data
        return None
        
    except Exception as e:
        print(f"獲取員工資料時發生錯誤: {e}")
        return None

def realtime_face_recognition():
    """即時人臉識別主函數 - 優化版本"""
    print("啟動即時人臉識別...")
    
    # 初始化資源監視器
    resource_monitor = ResourceMonitor(target_cpu_percent=args.cpu_limit)
    resource_monitor.start_monitoring()
    
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
    
    # 使用較小的檢測尺寸
    det_size = (160, 160)  # 原來是 (320, 320)
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=det_size)
    
    # 確保使用 GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        print("正在使用 GPU 設備 ID:", torch.cuda.current_device())
    
    # 初始化增強版對話記憶
    conversation_memory = EnhancedConversationMemory()
    
    # 初始化攝像頭
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 減少緩衝以降低延遲
    
    if not cap.isOpened():
        print("無法開啟攝像頭")
        return
    
    # 初始化當前執行期間的檢測記錄
    current_session_log = {}
    current_date = datetime.now().strftime("%Y/%m/%d")
    current_session_log[current_date] = []
    
    # 用於追蹤最近檢測到的人臉和其完整資料
    recent_detections = {}
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
    
    # 性能監控變量
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
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
    
    # 批量處理特徵比對的線程安全隊列
    feature_queue = Queue()
    result_queue = Queue()
    
    def feature_matching_worker():
        """特徵比對工作線程"""
        while True:
            try:
                if not feature_queue.empty():
                    # 獲取要處理的特徵
                    face_feature = feature_queue.get()
                    
                    # 執行批量比對
                    best_match, min_distance = batch_feature_matching(face_feature, known_face_data)
                    
                    # 放入結果隊列
                    result_queue.put((best_match, min_distance))
                else:
                    # 無任務時短暫休眠以減少 CPU 使用
                    time.sleep(0.01)
            except Exception as e:
                print(f"特徵比對錯誤: {e}")
                time.sleep(0.1)
    
    # 啟動特徵比對工作線程
    feature_thread = threading.Thread(target=feature_matching_worker, daemon=True)
    feature_thread.start()
    
    # 主循環
    frame_count = 0
    current_person = None
    
    # 運動檢測參數
    prev_gray = None
    motion_threshold = 5.0
    motion_detected = False
    
    print("即時人臉識別系統已啟動...")
    
    while True:
        try:
            loop_start = time.time()  # 測量每次循環時間
            
            ret, frame = cap.read()
            if not ret:
                print("無法讀取影像")
                time.sleep(0.1)
                continue
            
            # 更新 FPS 計數
            fps_counter += 1
            if time.time() - fps_start_time >= 5:  # 每5秒更新一次FPS
                current_fps = fps_counter / (time.time() - fps_start_time)
                print(f"目前 FPS: {current_fps:.1f}, CPU 使用率: {resource_monitor.current_cpu_percent:.1f}%")
                fps_counter = 0
                fps_start_time = time.time()
            
            # 幀計數增加
            frame_count += 1
            
            # 根據 CPU 使用率決定是否跳過處理
            if not resource_monitor.should_process_frame(frame_count):
                # 只顯示影像，不處理
                if current_person:
                    # 如果之前已識別到用戶，顯示姓名
                    cv2.putText(frame, f"Identified: {current_person}", (bbox[0], bbox[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # 簡單的移動檢測
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if prev_gray is not None:
                # 計算幀差
                frame_diff = cv2.absdiff(gray, prev_gray)
                motion_score = np.mean(frame_diff)
                motion_detected = motion_score > motion_threshold
            
            prev_gray = gray.copy()
            
            # 只有在檢測到運動或定期處理時才進行人臉檢測
            if motion_detected or frame_count % 15 == 0:
                # 檢測人臉
                faces = face_app.get(frame)
            else:
                faces = []
            
            # 更新無人臉計數器
            if not faces:
                no_face_counter += 1
                if no_face_counter >= NO_FACE_THRESHOLD:
                    sleep_mode = False  # 重置休眠模式
                    last_face_position = None
                    current_person = None
                
                # 繼續顯示
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            else:
                no_face_counter = 0
            
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # 在休眠模式下只進行位置檢查
            if sleep_mode and faces:
                face = faces[0]
                current_pos = (face.bbox[0], face.bbox[1])
                
                if last_face_position:
                    dx = current_pos[0] - last_face_position[0]
                    dy = current_pos[1] - last_face_position[1]
                    distance = math.sqrt(dx*dx + dy*dy)
                    
                    if distance > POSITION_THRESHOLD:
                        print("檢測到顯著移動，退出休眠模式")
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
                
                # 放入特徵比對隊列
                if feature_queue.qsize() < 5:  # 限制隊列大小
                    feature_queue.put(face_feature)
                
                # 檢查是否有比對結果
                if not result_queue.empty():
                    best_match, min_distance = result_queue.get()
                    
                    # 如果找到匹配的人臉
                    if best_match and min_distance < 0.3:
                        print(f"識別到用戶: {best_match}, 距離: {min_distance:.4f}")
                        current_person = best_match
                        recent_detections[current_person] = current_time
                        
                        # 如果這個人還沒有被快取
                        if current_person not in employee_cache:
                            try:
                                print(f"嘗試獲取員工資料: {current_person}")
                                employee_data = get_employee_data(current_person)
                                if employee_data:
                                    print(f"成功獲取員工資料: {employee_data['name']}")
                                    
                                    # 如果這是新的對話
                                    if current_person not in active_conversations:
                                        print("開始新對話")
                                        # 啟動單獨線程進行AI回應，避免阻塞主循環
                                        def start_conversation():
                                            response = chat_with_employee(employee_data, is_first_chat=True)
                                            if response:
                                                chat_window.show_message(response)
                                                print("顯示回應完成")
                                            
                                        threading.Thread(target=start_conversation, daemon=True).start()
                                        active_conversations.add(current_person)
                            except Exception as e:
                                print(f"獲取員工資料時發生錯誤: {e}")
                        
                        # 每10分鐘檢查是否需要生成對話摘要
                        if current_person in active_conversations:
                            current_hour = datetime.now().hour
                            current_minute = datetime.now().minute
                            if current_minute % 10 == 0 and current_minute != 0:
                                # 執行對話摘要生成
                                threading.Thread(
                                    target=conversation_memory.generate_conversation_summary,
                                    args=(current_person, openai_client if USE_OPENAI else None),
                                    daemon=True
                                ).start()
                    else:
                        print(f"無法識別用戶，最小距離: {min_distance:.4f}")
                        # 僅當距離非常大時才重置當前用戶
                        if min_distance > 0.6:
                            current_person = None
            
            # 處理語音輸入
            process_speech_input()
            
            # 顯示框架
            if faces:
                face = faces[0]
                bbox = face.bbox.astype(int)
                # 根據識別結果選擇顏色
                if current_person:
                    color = (0, 255, 0)  # 綠色 - 已識別
                else:
                    color = (0, 165, 255)  # 橙色 - 未識別
                    
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                if current_person:
                    # 在人臉上方顯示姓名
                    cv2.putText(frame, current_person, (bbox[0], bbox[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
                    # 在畫面角落顯示更多信息
                    cv2.putText(frame, f"CPU: {resource_monitor.current_cpu_percent:.1f}%", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # 計算並顯示處理時間
            process_time = time.time() - loop_start
            if process_time > 0.1:  # 如果處理時間過長，顯示警告
                print(f"警告: 處理時間較長 {process_time:.3f}秒")
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # 釋放 CPU，避免 100% 佔用
            if process_time < 0.03:  # 如果處理很快
                time.sleep(0.01)  # 短暫休眠
                
        except Exception as e:
            print(f"主循環錯誤: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)  # 出錯時短暫休眠
    
    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()
    resource_monitor.stop_monitoring()
    speech_recognizer.stop_listening()
    
    print("即時人臉識別系統已關閉")

def train_face(name=None):
    """訓練人臉特徵"""
    if name is None:
        name = input("請輸入姓名: ")
    
    print(f"開始訓練 {name} 的人臉特徵...")
    
    # 初始化 InsightFace
    providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
    face_app = FaceAnalysis(
        name='buffalo_l',
        allowed_modules=['detection', 'recognition'],
        providers=providers
    )
    face_app.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(320, 320))
    
    # 創建訓練圖像目錄
    os.makedirs("training_images", exist_ok=True)
    person_dir = os.path.join("training_images", name)
    os.makedirs(person_dir, exist_ok=True)
    
    # 初始化攝像頭
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("無法開啟攝像頭")
        return
    
    # 收集人臉圖像和特徵
    features = []
    image_count = 0
    max_images = 5
    
    print(f"請面對攝像頭，將收集 {max_images} 張不同角度的人臉圖像...")
    
    while image_count < max_images:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # 檢測人臉
        faces = face_app.get(frame)
        
        # 顯示提示信息
        cv2.putText(frame, f"收集圖像: {image_count}/{max_images}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, "按空格鍵捕獲人臉", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 繪製檢測到的人臉
        if faces:
            for face in faces:
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        cv2.imshow("人臉訓練", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):  # 空格鍵
            if faces:
                face = faces[0]  # 使用第一個檢測到的人臉
                
                # 保存人臉圖像
                face_image = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                image_path = os.path.join(person_dir, f"{name}_{image_count}.jpg")
                cv2.imwrite(image_path, face_image)
                
                # 獲取人臉特徵
                feature = face.normed_embedding.tolist()
                features.append(feature)
                
                print(f"捕獲第 {image_count+1} 張人臉圖像")
                image_count += 1
                
                # 等待一段時間，避免連續捕獲相同角度
                time.sleep(1)
            else:
                print("未檢測到人臉，請調整位置")
        
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 保存特徵到 JSON 文件
    try:
        if os.path.exists('face_features.json'):
            with open('face_features.json', 'r', encoding='utf-8') as f:
                face_features = json.load(f)
        else:
            face_features = {}
        
        face_features[name] = features
        
        with open('face_features.json', 'w', encoding='utf-8') as f:
            json.dump(face_features, f, ensure_ascii=False, indent=2)
        
        print(f"已成功保存 {name} 的 {len(features)} 個人臉特徵")
    except Exception as e:
        print(f"保存人臉特徵時發生錯誤: {e}")

if __name__ == "__main__":
    realtime_face_recognition()