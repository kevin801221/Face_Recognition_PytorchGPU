# import sqlite3
# from typing import List, Tuple, Optional
# import json

# class EnhancedConversationMemory:
#     """增強版對話記憶系統，改善上下文處理"""
#     def __init__(self):
#         self.conn = sqlite3.connect('conversation_memory.db')
#         self.setup_database()
#         # 快取機制
#         self.message_cache = {}  # 用戶ID -> 最近消息列表
#         self.cache_size = 20     # 每個用戶快取的消息數量
        
#     def setup_database(self):
#         """創建或升級對話記憶資料庫"""
#         cursor = self.conn.cursor()
#         cursor.execute("""
#         CREATE TABLE IF NOT EXISTS conversations (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             person_name TEXT,
#             message TEXT,
#             role TEXT,
#             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#             importance REAL DEFAULT 1.0
#         )
#         """)
#         self.conn.commit()
        
#     def add_message(self, person_name: str, message: str, role: str = 'user', importance: float = 1.0):
#         """添加新的對話記錄，自動管理快取"""
#         cursor = self.conn.cursor()
#         cursor.execute(
#             "INSERT INTO conversations (person_name, message, role, importance) VALUES (?, ?, ?, ?)",
#             (person_name, message, role, importance)
#         )
#         self.conn.commit()
        
#         # 更新快取
#         if person_name not in self.message_cache:
#             self.message_cache[person_name] = []
#         self.message_cache[person_name].append((role, message))
        
#         # 維護快取大小
#         if len(self.message_cache[person_name]) > self.cache_size:
#             self.message_cache[person_name].pop(0)
            
#     def get_recent_messages(self, person_name: str, limit: int = 8) -> List[Tuple[str, str]]:
#         """獲取最近的對話記錄，優先使用快取"""
#         # 如果快取中有足夠的消息，直接返回
#         if person_name in self.message_cache and len(self.message_cache[person_name]) >= limit:
#             return self.message_cache[person_name][-limit:]
            
#         # 否則從資料庫讀取
#         cursor = self.conn.cursor()
#         cursor.execute(
#             "SELECT role, message FROM conversations WHERE person_name = ? ORDER BY timestamp DESC LIMIT ?",
#             (person_name, limit)
#         )
#         messages = [(role, msg) for role, msg in cursor.fetchall()]
#         messages.reverse()  # 按時間順序排列
        
#         # 更新快取
#         self.message_cache[person_name] = messages
        
#         return messages
        
#     def calculate_message_importance(self, message: str) -> float:
#         """計算消息重要性分數"""
#         # 這裡可以實現更複雜的重要性計算邏輯
#         return 1.0
        
#     def generate_conversation_summary(self, person_name: str, ai_client=None):
#         """生成對話摘要並存儲"""
#         messages = self.get_recent_messages(person_name, limit=20)
#         if not messages:
#             return None
            
#         # 這裡可以實現對話摘要生成邏輯
#         return None
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sqlite3
from datetime import datetime
import os
import threading

class EnhancedConversationMemory:
    """增強版對話記憶系統，改善上下文處理"""
    
    def __init__(self, db_path='conversation_memory.db'):
        """初始化對話記憶系統
        
        Args:
            db_path: 資料庫文件路徑
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.RLock()  # 用於線程安全的鎖
        
        self.setup_database()
        
        # 快取機制
        self.message_cache = {}  # 用戶ID -> 最近消息列表
        self.cache_size = 20     # 每個用戶快取的消息數量
        
    def __del__(self):
        """析構函數，確保關閉資料庫連接"""
        if hasattr(self, 'conn'):
            self.conn.close()
    
    def setup_database(self):
        """創建或升級對話記憶資料庫"""
        with self.lock:
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
        """添加新的對話記錄，自動管理快取
        
        Args:
            person_name: 用戶名稱
            message: 消息內容
            role: 角色 (user 或 assistant)
            importance: 重要性分數
        """
        # 生成對話ID
        current_date = datetime.now().strftime("%Y%m%d")
        conversation_id = f"{person_name}_{current_date}"
        
        with self.lock:
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
        """計算消息重要性分數
        
        Args:
            message: 消息內容
            
        Returns:
            float: 重要性分數
        """
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
        """獲取最近的對話記錄，優先使用快取
        
        Args:
            person_name: 用戶名稱
            limit: 最大消息數量
            
        Returns:
            list: 消息列表，每個元素為 (消息內容, 角色) 元組
        """
        # 先檢查快取
        if person_name in self.message_cache and len(self.message_cache[person_name]) >= limit:
            # 直接從快取返回
            recent = self.message_cache[person_name][-limit:]
            return [(msg['role'], msg['message']) for msg in recent]
        
        # 快取不足，從數據庫查詢
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                '''
                SELECT role, message FROM conversations 
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
                        result.append(("system", f"前次對話摘要: {summary[0]}"))
        
        # 返回結果，注意返回順序從舊到新
        messages = list(reversed(result))
        
        # 更新快取
        self.message_cache[person_name] = [
            {'role': role, 'message': msg, 'timestamp': datetime.now(), 'importance': 1.0}
            for role, msg in messages
        ]
        
        return messages
    
    def generate_conversation_summary(self, person_name, ai_client=None):
        """生成對話摘要並存儲
        
        Args:
            person_name: 用戶名稱
            ai_client: AI 客戶端 (OpenAI 或 Ollama)
            
        Returns:
            str: 生成的摘要，如果生成失敗則返回 None
        """
        # 檢查摘要表是否存在
        with self.lock:
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
                SELECT role, message FROM conversations 
                WHERE person_name = ?
                ORDER BY timestamp DESC LIMIT 20
                ''', 
                (person_name,)
            )
            messages = cursor.fetchall()
        
        if not messages or len(messages) < 5:
            return None  # 消息太少，不生成摘要
            
        # 構建對話文本
        conversation_text = "\n".join([f"{role}: {message}" for role, message in messages])
        
        try:
            summary = ""
            
            if ai_client:
                # 嘗試使用 AI 生成摘要
                try:
                    # 檢查 AI 客戶端類型
                    client_type = type(ai_client).__name__
                    
                    if hasattr(ai_client, 'use_openai') and ai_client.use_openai:
                        # 使用 OpenAI
                        response = ai_client.openai_client.chat.completions.create(
                            model=ai_client.model_name if ai_client.model_name != "gpt4o" else "gpt-4o",
                            messages=[
                                {"role": "system", "content": "請為以下對話生成簡短摘要，提取關鍵資訊。"},
                                {"role": "user", "content": conversation_text}
                            ],
                            max_tokens=100
                        )
                        summary = response.choices[0].message.content
                    else:
                        # 使用 Ollama
                        import ollama
                        response = ollama.chat(
                            model='deepseek-r1:8b',
                            messages=[
                                {"role": "system", "content": "請為以下對話生成簡短摘要，提取關鍵資訊。"},
                                {"role": "user", "content": conversation_text}
                            ]
                        )
                        summary = response['message']['content']
                except Exception as e:
                    print(f"使用 AI 生成摘要失敗: {e}")
                    # 使用簡單規則生成摘要
                    important_sentences = []
                    for role, msg in messages:
                        if role == "user" and len(msg) > 20:
                            important_sentences.append(msg)
                    
                    summary = "對話涉及: " + ", ".join(important_sentences[:3])
            else:
                # 使用簡單規則生成摘要
                important_sentences = []
                for role, msg in messages:
                    if role == "user" and len(msg) > 20:
                        important_sentences.append(msg)
                
                summary = "對話涉及: " + ", ".join(important_sentences[:3])
            
            # 存儲摘要
            current_time = datetime.now()
            with self.lock:
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
            
    def clean_old_records(self, days=30):
        """清理舊的對話記錄
        
        Args:
            days: 保留天數，超過此天數的記錄將被刪除
        """
        try:
            # 計算截止日期
            cutoff_date = datetime.now() - datetime.timedelta(days=days)
            cutoff_str = cutoff_date.strftime("%Y-%m-%d %H:%M:%S")
            
            with self.lock:
                cursor = self.conn.cursor()
                
                # 刪除舊記錄
                cursor.execute(
                    "DELETE FROM conversations WHERE timestamp < ?",
                    (cutoff_str,)
                )
                
                # 刪除舊摘要
                cursor.execute(
                    "DELETE FROM conversation_summaries WHERE timestamp < ?",
                    (cutoff_str,)
                )
                
                self.conn.commit()
                
                print(f"已清理 {cursor.rowcount} 條舊記錄")
        except Exception as e:
            print(f"清理舊記錄時出錯: {e}")
            
    def get_conversation_by_date(self, person_name, date):
        """獲取指定日期的對話
        
        Args:
            person_name: 用戶名稱
            date: 日期字符串，格式為 YYYYMMDD
            
        Returns:
            list: 對話列表
        """
        conversation_id = f"{person_name}_{date}"
        
        with self.lock:
            cursor = self.conn.cursor()
            cursor.execute(
                '''
                SELECT role, message, timestamp FROM conversations 
                WHERE conversation_id = ? 
                ORDER BY timestamp ASC
                ''',
                (conversation_id,)
            )
            result = cursor.fetchall()
            
        return [(role, message, timestamp) for role, message, timestamp in result]