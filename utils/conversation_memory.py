import sqlite3
from typing import List, Tuple, Optional
import json

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
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_name TEXT,
            message TEXT,
            role TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            importance REAL DEFAULT 1.0
        )
        """)
        self.conn.commit()
        
    def add_message(self, person_name: str, message: str, role: str = 'user', importance: float = 1.0):
        """添加新的對話記錄，自動管理快取"""
        cursor = self.conn.cursor()
        cursor.execute(
            "INSERT INTO conversations (person_name, message, role, importance) VALUES (?, ?, ?, ?)",
            (person_name, message, role, importance)
        )
        self.conn.commit()
        
        # 更新快取
        if person_name not in self.message_cache:
            self.message_cache[person_name] = []
        self.message_cache[person_name].append((role, message))
        
        # 維護快取大小
        if len(self.message_cache[person_name]) > self.cache_size:
            self.message_cache[person_name].pop(0)
            
    def get_recent_messages(self, person_name: str, limit: int = 8) -> List[Tuple[str, str]]:
        """獲取最近的對話記錄，優先使用快取"""
        # 如果快取中有足夠的消息，直接返回
        if person_name in self.message_cache and len(self.message_cache[person_name]) >= limit:
            return self.message_cache[person_name][-limit:]
            
        # 否則從資料庫讀取
        cursor = self.conn.cursor()
        cursor.execute(
            "SELECT role, message FROM conversations WHERE person_name = ? ORDER BY timestamp DESC LIMIT ?",
            (person_name, limit)
        )
        messages = [(role, msg) for role, msg in cursor.fetchall()]
        messages.reverse()  # 按時間順序排列
        
        # 更新快取
        self.message_cache[person_name] = messages
        
        return messages
        
    def calculate_message_importance(self, message: str) -> float:
        """計算消息重要性分數"""
        # 這裡可以實現更複雜的重要性計算邏輯
        return 1.0
        
    def generate_conversation_summary(self, person_name: str, ai_client=None):
        """生成對話摘要並存儲"""
        messages = self.get_recent_messages(person_name, limit=20)
        if not messages:
            return None
            
        # 這裡可以實現對話摘要生成邏輯
        return None
