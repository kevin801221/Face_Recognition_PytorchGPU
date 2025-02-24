import sqlite3
import json
from datetime import datetime, timedelta
import os

class ChatMemory:
    def __init__(self, db_path="chat_memory.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """初始化數據庫"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # 創建對話歷史表
        c.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                message TEXT,
                role TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 創建問題記錄表（避免重複問題）
        c.execute('''
            CREATE TABLE IF NOT EXISTS asked_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT,
                question TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_message(self, user_name, message, role="assistant"):
        """添加新的對話消息"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO chat_history (user_name, message, role) VALUES (?, ?, ?)",
            (user_name, message, role)
        )
        conn.commit()
        conn.close()
    
    def add_asked_question(self, user_name, question):
        """記錄已問過的問題"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            "INSERT INTO asked_questions (user_name, question) VALUES (?, ?)",
            (user_name, question)
        )
        conn.commit()
        conn.close()
    
    def get_recent_messages(self, user_name, limit=5):
        """獲取最近的對話記錄"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(
            """
            SELECT message, role FROM chat_history 
            WHERE user_name = ? 
            ORDER BY timestamp DESC LIMIT ?
            """,
            (user_name, limit)
        )
        messages = c.fetchall()
        conn.close()
        return [(msg[0], msg[1]) for msg in messages][::-1]  # 反轉順序，從舊到新
    
    def is_question_asked(self, user_name, question, within_hours=24):
        """檢查問題是否在指定時間內問過"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        time_threshold = datetime.now() - timedelta(hours=within_hours)
        
        c.execute(
            """
            SELECT COUNT(*) FROM asked_questions 
            WHERE user_name = ? 
            AND question = ? 
            AND timestamp > ?
            """,
            (user_name, question, time_threshold)
        )
        count = c.fetchone()[0]
        conn.close()
        return count > 0
    
    def clean_old_records(self, days=7):
        """清理舊記錄"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        time_threshold = datetime.now() - timedelta(days=days)
        
        c.execute(
            "DELETE FROM chat_history WHERE timestamp < ?",
            (time_threshold,)
        )
        c.execute(
            "DELETE FROM asked_questions WHERE timestamp < ?",
            (time_threshold,)
        )
        
        conn.commit()
        conn.close()
