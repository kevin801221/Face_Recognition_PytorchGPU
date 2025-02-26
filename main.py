import os
import cv2
import time
from dotenv import load_dotenv
from modules import (
    FaceRecognizer,
    ChatWindow,
    SpeechRecognizer,
    EmployeeManager,
    ChatAI
)

# 載入環境變數
load_dotenv()

class FaceRecognitionSystem:
    def __init__(self):
        # 初始化各個模組
        self.face_recognizer = FaceRecognizer(model_path="buffalo_l")
        self.employee_manager = EmployeeManager()
        self.chat_ai = ChatAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # 初始化狀態變數
        self.current_person = None
        self.active_conversations = set()
        
        # 創建聊天窗口
        self.chat_window = ChatWindow(on_send=self.handle_user_message)
        
        # 初始化語音識別
        self.speech_recognizer = SpeechRecognizer(
            on_speech_detected=self.handle_speech_input
        )
        
    def handle_user_message(self, message: str):
        """處理用戶文字輸入"""
        if self.current_person:
            employee_data = self.employee_manager.get_employee(self.current_person)
            if employee_data:
                response = self.chat_ai.chat(self.current_person, message, employee_data)
                self.chat_window.show_message(f"AI: {response}")
        else:
            self.chat_window.show_message("AI: 抱歉，我現在無法確認您的身份。")
            
    def handle_speech_input(self, text: str):
        """處理語音輸入"""
        if text:
            self.chat_window.input_field.delete(0, 'end')
            self.chat_window.input_field.insert(0, text)
            self.chat_window.send_message()
            
    def process_frame(self, frame):
        """處理視頻幀"""
        # 檢測和識別人臉
        person_id, distance = self.face_recognizer.process_frame(
            frame, 
            self.employee_manager.employee_cache
        )
        
        # 更新當前用戶
        if person_id != self.current_person:
            if person_id:
                print(f"識別到用戶: {person_id}, 距離: {distance}")
                self.current_person = person_id
                
                # 如果是新的對話
                if person_id not in self.active_conversations:
                    employee_data = self.employee_manager.get_employee(person_id)
                    if employee_data:
                        response = self.chat_ai.chat(person_id, "", employee_data)
                        self.chat_window.show_message(f"AI: {response}")
                        self.active_conversations.add(person_id)
            else:
                self.current_person = None
                
        # 在框架上繪製人臉框
        if person_id:
            faces = self.face_recognizer.detect_faces(frame)
            if faces:
                face = faces[0]
                bbox = face.bbox.astype(int)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{person_id} ({distance:.2f})", 
                           (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (0, 255, 0), 2)
                           
    def run(self):
        """運行系統"""
        # 開始語音識別
        self.speech_recognizer.start()
        
        # 開啟攝像頭
        cap = cv2.VideoCapture(0)
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # 處理當前幀
                self.process_frame(frame)
                
                # 顯示視頻
                cv2.imshow('Face Recognition', frame)
                
                # 檢查退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        finally:
            # 清理資源
            cap.release()
            cv2.destroyAllWindows()
            self.speech_recognizer.stop()
            self.chat_window.close()

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()
