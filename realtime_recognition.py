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
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from PySide6.QtWidgets import QApplication
from gui.chat_window import ChatWindow
import sys

# API 設置
API_BASE_URL = "https://ddfab4235344.ngrok.app/api/employees/search/by-name"

# 初始化 Qt 應用
qt_app = QApplication.instance()
if not qt_app:
    qt_app = QApplication(sys.argv)
chat_window = ChatWindow()

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

def generate_prompt(employee_data):
    """根據員工資料生成 prompt"""
    prompt = f"""你現在是 YCM 館長，一個友善、專業的智能助手。你正在與 {employee_data['name']} 進行對話。

你應該：
1. 以 "哈囉 {employee_data['name']}，今天過得還好嗎？我是 YCM 館長" 開始對話
2. 根據以下資訊，從中選擇一個有趣的點來延續對話
3. 保持專業但友善的態度

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

請根據以上資訊，生成一個自然的開場白，包含：
1. "哈囉 [名字]，今天過得還好嗎？我是 YCM 館長" 
2. 從上述資訊中選擇一個有趣的點（證書、興趣、技能等）來延續對話
3. 保持簡短但友善的語氣
"""
    return prompt

def chat_with_employee(employee_data):
    """使用 Ollama 與員工對話"""
    try:
        # 生成初始 prompt
        system_prompt = generate_prompt(employee_data)
        
        # 建立對話
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
        print(f"與 Ollama 通信時發生錯誤: {e}")
        return None

def get_employee_data(name):
    """從 API 獲取員工資料"""
    try:
        response = requests.get(f"{API_BASE_URL}/{name}")
        response.encoding = 'utf-8'  # 設置響應的編碼為 UTF-8
        if response.status_code == 200:
            return response.json()["data"][0] if response.json()["data"] else None
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
                
                # 清理舊的檢測記錄和對話冷卻時間
                for person_id in list(recent_detections.keys()):
                    last_detection_time = datetime.strptime(recent_detections[person_id], "%H:%M:%S")
                    time_diff = (current_time_obj - last_detection_time).total_seconds()
                    if time_diff > 5:  # 如果超過5秒沒有檢測到，則移除記錄
                        del recent_detections[person_id]
                        if person_id in chat_cooldown:
                            del chat_cooldown[person_id]
                
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
                    
                    # 如果是已知人臉且置信度足夠，記錄到當前會話日誌
                    if identity != "Unknown" and confidence > 60:
                        # 檢查是否需要添加新的檢測記錄（避免重複記錄同一個人）
                        if identity not in recent_detections or \
                           (datetime.strptime(current_time, "%H:%M:%S") - 
                            datetime.strptime(recent_detections[identity], "%H:%M:%S")).total_seconds() > 5:
                            
                            # 獲取員工完整資料
                            if identity not in employee_cache:
                                employee_data = get_employee_data(identity)
                                if employee_data:
                                    employee_cache[identity] = employee_data
                                    print(f"已獲取 {identity} 的完整資料")
                                    
                                    # 首次檢測到時，啟動對話
                                    response = chat_with_employee(employee_data)
                                    if response:
                                        chat_window.show_message(f"YCM館長: {response}")
                            
                            detection_record = {
                                "time": current_time,
                                "identity": identity,
                                "confidence": confidence,
                                "bbox": bbox.tolist(),
                                "employee_data": employee_cache.get(identity)
                            }
                            current_session_log[current_date].append(detection_record)
                            recent_detections[identity] = current_time
                            
                            # 檢查是否需要進行新的對話（每30秒一次）
                            if identity not in chat_cooldown or \
                               (current_time_obj - datetime.strptime(chat_cooldown[identity], "%H:%M:%S")).total_seconds() > 30:
                                if identity in employee_cache:
                                    response = chat_with_employee(employee_cache[identity])
                                    if response:
                                        chat_window.show_message(f"YCM館長: {response}")
                                    chat_cooldown[identity] = current_time
                            
                            # 保存當前會話日誌，使用 UTF-8 編碼
                            with open('current_session_log.json', 'w', encoding='utf-8') as f:
                                json.dump(current_session_log, f, indent=2, ensure_ascii=False)
                    
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
