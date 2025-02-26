#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import argparse
import threading
import time
from datetime import datetime
from queue import Queue
from dotenv import load_dotenv

# UI Components
from PySide6.QtWidgets import QApplication
from gui.chat_window import ChatWindow

# Utilities
from utils.resource_monitor import ResourceMonitor
from utils.speech_input import SpeechRecognizer
from utils.conversation_memory import EnhancedConversationMemory

# Services
from services.face_service import FaceService
from services.api_client import get_employee_data
from services.llm_service import LLMService

# Configuration
from config import CAMERA_WIDTH, CAMERA_HEIGHT, parse_arguments

def main():
    """Main application entry point"""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Initialize Qt application
    qt_app = QApplication.instance()
    if not qt_app:
        qt_app = QApplication([])
    
    # Initialize chat window
    chat_window = ChatWindow()
    chat_window.show()
    
    # 先顯示測試消息
    chat_window.show_message("系統啟動測試：請問您能看到這條消息嗎？")

    # 檢查 API 密鑰
    if args.model in ["gpt4o", "gpt-4o", "gpt-4"]:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("警告：找不到 OPENAI_API_KEY 環境變數！")
            chat_window.show_message("警告: OpenAI API 密鑰未設置，系統可能無法正常工作。")
    
    chat_window.show_message(f"系統啟動: 使用 {args.model} 模型，如果您看到此消息，界面正常工作。")
    
    # Initialize resource monitor
    resource_monitor = ResourceMonitor(target_cpu_percent=args.cpu_limit)
    resource_monitor.start_monitoring()
    
    # Initialize conversation memory
    conversation_memory = EnhancedConversationMemory()
    
    # Initialize LLM service
    llm_service = LLMService(model_name=args.model)
    
    # Initialize face service
    face_service = FaceService()
    
    # Initialize speech recognizer
    speech_recognizer = SpeechRecognizer()
    
    # Feature matching queue
    feature_queue = Queue()
    result_queue = Queue()
    
    # Start face recognition
    realtime_face_recognition(
        args=args,
        chat_window=chat_window,
        resource_monitor=resource_monitor,
        conversation_memory=conversation_memory,
        llm_service=llm_service,
        face_service=face_service,
        speech_recognizer=speech_recognizer,
        feature_queue=feature_queue,
        result_queue=result_queue
    )
    
    # Start Qt event loop
    qt_app.exec()
    
def realtime_face_recognition(
    args, 
    chat_window, 
    resource_monitor, 
    conversation_memory, 
    llm_service, 
    face_service, 
    speech_recognizer, 
    feature_queue, 
    result_queue
):
    """Real-time face recognition main function"""
    print("啟動即時人臉識別...")
    
    # Load face features
    known_face_data = face_service.load_face_features()
    if not known_face_data:
        print("無法加載人臉特徵，系統將無法識別用戶")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to decrease latency
    
    if not cap.isOpened():
        print("無法開啟攝像頭")
        return
    
    # Initialize variables for main loop
    frame_count = 0
    current_person = None
    employee_cache = {}
    recent_detections = {}
    active_conversations = set()
    
    # Sleep mode variables
    sleep_mode = False
    last_face_position = None
    no_face_counter = 0
    POSITION_THRESHOLD = 50
    NO_FACE_THRESHOLD = 30
    
    # Performance monitoring variables
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    # Motion detection variables
    prev_gray = None
    motion_threshold = 5.0
    
    # Initialize feature matching worker thread
    def feature_matching_worker():
        while True:
            try:
                if not feature_queue.empty():
                    face_feature = feature_queue.get()
                    best_match, min_distance = face_service.batch_feature_matching(
                        face_feature, known_face_data
                    )
                    result_queue.put((best_match, min_distance))
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"特徵比對錯誤: {e}")
                time.sleep(0.1)
    
    # Start feature matching thread
    feature_thread = threading.Thread(target=feature_matching_worker, daemon=True)
    feature_thread.start()
    
    # Initialize speech recognition
    speech_text_buffer = ""
    last_speech_time = time.time()
    SPEECH_TIMEOUT = 2.0
    
    def process_speech_input():
        nonlocal speech_text_buffer, last_speech_time
        
        current_time = time.time()
        if speech_text_buffer and (current_time - last_speech_time) >= SPEECH_TIMEOUT:
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
            chat_window.input_field.setText(speech_text_buffer)
    
    # Set speech recognition callback
    speech_recognizer.on_speech_detected = on_speech_detected
    
    # Start speech recognition thread
    speech_thread = threading.Thread(target=speech_recognizer.start_listening, daemon=True)
    speech_thread.start()
    
    # Set chat window message handler
    def on_user_message(message):
        nonlocal current_person, employee_cache
        print(f"收到用戶消息: {message}")  # 添加日誌
        
        # 測試直接回應，檢查界面更新是否正常
        chat_window.show_message(f"收到消息: {message}，正在處理...")
        
        # 正常回應邏輯
        if current_person and current_person in employee_cache:
            try:
                response = llm_service.handle_user_message(
                    employee_cache[current_person],
                    message,
                    conversation_memory
                )
                print(f"AI 回應: {response}")  # 添加日誌
                chat_window.show_message(response)
            except Exception as e:
                print(f"處理消息時出錯: {e}")
                chat_window.show_message(f"處理消息時出錯: {e}")
        else:
            chat_window.show_message("抱歉，我現在無法確定你是誰。請讓我看清你的臉。")

    # 確保正確連接信號
    chat_window.message_sent.connect(on_user_message)
    
    print("即時人臉識別系統已啟動...")
    
    # Main loop
    while True:
        try:
            loop_start = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print("無法讀取影像")
                time.sleep(0.1)
                continue
            
            # Update FPS counter
            fps_counter += 1
            if time.time() - fps_start_time >= 5:
                current_fps = fps_counter / (time.time() - fps_start_time)
                print(f"目前 FPS: {current_fps:.1f}, CPU 使用率: {resource_monitor.current_cpu_percent:.1f}%")
                fps_counter = 0
                fps_start_time = time.time()
            
            # Increment frame count
            frame_count += 1
            
            # Skip processing based on CPU usage
            if not resource_monitor.should_process_frame(frame_count):
                if current_person:
                    # If a user was previously identified, display the name
                    cv2.putText(
                        frame,
                        f"Identified: {current_person}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 0),
                        2
                    )
                
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            
            # Simple motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            motion_detected = False
            if prev_gray is not None:
                frame_diff = cv2.absdiff(gray, prev_gray)
                motion_score = np.mean(frame_diff)
                motion_detected = motion_score > motion_threshold
            
            prev_gray = gray.copy()
            
            # Only detect faces when motion is detected or periodically
            faces = []
            if motion_detected or frame_count % 15 == 0:
                faces = face_service.detect_faces(frame)
            
            # Update no face counter
            if not faces:
                no_face_counter += 1
                if no_face_counter >= NO_FACE_THRESHOLD:
                    sleep_mode = False
                    last_face_position = None
                    current_person = None
                
                cv2.imshow('Face Recognition', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue
            else:
                no_face_counter = 0
            
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # In sleep mode, only check for position changes
            if sleep_mode and faces:
                face = faces[0]
                # 檢查 face 的結構類型，根據不同的結構獲取 bbox
                if hasattr(face, 'bbox'):
                    # InsightFace 直接返回的物件
                    current_pos = (face.bbox[0], face.bbox[1])
                elif isinstance(face, dict) and 'bbox' in face:
                    # 以字典形式封裝的物件
                    current_pos = (face['bbox'][0], face['bbox'][1])
                else:
                    # 未能識別的結構，使用默認值
                    current_pos = (0, 0)
                    print("警告: 無法識別人臉結構類型")
                
                if last_face_position:
                    dx = current_pos[0] - last_face_position[0]
                    dy = current_pos[1] - last_face_position[1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance > POSITION_THRESHOLD:
                        print("檢測到顯著移動，退出休眠模式")
                        sleep_mode = False
                
                last_face_position = current_pos
                
                # Use the last identified person in sleep mode
                for person_id, last_time in recent_detections.items():
                    if (datetime.strptime(current_time, "%H:%M:%S") - 
                        datetime.strptime(last_time, "%H:%M:%S")).total_seconds() < 30:
                        current_person = person_id
                        break
            
            # Full face recognition in non-sleep mode
            if not sleep_mode and faces:
                face = faces[0]
                
                # 檢查 face 的結構類型，根據不同的結構獲取資訊
                if hasattr(face, 'bbox') and hasattr(face, 'normed_embedding'):
                    # InsightFace 直接返回的物件
                    current_pos = (face.bbox[0], face.bbox[1])
                    face_feature = face.normed_embedding
                    bbox = face.bbox.astype(int)
                elif isinstance(face, dict):
                    # 以字典形式封裝的物件
                    current_pos = (face['bbox'][0], face['bbox'][1])
                    face_feature = face['embedding'] if 'embedding' in face else None
                    bbox = face['bbox'].astype(int) if 'bbox' in face else np.array([0, 0, 100, 100])
                else:
                    # 未能識別的結構，使用默認值
                    current_pos = (0, 0)
                    face_feature = None
                    bbox = np.array([0, 0, 100, 100])
                    print("警告: 無法識別人臉結構類型")
                
                last_face_position = current_pos
                
                # Extract face feature
                if face_feature is not None:
                    # Add to feature matching queue
                    if feature_queue.qsize() < 5:
                        feature_queue.put(face_feature)
                    
                    # Check for matching results
                    if not result_queue.empty():
                        best_match, min_distance = result_queue.get()
                        
                        # If a match is found
                        if best_match and min_distance < 0.3:
                            print(f"識別到用戶: {best_match}, 距離: {min_distance:.4f}")
                            current_person = best_match
                            recent_detections[current_person] = current_time
                            
                            # If this person is not cached
                            if current_person not in employee_cache:
                                try:
                                    print(f"嘗試獲取員工資料: {current_person}")
                                    employee_data = get_employee_data(current_person)
                                    if employee_data:
                                        print(f"成功獲取員工資料: {employee_data['name']}")
                                        employee_cache[current_person] = employee_data
                                        
                                        # If this is a new conversation
                                        if current_person not in active_conversations:
                                            print("開始新對話")
                                            
                                            # Start a new conversation in a separate thread
                                            def start_conversation():
                                                response = llm_service.chat_with_employee(
                                                    employee_data,
                                                    is_first_chat=True
                                                )
                                                if response:
                                                    chat_window.show_message(response)
                                            
                                            threading.Thread(target=start_conversation, daemon=True).start()
                                            active_conversations.add(current_person)
                                except Exception as e:
                                    print(f"獲取員工資料時發生錯誤: {e}")
                            
                            # Generate conversation summary every 10 minutes
                            if current_person in active_conversations:
                                current_minute = datetime.now().minute
                                if current_minute % 10 == 0 and current_minute != 0:
                                    threading.Thread(
                                        target=conversation_memory.generate_conversation_summary,
                                        args=(current_person, llm_service),
                                        daemon=True
                                    ).start()
                        else:
                            print(f"無法識別用戶，最小距離: {min_distance:.4f}")
                            # Only reset current user when distance is very large
                            if min_distance > 0.6:
                                current_person = None
            
            # Process speech input
            process_speech_input()
            
            # Draw faces on frame
            if faces:
                face = faces[0]
                
                # 檢查 face 的結構類型，根據不同的結構獲取 bbox
                if hasattr(face, 'bbox'):
                    # InsightFace 直接返回的物件
                    bbox = face.bbox.astype(int)
                elif isinstance(face, dict) and 'bbox' in face:
                    # 以字典形式封裝的物件
                    bbox = face['bbox'].astype(int)
                else:
                    # 未能識別的結構，使用默認值
                    bbox = np.array([0, 0, 100, 100])
                    print("警告: 無法識別人臉結構類型")
                
                # Choose color based on recognition result
                if current_person:
                    color = (0, 255, 0)  # Green - identified
                else:
                    color = (0, 165, 255)  # Orange - unidentified
                
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                if current_person:
                    # Display name above the face
                    cv2.putText(
                        frame,
                        current_person,
                        (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2
                    )
                    
                    # Display system info in corner
                    cv2.putText(
                        frame,
                        f"CPU: {resource_monitor.current_cpu_percent:.1f}%",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"FPS: {current_fps:.1f}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
            
            # Calculate and display processing time
            process_time = time.time() - loop_start
            if process_time > 0.1:
                print(f"警告: 處理時間較長 {process_time:.3f}秒")
            
            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Release CPU to avoid 100% usage
            if process_time < 0.03:
                time.sleep(0.01)
                
        except Exception as e:
            print(f"主循環錯誤: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.1)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    resource_monitor.stop_monitoring()
    speech_recognizer.stop_listening()
    
    print("即時人臉識別系統已關閉")

def train_face(name=None):
    """Function to train a new face for recognition"""
    face_service = FaceService()
    face_service.train_face(name)

if __name__ == "__main__":
    main()