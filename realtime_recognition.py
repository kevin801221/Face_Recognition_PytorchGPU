import os
import cv2
import json
import numpy as np
from datetime import datetime
from collections import deque
import time
import math
import torch
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# 初始化 InsightFace
app = FaceAnalysis(
    name='buffalo_l',  # 使用輕量級模型
    allowed_modules=['detection', 'recognition'],  # 只啟用需要的模組
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
)
app.prepare(ctx_id=0 if torch.cuda.is_available() else -1)

# 初始化存取機制
detection_log_file = 'detection_log.json'
detection_log = {}
if os.path.exists(detection_log_file):
    with open(detection_log_file, 'r') as f:
        detection_log = json.load(f)

def calculate_face_quality(face_img):
    """計算人臉圖片的質量分數"""
    # 計算亮度
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    brightness_score = 1.0 - abs(128 - brightness) / 128
    
    # 計算清晰度（使用Laplacian算子）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    sharpness_score = min(1.0, sharpness / 500)
    
    # 計算人臉大小分數
    size_score = min(1.0, (face_img.shape[0] * face_img.shape[1]) / (200 * 200))
    
    # 綜合分數
    quality_score = (brightness_score + sharpness_score + size_score) / 3
    return quality_score

def draw_label(img, text, pos, color):
    """在圖像上繪製文字標籤"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    
    # 獲取文字大小
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # 繪製背景矩形
    padding = 5
    cv2.rectangle(img, 
                 (pos[0], pos[1] - text_height - 2*padding),
                 (pos[0] + text_width + 2*padding, pos[1]),
                 color, -1)
    
    # 繪製文字
    cv2.putText(img, text, (pos[0] + padding, pos[1] - padding), 
                font, scale, (0, 0, 0), thickness)


class FaceTracker:
    """追蹤單個人臉的類別"""
    def __init__(self, buffer_size=10):
        self.positions = deque(maxlen=buffer_size)
        self.last_identity = None
        self.identity_counter = 0
        self.identity_threshold = 2  # 降低閾值，使其更容易穩定
        self.best_confidence = 0
        self.detection_time = time.time()  # 初始化為當前時間
        self.face_id = None
        self.quality_scores = deque(maxlen=buffer_size)
        self.confidence_scores = deque(maxlen=buffer_size)
        
    def calculate_stability_score(self):
        """計算追蹤穩定性分數"""
        if len(self.positions) < 2:
            return 0.0
            
        movements = []
        for i in range(1, len(self.positions)):
            prev = self.positions[i-1]
            curr = self.positions[i]
            movement = math.sqrt(
                (curr[0] - prev[0])**2 +
                (curr[1] - prev[1])**2
            )
            movements.append(movement)
            
        avg_movement = np.mean(movements)
        stability = max(0.0, 1.0 - (avg_movement / 50))
        return stability
        
    def update(self, x, y, w, h, identity=None, confidence=None, quality_score=None):
        """更新追蹤器狀態"""
        self.positions.append((x, y, w, h))
        
        if quality_score is not None:
            self.quality_scores.append(quality_score)
            
        if confidence is not None:
            self.confidence_scores.append(confidence)
        
        stability_score = self.calculate_stability_score()
        
        if identity and confidence:
            avg_quality = np.mean(self.quality_scores) if self.quality_scores else 0.5
            weighted_conf = confidence * 0.6 + avg_quality * 100 * 0.2 + stability_score * 100 * 0.2
            
            if identity == self.last_identity:
                self.identity_counter += 1
                if weighted_conf > self.best_confidence:
                    self.best_confidence = weighted_conf
                    self.detection_time = time.time()
            else:
                self.identity_counter = 1
                self.last_identity = identity
                self.best_confidence = weighted_conf
                self.detection_time = time.time()
                self.face_id = f"{identity}_{int(time.time())}"
        
        if len(self.positions) > 0:
            x = int(sum(p[0] for p in self.positions) / len(self.positions))
            y = int(sum(p[1] for p in self.positions) / len(self.positions))
            w = int(sum(p[2] for p in self.positions) / len(self.positions))
            h = int(sum(p[3] for p in self.positions) / len(self.positions))
            return x, y, w, h
        return None

class MultiPersonTracker:
    """管理多個人臉追蹤器的類別"""
    def __init__(self, max_persons=10):
        self.trackers = {}
        self.max_persons = max_persons
        self.last_cleanup = time.time()
        
    def _cleanup_old_trackers(self):
        """清理過期的追蹤器"""
        current_time = time.time()
        if current_time - self.last_cleanup < 5:
            return
            
        to_remove = []
        for face_id, tracker in self.trackers.items():
            if current_time - tracker.detection_time > 5:
                to_remove.append(face_id)
                
        for face_id in to_remove:
            del self.trackers[face_id]
            
        self.last_cleanup = current_time
        
    def update(self, faces_data):
        """更新所有追蹤器"""
        self._cleanup_old_trackers()
        
        # 更新現有追蹤器
        for face_data in faces_data:
            x, y, w, h = face_data['bbox']
            identity = face_data['identity']
            confidence = face_data['confidence']
            quality_score = face_data.get('quality_score', None)
            
            # 尋找最近的追蹤器
            best_tracker = None
            min_distance = float('inf')
            
            for tracker in self.trackers.values():
                if not tracker.positions:
                    continue
                tx, ty, _, _ = tracker.positions[-1]
                distance = math.sqrt((x - tx)**2 + (y - ty)**2)
                if distance < min_distance and distance < 100:
                    min_distance = distance
                    best_tracker = tracker
            
            tracker = best_tracker
            if not tracker and len(self.trackers) < self.max_persons:
                tracker = FaceTracker()
                self.trackers[f"face_{len(self.trackers)}"] = tracker
                
            if tracker:
                pos = tracker.update(x, y, w, h, identity, confidence, quality_score)
                if pos:
                    face_data['bbox'] = pos
        
        return faces_data
    
    def get_best_detections(self):
        """獲取最佳檢測結果"""
        results = []
        for tracker in self.trackers.values():
            if (tracker.identity_counter >= tracker.identity_threshold and
                tracker.positions and tracker.last_identity):
                pos = tracker.positions[-1]
                results.append({
                    'bbox': pos,
                    'identity': tracker.last_identity,
                    'confidence': tracker.best_confidence,
                    'face_id': tracker.face_id
                })
        return results

def cosine_similarity(a, b):
    """計算兩個向量的餘弦相似度"""
    a = np.array(a)
    b = np.array(b)
    
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0  # 避免除零錯誤
    
    similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.clip(similarity, -1.0, 1.0)  # 確保範圍在 [-1,1]

def train_face():
    """訓練人臉特徵"""
    print("開始訓練人臉特徵...")
    
    training_dir = "training_images"
    if not os.path.exists(training_dir):
        print(f"找不到訓練資料夾：{training_dir}")
        return
        
    known_face_data = {}
    
    # 遍歷所有人臉圖片
    for person_name in os.listdir(training_dir):
        person_dir = os.path.join(training_dir, person_name)
        if not os.path.isdir(person_dir):
            continue
            
        print(f"正在處理 {person_name} 的圖片...")
        features = []
        
        for img_name in os.listdir(person_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"無法讀取圖片：{img_path}")
                continue
                
            # 檢測和提取特徵
            faces = app.get(img)
            if len(faces) == 0:
                print(f"在圖片中沒有檢測到人臉：{img_path}")
                continue
            elif len(faces) > 1:
                print(f"在圖片中檢測到多個人臉：{img_path}")
                continue
                
            # 使用第一個檢測到的人臉
            face = faces[0]
            features.append(face.embedding.tolist())
            
        if features:
            known_face_data[person_name] = features
            print(f"成功提取 {len(features)} 個特徵")
        
    # 保存特徵到文件
    if known_face_data:
        with open('face_features.json', 'w') as f:
            json.dump(known_face_data, f)
        print("特徵提取完成，已保存到 face_features.json")
    else:
        print("沒有提取到任何特徵")

def realtime_face_recognition():
    """即時人臉識別主函數"""
    print("正在初始化人臉識別系統...")
    print(f"GPU 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    known_face_data = {}
    if os.path.exists('face_features.json'):
        with open('face_features.json', 'r') as f:
            known_face_data = json.load(f)
            print(f"已載入 {len(known_face_data)} 個人的特徵")
    else:
        print("警告：找不到 face_features.json，請先運行 train_face.py")
    
    print("系統初始化完成，開始識別...")
    
    # 記錄本次執行的開始時間
    start_time = datetime.now()
    session_key = start_time.strftime("%Y/%m/%d %H:%M")
    detection_log[session_key] = []
    
    # 用於追蹤5秒內的檢測結果
    recent_detections = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("無法讀取攝像頭畫面")
            break
            
        faces = app.get(frame)
        faces_data = []
        current_time = datetime.now()
        
        # 清理超過5秒的檢測結果
        recent_detections = [d for d in recent_detections 
                           if (current_time - d['timestamp']).total_seconds() <= 5]
        
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            
            face_img = frame[y1:y2, x1:x2]
            if face_img.size == 0:
                continue
            
            embedding = face.embedding
            best_match = "Unknown"
            best_similarity = -1
            
            for name, features in known_face_data.items():
                for feature in features:
                    similarity = cosine_similarity(embedding, feature)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
            
            confidence = max(0, min(100, (best_similarity + 1) / 2 * 100))
            identity = best_match if confidence > 60 else "Unknown"
            
            # 記錄高置信度的檢測結果
            if confidence > 90:
                detection = {
                    'timestamp': current_time,
                    'identity': identity,
                    'confidence': confidence,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                }
                recent_detections.append(detection)
                
                # 將檢測結果添加到本次執行的記錄中
                detection_log[session_key].append({
                    'time': current_time.strftime("%H:%M:%S"),
                    'identity': identity,
                    'confidence': float(confidence),  # 轉換為float以便JSON序列化
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
            
            faces_data.append({
                'bbox': (x1, y1, x2-x1, y2-y1),
                'identity': identity,
                'confidence': confidence
            })
            print(f"識別結果: {identity}, 置信度: {confidence:.1f}%")
        
        # 在畫面上顯示最近5秒內的高置信度檢測數量
        high_conf_count = len(recent_detections)
        if high_conf_count > 0:
            cv2.putText(frame, f"High confidence detections (5s): {high_conf_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for face in faces_data:
            x, y, w, h = face['bbox']
            identity = face['identity']
            conf = face['confidence']
            
            color = (0, 255, 0) if conf > 70 else (0, 255, 255) if conf > 50 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            label = f"{identity} | confidence: {conf:.1f}%"
            draw_label(frame, label, (x, y-10), color)
        
        cv2.imshow('Face Recognition', frame)
        
        # 定期保存檢測記錄
        if len(detection_log[session_key]) > 0 and len(detection_log[session_key]) % 10 == 0:
            with open(detection_log_file, 'w') as f:
                json.dump(detection_log, f, indent=2)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # 退出前保存檢測記錄
            if len(detection_log[session_key]) > 0:
                with open(detection_log_file, 'w') as f:
                    json.dump(detection_log, f, indent=2)
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    realtime_face_recognition()
