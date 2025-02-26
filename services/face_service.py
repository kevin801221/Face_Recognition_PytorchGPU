#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import json
import time
import numpy as np
import torch
from insightface.app import FaceAnalysis
from datetime import datetime

from utils.face_utils import batch_feature_matching, gpu_cosine_similarity
from config import FACE_DETECTION_SIZE

class FaceService:
    """Face recognition service using InsightFace"""
    
    def __init__(self, model_name='buffalo_l'):
        """Initialize face recognition service"""
        self.model_name = model_name
        self.providers = ['CUDAExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        
        # Initialize InsightFace model
        self.face_app = FaceAnalysis(
            name=model_name,
            allowed_modules=['detection', 'recognition'],
            providers=self.providers
        )
        
        # Use smaller detection size for better performance
        self.face_app.prepare(
            ctx_id=0 if torch.cuda.is_available() else -1,
            det_size=FACE_DETECTION_SIZE
        )
        
        # Print CUDA status
        if torch.cuda.is_available():
            print(f"使用 CUDA 加速，設備: {torch.cuda.get_device_name(0)}")
        else:
            print("無法使用 CUDA 加速，將使用 CPU 模式")
    
    def load_face_features(self, features_path='face_features.json'):
        """Load face features from JSON file"""
        try:
            if os.path.exists(features_path):
                with open(features_path, 'r', encoding='utf-8') as f:
                    face_features = json.load(f)
                
                print(f"已載入人臉特徵，共 {len(face_features)} 人")
                for person_id, features in face_features.items():
                    print(f"- {person_id}: {len(features)} 個特徵")
                
                return face_features
            else:
                print(f"找不到人臉特徵文件: {features_path}")
                return {}
        except Exception as e:
            print(f"載入人臉特徵時發生錯誤: {e}")
            return {}
    
    def save_face_features(self, face_features, features_path='face_features.json'):
        """Save face features to JSON file"""
        try:
            with open(features_path, 'w', encoding='utf-8') as f:
                json.dump(face_features, f, ensure_ascii=False, indent=2)
            
            print(f"成功保存人臉特徵到: {features_path}")
            return True
        except Exception as e:
            print(f"保存人臉特徵時發生錯誤: {e}")
            return False
    
    def detect_faces(self, frame):
        """Detect faces in a frame using InsightFace"""
        try:
            start_time = time.time()
            faces = self.face_app.get(frame)
            
            if len(faces) > 0:
                process_time = time.time() - start_time
                if process_time > 0.1:
                    print(f"人臉檢測耗時: {process_time:.4f}秒")
            
            return faces
        except Exception as e:
            print(f"人臉檢測錯誤: {e}")
            return []
    
    def batch_feature_matching(self, query_feature, known_features_dict, top_k=3):
        """Batch calculate feature similarity and return best match"""
        start_time = time.time()
        
        # Prepare batch processing data
        all_features = []
        feature_mapping = []  # For tracking which person each feature belongs to
        
        for person_id, features in known_features_dict.items():
            for feature in features:
                all_features.append(feature)
                feature_mapping.append(person_id)
        
        # Check if there are features to match
        if not all_features:
            return None, 1.0
        
        # Use GPU batch computation if available
        if torch.cuda.is_available():
            # Convert to tensors
            query_tensor = torch.tensor([query_feature], dtype=torch.float32).cuda()
            features_tensor = torch.tensor(all_features, dtype=torch.float32).cuda()
            
            # Calculate normalization
            query_norm = torch.norm(query_tensor, dim=1, keepdim=True)
            features_norm = torch.norm(features_tensor, dim=1, keepdim=True)
            
            # Calculate similarity
            similarity = torch.matmul(
                query_tensor / query_norm, 
                (features_tensor / features_norm).t()
            )
            
            # Get best matches
            similarity_np = similarity.cpu().numpy()[0]
            
            # Find top-k largest values' indices
            top_indices = np.argsort(similarity_np)[-top_k:][::-1]
            
            # Get results
            results = []
            for idx in top_indices:
                person_id = feature_mapping[idx]
                distance = 1.0 - similarity_np[idx]  # Convert to distance
                results.append((person_id, distance))
            
            # Find result with minimum distance
            best_match = min(results, key=lambda x: x[1])
            
            print(f"批量特徵比對耗時: {time.time() - start_time:.4f}秒")
            return best_match
        else:
            # CPU version of matching
            min_distance = float('inf')
            best_match = None
            
            for person_id, features in known_features_dict.items():
                for feature in features:
                    distance = 1.0 - gpu_cosine_similarity(query_feature, feature)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = person_id
            
            print(f"CPU 特徵比對耗時: {time.time() - start_time:.4f}秒")
            return (best_match, min_distance) if best_match else (None, 1.0)
    
    def train_face(self, name=None):
        """Train face features for a person"""
        if name is None:
            name = input("請輸入姓名: ")
        
        print(f"開始訓練 {name} 的人臉特徵...")
        
        # Create training image directory
        os.makedirs("training_images", exist_ok=True)
        person_dir = os.path.join("training_images", name)
        os.makedirs(person_dir, exist_ok=True)
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if not cap.isOpened():
            print("無法開啟攝像頭")
            return
        
        # Collect face images and features
        features = []
        image_count = 0
        max_images = 5
        
        print(f"請面對攝像頭，將收集 {max_images} 張不同角度的人臉圖像...")
        
        while image_count < max_images:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Display instructions
            cv2.putText(
                frame,
                f"收集圖像: {image_count}/{max_images}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                "按空格鍵捕獲人臉",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            
            # Draw detected faces
            if faces:
                for face in faces:
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(
                        frame,
                        (bbox[0], bbox[1]),
                        (bbox[2], bbox[3]),
                        (0, 255, 0),
                        2
                    )
            
            cv2.imshow("人臉訓練", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # 空格鍵
                if faces:
                    face = faces[0]  # 使用第一個檢測到的人臉
                    
                    # 保存人臉圖像
                    bbox = face.bbox.astype(int)
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
            face_features = self.load_face_features()
            
            face_features[name] = features
            
            self.save_face_features(face_features)
            
            print(f"已成功保存 {name} 的 {len(features)} 個人臉特徵")
            return True
        except Exception as e:
            print(f"保存人臉特徵時發生錯誤: {e}")
            return False