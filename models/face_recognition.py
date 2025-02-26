#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from insightface.app import FaceAnalysis
import os
import json
import time

class FaceRecognitionModel:
    """人臉識別模型封裝類"""
    
    def __init__(self, model_name='buffalo_l', det_size=(320, 320)):
        """初始化人臉識別模型
        
        Args:
            model_name: 模型名稱
            det_size: 檢測大小
        """
        # 檢查 CUDA 可用性
        self.use_cuda = torch.cuda.is_available()
        
        # 選擇執行提供者
        providers = ['CUDAExecutionProvider'] if self.use_cuda else ['CPUExecutionProvider']
        
        # 初始化 InsightFace
        self.face_app = FaceAnalysis(
            name=model_name,
            allowed_modules=['detection', 'recognition'],
            providers=providers
        )
        
        # 準備模型
        self.face_app.prepare(
            ctx_id=0 if self.use_cuda else -1,
            det_size=det_size
        )
        
        # 打印設備信息
        self._print_device_info()
    
    def _print_device_info(self):
        """打印設備信息"""
        print("\n模型初始化:")
        print("=" * 30)
        if self.use_cuda:
            print(f"使用 CUDA 加速")
            print(f"設備名稱: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 版本: {torch.version.cuda}")
            print(f"可用 GPU 數量: {torch.cuda.device_count()}")
            
            # 打印 GPU 詳細信息
            for i in range(torch.cuda.device_count()):
                print(f"\nGPU {i} 信息:")
                print(f"  名稱: {torch.cuda.get_device_name(i)}")
                print(f"  總內存: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
        else:
            print("使用 CPU 模式")
        print("=" * 30)
    
    def detect_faces(self, frame):
        """檢測圖像中的人臉
        
        Args:
            frame: 輸入圖像
            
        Returns:
            list: 檢測到的人臉列表
        """
        start_time = time.time()
        
        try:
            # 檢測人臉
            faces = self.face_app.get(frame)
            
            process_time = time.time() - start_time
            
            # 只有當處理時間過長時才打印
            if process_time > 0.1:
                print(f"人臉檢測耗時: {process_time:.4f}秒")
            
            return faces
        except Exception as e:
            print(f"人臉檢測錯誤: {e}")
            return []
    
    def get_face_embedding(self, face):
        """獲取人臉特徵向量
        
        Args:
            face: InsightFace 檢測到的人臉
            
        Returns:
            np.ndarray: 特徵向量
        """
        return face.normed_embedding
    
    def extract_face_crop(self, frame, face):
        """從圖像中裁剪出人臉
        
        Args:
            frame: 原始圖像
            face: InsightFace 檢測到的人臉
            
        Returns:
            np.ndarray: 裁剪後的人臉圖像
        """
        bbox = face.bbox.astype(int)
        return frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    
    def save_model_info(self, output_file='model_info.json'):
        """保存模型信息到文件
        
        Args:
            output_file: 輸出文件路徑
            
        Returns:
            bool: 是否成功保存
        """
        try:
            info = {
                "model_name": self.face_app.models,
                "device": "CUDA" if self.use_cuda else "CPU",
            }
            
            if self.use_cuda:
                info.update({
                    "device_name": torch.cuda.get_device_name(0),
                    "cuda_version": torch.version.cuda,
                    "gpu_count": torch.cuda.device_count()
                })
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)
                
            return True
        except Exception as e:
            print(f"保存模型信息失敗: {e}")
            return False