# import torch
# import numpy as np

# def gpu_cosine_similarity(a, b):
#     """使用 GPU 加速的餘弦相似度計算"""
#     if isinstance(a, np.ndarray):
#         a = torch.from_numpy(a).cuda()
#     if isinstance(b, np.ndarray):
#         b = torch.from_numpy(b).cuda()
        
#     a_norm = torch.nn.functional.normalize(a, p=2, dim=0)
#     b_norm = torch.nn.functional.normalize(b, p=2, dim=0)
#     return torch.dot(a_norm, b_norm).cpu().item()

# def cosine_similarity(a, b):
#     """CPU 版本的餘弦相似度計算"""
#     return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# def cosine_distance(a, b):
#     """計算餘弦距離"""
#     return 1 - cosine_similarity(a, b)

# def batch_feature_matching(query_feature, known_features_dict, top_k=3):
#     """批量計算特徵相似度並返回最佳匹配"""
#     if not known_features_dict:
#         return None, float('inf')
        
#     try:
#         # 將查詢特徵轉換為GPU tensor
#         if isinstance(query_feature, np.ndarray):
#             query_tensor = torch.from_numpy(query_feature).cuda()
#         else:
#             query_tensor = query_feature.cuda()
            
#         # 正規化查詢特徵
#         query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=0)
        
#         best_match = None
#         min_distance = float('inf')
        
#         # 對每個人的特徵進行比對
#         for person_id, features in known_features_dict.items():
#             for feature in features:
#                 if isinstance(feature, np.ndarray):
#                     feature_tensor = torch.from_numpy(feature).cuda()
#                 else:
#                     feature_tensor = feature.cuda()
                    
#                 # 正規化特徵
#                 feature_norm = torch.nn.functional.normalize(feature_tensor, p=2, dim=0)
                
#                 # 計算餘弦距離
#                 distance = 1 - torch.dot(query_norm, feature_norm).item()
                
#                 if distance < min_distance:
#                     min_distance = distance
#                     best_match = person_id
                    
#         return best_match, min_distance
        
#     except Exception as e:
#         print(f"批量特徵匹配錯誤: {e}")
#         return None, float('inf')
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import time

def gpu_cosine_similarity(a, b):
    """使用 GPU 加速的餘弦相似度計算
    
    Args:
        a: 特徵向量 1
        b: 特徵向量 2
        
    Returns:
        float: 餘弦相似度 (-1 到 1)
    """
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
                if isinstance(a, np.ndarray):
                    a = torch.tensor(a, dtype=torch.float32).cuda()
                elif not isinstance(a, torch.Tensor):
                    a = torch.tensor(a, dtype=torch.float32).cuda()
                elif a.device.type != 'cuda':
                    a = a.cuda()
                
                if isinstance(b, np.ndarray):
                    b = torch.tensor(b, dtype=torch.float32).cuda()
                elif not isinstance(b, torch.Tensor):
                    b = torch.tensor(b, dtype=torch.float32).cuda()
                elif b.device.type != 'cuda':
                    b = b.cuda()
                
                # 計算相似度
                similarity = torch.dot(a, b) / (torch.norm(a) * torch.norm(b))
                return float(similarity.cpu().numpy())  # 轉回 CPU 並轉為 Python float
        else:
            return cosine_similarity(a, b)
    except Exception as e:
        print(f"GPU 計算出錯，切換到 CPU: {e}")
        return cosine_similarity(a, b)

def cosine_similarity(a, b):
    """CPU 版本的餘弦相似度計算
    
    Args:
        a: 特徵向量 1
        b: 特徵向量 2
        
    Returns:
        float: 餘弦相似度 (-1 到 1)
    """
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()
    
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_distance(a, b):
    """計算餘弦距離
    
    Args:
        a: 特徵向量 1
        b: 特徵向量 2
        
    Returns:
        float: 餘弦距離 (0 到 2)
    """
    return 1 - gpu_cosine_similarity(a, b)

def batch_feature_matching(query_feature, known_features_dict, top_k=3):
    """批量計算特徵相似度並返回最佳匹配
    
    Args:
        query_feature: 查詢特徵
        known_features_dict: 已知特徵字典 {person_id: [features...]}
        top_k: 返回的最佳匹配數量
        
    Returns:
        tuple: (best_match_id, min_distance)
    """
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
        if isinstance(query_feature, np.ndarray):
            query_tensor = torch.tensor([query_feature], dtype=torch.float32).cuda()
        elif isinstance(query_feature, list):
            query_tensor = torch.tensor([query_feature], dtype=torch.float32).cuda()
        else:
            query_tensor = query_feature.unsqueeze(0).cuda()
            
        if isinstance(all_features[0], np.ndarray):
            features_tensor = torch.tensor(all_features, dtype=torch.float32).cuda()
        elif isinstance(all_features[0], list):
            features_tensor = torch.tensor(all_features, dtype=torch.float32).cuda()
        else:
            features_tensor = torch.stack(all_features).cuda()
        
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