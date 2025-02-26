import torch
import numpy as np

def gpu_cosine_similarity(a, b):
    """使用 GPU 加速的餘弦相似度計算"""
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(a).cuda()
    if isinstance(b, np.ndarray):
        b = torch.from_numpy(b).cuda()
        
    a_norm = torch.nn.functional.normalize(a, p=2, dim=0)
    b_norm = torch.nn.functional.normalize(b, p=2, dim=0)
    return torch.dot(a_norm, b_norm).cpu().item()

def cosine_similarity(a, b):
    """CPU 版本的餘弦相似度計算"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_distance(a, b):
    """計算餘弦距離"""
    return 1 - cosine_similarity(a, b)

def batch_feature_matching(query_feature, known_features_dict, top_k=3):
    """批量計算特徵相似度並返回最佳匹配"""
    if not known_features_dict:
        return None, float('inf')
        
    try:
        # 將查詢特徵轉換為GPU tensor
        if isinstance(query_feature, np.ndarray):
            query_tensor = torch.from_numpy(query_feature).cuda()
        else:
            query_tensor = query_feature.cuda()
            
        # 正規化查詢特徵
        query_norm = torch.nn.functional.normalize(query_tensor, p=2, dim=0)
        
        best_match = None
        min_distance = float('inf')
        
        # 對每個人的特徵進行比對
        for person_id, features in known_features_dict.items():
            for feature in features:
                if isinstance(feature, np.ndarray):
                    feature_tensor = torch.from_numpy(feature).cuda()
                else:
                    feature_tensor = feature.cuda()
                    
                # 正規化特徵
                feature_norm = torch.nn.functional.normalize(feature_tensor, p=2, dim=0)
                
                # 計算餘弦距離
                distance = 1 - torch.dot(query_norm, feature_norm).item()
                
                if distance < min_distance:
                    min_distance = distance
                    best_match = person_id
                    
        return best_match, min_distance
        
    except Exception as e:
        print(f"批量特徵匹配錯誤: {e}")
        return None, float('inf')
