#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import json
import os
from datetime import datetime, timedelta
from config import API_BASE_URL

# 員工資料緩存，避免頻繁 API 調用
employee_cache = {}
cache_expire_time = {}  # 記錄緩存過期時間
CACHE_DURATION = timedelta(hours=6)  # 緩存持續時間（6小時）

def get_employee_data(name, force_refresh=False):
    """從 API 獲取員工資料，帶緩存機制
    
    Args:
        name: 員工名稱
        force_refresh: 是否強制刷新緩存
        
    Returns:
        dict: 員工資料，如果獲取失敗則返回 None
    """
    current_time = datetime.now()
    
    # 檢查緩存是否有效
    if not force_refresh and name in employee_cache:
        if name in cache_expire_time and current_time < cache_expire_time[name]:
            print(f"從緩存獲取 {name} 的資料")
            return employee_cache[name]
    
    try:
        # 發送 API 請求
        response = requests.get(f"{API_BASE_URL}/{name}", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and len(data['data']) > 0:
                employee_data = data['data'][0]
                
                # 更新緩存
                employee_cache[name] = employee_data
                cache_expire_time[name] = current_time + CACHE_DURATION
                
                # 保存到本地文件作為備份
                save_employee_data_to_file(name, employee_data)
                
                print(f"從 API 取得 {name} 的資料")
                return employee_data
            else:
                print(f"API 返回空數據：{name}")
                
                # 嘗試從本地文件加載
                local_data = load_employee_data_from_file(name)
                if local_data:
                    print(f"從本地檔案加載 {name} 的資料")
                    employee_cache[name] = local_data
                    cache_expire_time[name] = current_time + CACHE_DURATION
                    return local_data
                
                return None
        else:
            print(f"API 請求失敗: {response.status_code}")
            
            # 嘗試從本地文件加載
            local_data = load_employee_data_from_file(name)
            if local_data:
                print(f"從本地檔案加載 {name} 的資料")
                employee_cache[name] = local_data
                cache_expire_time[name] = current_time + CACHE_DURATION
                return local_data
            
            return None
        
    except Exception as e:
        print(f"獲取員工資料時發生錯誤: {e}")
        
        # 嘗試從本地文件加載
        local_data = load_employee_data_from_file(name)
        if local_data:
            print(f"從本地檔案加載 {name} 的資料")
            employee_cache[name] = local_data
            cache_expire_time[name] = current_time + CACHE_DURATION
            return local_data
        
        return None

def save_employee_data_to_file(name, data):
    """保存員工資料到本地文件，作為離線備份"""
    try:
        # 創建資料目錄
        os.makedirs('employee_data', exist_ok=True)
        
        # 保存資料
        file_path = os.path.join('employee_data', f"{name}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            
        return True
    except Exception as e:
        print(f"保存員工資料到文件時發生錯誤: {e}")
        return False

def load_employee_data_from_file(name):
    """從本地文件加載員工資料"""
    try:
        file_path = os.path.join('employee_data', f"{name}.json")
        
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return None
    except Exception as e:
        print(f"從文件加載員工資料時發生錯誤: {e}")
        return None

def clear_cache():
    """清空緩存"""
    global employee_cache, cache_expire_time
    employee_cache = {}
    cache_expire_time = {}
    print("已清空員工資料緩存")