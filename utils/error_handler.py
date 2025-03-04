#!/usr/bin/env python
# -*- coding: utf-8 -*-

import traceback
import functools
import logging
from datetime import datetime
import os

# 配置日誌
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("face_recognition")

def log_error(func):
    """記錄函數執行中的錯誤
    
    Args:
        func: 要裝飾的函數
        
    Returns:
        裝飾後的函數
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # 獲取函數名稱
            func_name = func.__name__
            # 記錄錯誤
            logger.error(f"執行 {func_name} 時發生錯誤: {str(e)}")
            logger.error(traceback.format_exc())
            # 重新拋出異常
            raise
    return wrapper

def safe_execute(default_return=None, log_exception=True):
    """安全執行函數，出錯時返回默認值
    
    Args:
        default_return: 出錯時返回的默認值
        log_exception: 是否記錄異常
        
    Returns:
        裝飾器函數
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # 獲取函數名稱
                func_name = func.__name__
                # 記錄錯誤
                if log_exception:
                    logger.error(f"執行 {func_name} 時發生錯誤: {str(e)}")
                    logger.error(traceback.format_exc())
                # 返回默認值
                return default_return
        return wrapper
    return decorator

def log_execution_time(func):
    """記錄函數執行時間
    
    Args:
        func: 要裝飾的函數
        
    Returns:
        裝飾後的函數
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        end_time = datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        logger.info(f"函數 {func.__name__} 執行時間: {execution_time:.4f} 秒")
        return result
    return wrapper
