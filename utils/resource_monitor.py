# import psutil
# import threading
# import time

# class ResourceMonitor:
#     """監控和管理系統資源使用"""
#     def __init__(self, target_cpu_percent=70.0):
#         self.target_cpu_percent = target_cpu_percent
#         self.current_cpu_percent = 0.0
#         self.monitoring = False
#         self.monitor_thread = None
        
#     def start_monitoring(self):
#         """開始監控線程"""
#         if not self.monitoring:
#             self.monitoring = True
#             self.monitor_thread = threading.Thread(target=self._monitor_loop)
#             self.monitor_thread.daemon = True
#             self.monitor_thread.start()
            
#     def stop_monitoring(self):
#         """停止監控"""
#         self.monitoring = False
#         if self.monitor_thread:
#             self.monitor_thread.join()
            
#     def _monitor_loop(self):
#         """監控CPU使用率的循環"""
#         while self.monitoring:
#             self.current_cpu_percent = psutil.cpu_percent(interval=1)
#             time.sleep(0.1)
            
#     def get_processing_delay(self):
#         """根據CPU使用率計算處理延遲"""
#         if self.current_cpu_percent > self.target_cpu_percent:
#             return (self.current_cpu_percent - self.target_cpu_percent) / 100.0
#         return 0
        
#     def get_frame_skip_rate(self, base_skip=2):
#         """計算應跳過的幀數"""
#         if self.current_cpu_percent > self.target_cpu_percent:
#             return base_skip + int((self.current_cpu_percent - self.target_cpu_percent) / 10)
#         return base_skip
        
#     def should_process_frame(self, frame_count):
#         """決定是否處理當前幀"""
#         skip_rate = self.get_frame_skip_rate()
#         return frame_count % (skip_rate + 1) == 0
        
#     def __str__(self):
#         return f"CPU使用率: {self.current_cpu_percent:.1f}% (目標: {self.target_cpu_percent}%)"
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import psutil
import threading
import time
import math

class ResourceMonitor:
    """監控和管理系統資源使用"""
    
    def __init__(self, target_cpu_percent=70.0):
        """初始化資源監控器
        
        Args:
            target_cpu_percent: 目標CPU使用率上限
        """
        self.target_cpu_percent = target_cpu_percent
        self.current_cpu_percent = 0.0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """開始監控線程"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_loop(self):
        """監控CPU使用率的循環"""
        while self.monitoring:
            self.current_cpu_percent = psutil.cpu_percent(interval=1.0)
            time.sleep(0.5)  # 更新頻率
            
    def get_processing_delay(self):
        """根據CPU使用率計算處理延遲
        
        Returns:
            float: 處理延遲時間（秒）
        """
        # 當CPU使用率超過目標時，增加延遲
        if self.current_cpu_percent > self.target_cpu_percent:
            # 越接近 100%，延遲越多
            excess = self.current_cpu_percent - self.target_cpu_percent
            # 將過量轉換為 0-1 範圍
            factor = min(excess / (100.0 - self.target_cpu_percent), 1.0)
            # 延遲範圍從 0 到 0.5 秒
            return factor * 0.5
        return 0.0
        
    def get_frame_skip_rate(self, base_skip=2):
        """計算應跳過的幀數
        
        Args:
            base_skip: 基本跳過率
            
        Returns:
            int: 應跳過的幀數
        """
        # 基本跳過率(base_skip)通常為2，表示處理每3幀數據
        
        if self.current_cpu_percent > 90.0:
            # CPU 負載非常高，大幅增加跳過率
            return base_skip + 5  # 處理每 8 幀
        elif self.current_cpu_percent > 80.0:
            # CPU 負載很高，增加跳過率
            return base_skip + 3  # 處理每 6 幀
        elif self.current_cpu_percent > 70.0:
            return base_skip + 2  # 處理每 5 幀
        elif self.current_cpu_percent > 60.0:
            return base_skip + 1  # 處理每 4 幀
        return base_skip  # 默認處理每 3 幀
        
    def should_process_frame(self, frame_count):
        """決定是否處理當前幀
        
        Args:
            frame_count: 當前幀計數
            
        Returns:
            bool: 是否處理當前幀
        """
        skip_rate = self.get_frame_skip_rate()
        return frame_count % (skip_rate + 1) == 0
        
    def get_system_info(self):
        """獲取系統資源信息
        
        Returns:
            dict: 系統資源信息
        """
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu': self.current_cpu_percent,
                'memory': {
                    'total': memory.total,
                    'available': memory.available,
                    'percent': memory.percent
                },
                'disk': {
                    'total': disk.total,
                    'free': disk.free,
                    'percent': disk.percent
                }
            }
        except Exception as e:
            print(f"獲取系統信息錯誤: {e}")
            return {
                'cpu': self.current_cpu_percent,
                'error': str(e)
            }
        
    def __str__(self):
        return f"CPU: {self.current_cpu_percent:.1f}% | Target: {self.target_cpu_percent}%"