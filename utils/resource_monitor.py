import psutil
import threading
import time

class ResourceMonitor:
    """監控和管理系統資源使用"""
    def __init__(self, target_cpu_percent=70.0):
        self.target_cpu_percent = target_cpu_percent
        self.current_cpu_percent = 0.0
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """開始監控線程"""
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
    def stop_monitoring(self):
        """停止監控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
            
    def _monitor_loop(self):
        """監控CPU使用率的循環"""
        while self.monitoring:
            self.current_cpu_percent = psutil.cpu_percent(interval=1)
            time.sleep(0.1)
            
    def get_processing_delay(self):
        """根據CPU使用率計算處理延遲"""
        if self.current_cpu_percent > self.target_cpu_percent:
            return (self.current_cpu_percent - self.target_cpu_percent) / 100.0
        return 0
        
    def get_frame_skip_rate(self, base_skip=2):
        """計算應跳過的幀數"""
        if self.current_cpu_percent > self.target_cpu_percent:
            return base_skip + int((self.current_cpu_percent - self.target_cpu_percent) / 10)
        return base_skip
        
    def should_process_frame(self, frame_count):
        """決定是否處理當前幀"""
        skip_rate = self.get_frame_skip_rate()
        return frame_count % (skip_rate + 1) == 0
        
    def __str__(self):
        return f"CPU使用率: {self.current_cpu_percent:.1f}% (目標: {self.target_cpu_percent}%)"
