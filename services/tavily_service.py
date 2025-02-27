#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from datetime import datetime
import requests
import json

class TavilyService:
    """Service for handling Tavily API interactions"""
    
    def __init__(self, api_key=None):
        """Initialize Tavily service with API key"""
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        
        if not self.api_key:
            print("警告: 未找到 Tavily API key。請在 .env 文件中設置 TAVILY_API_KEY。")
            self.initialized = False
        else:
            self.base_url = "https://api.tavily.com/v1"
            self.initialized = True
            print(f"Tavily 搜索服務初始化成功")
    
    def search(self, query, search_depth="basic", include_domains=None, exclude_domains=None, max_results=5):
        """Perform a search using Tavily API
        
        Args:
            query: Search query
            search_depth: 'basic' or 'advanced'
            include_domains: List of domains to include
            exclude_domains: List of domains to exclude
            max_results: Maximum number of results to return
            
        Returns:
            List of search results or None if error
        """
        if not self.initialized:
            print("Tavily 搜索服務未初始化，無法執行搜索")
            return None
            
        url = f"{self.base_url}/search"
        
        headers = {
            "Content-Type": "application/json",
            "X-Tavily-API-Key": self.api_key
        }
        
        # 如果查詢中包含 2025 年，修改查詢以獲取更相關的結果
        original_query = query
        if "2025" in query:
            # 添加更多關鍵詞以幫助找到相關內容
            query = f"{query} future trend prediction forecast 2024 2025"
            print(f"修改查詢: '{original_query}' -> '{query}'")
            
            # 使用高級搜索深度
            search_depth = "advanced"
            max_results = 10  # 增加結果數量
        
        data = {
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results
        }
        
        if include_domains:
            data["include_domains"] = include_domains
            
        if exclude_domains:
            data["exclude_domains"] = exclude_domains
            
        try:
            print(f"開始 Tavily 搜索: '{query}'")
            start_time = time.time()
            
            response = requests.post(url, json=data, headers=headers)
            
            if response.status_code == 200:
                result = response.json()
                process_time = time.time() - start_time
                print(f"Tavily 搜索完成，耗時: {process_time:.2f}秒，找到 {len(result.get('results', []))} 個結果")
                
                # 如果沒有結果，嘗試使用不同的查詢
                if len(result.get('results', [])) == 0 and "2025" in original_query:
                    print("未找到結果，嘗試使用替代查詢...")
                    # 使用更通用的未來預測查詢
                    alternative_query = original_query.replace("2025", "future trends predictions")
                    data["query"] = alternative_query
                    print(f"替代查詢: '{alternative_query}'")
                    
                    response = requests.post(url, json=data, headers=headers)
                    if response.status_code == 200:
                        result = response.json()
                        print(f"替代查詢完成，找到 {len(result.get('results', []))} 個結果")
                
                return result
            else:
                print(f"Tavily 搜索錯誤: {response.status_code}, {response.text}")
                return None
        except Exception as e:
            print(f"Tavily 搜索時出錯: {e}")
            return None
            
    def format_search_results(self, results):
        """Format search results into a readable string
        
        Args:
            results: Search results from Tavily API
            
        Returns:
            Formatted string of search results
        """
        if not results or "results" not in results:
            return "無法獲取搜索結果。"
            
        formatted = "以下是搜索結果：\n\n"
        
        for i, result in enumerate(results["results"], 1):
            title = result.get("title", "無標題")
            content = result.get("content", "無內容")
            url = result.get("url", "")
            
            formatted += f"{i}. {title}\n"
            formatted += f"   {content[:200]}...\n"
            formatted += f"   來源: {url}\n\n"
            
        return formatted
