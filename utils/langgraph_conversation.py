#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
from typing import Dict, Any, List, Optional, Tuple, Literal
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langgraph.graph import StateGraph, END

from utils.error_handler import log_error, safe_execute, log_execution_time

class ConversationState:
    """對話狀態類，用於在 LangGraph 中傳遞狀態"""
    
    def __init__(self):
        self.messages = []
        self.employee_data = {}
        self.current_user = None
        self.audio_buffer = None
        self.search_results = None
        self.should_search = False
        self.conversation_summary = None
    
    def to_dict(self) -> Dict[str, Any]:
        """將狀態轉換為字典"""
        return {
            "messages": self.messages,
            "employee_data": self.employee_data,
            "current_user": self.current_user,
            "audio_buffer": self.audio_buffer,
            "search_results": self.search_results,
            "should_search": self.should_search,
            "conversation_summary": self.conversation_summary
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationState":
        """從字典創建狀態"""
        state = cls()
        state.messages = data.get("messages", [])
        state.employee_data = data.get("employee_data", {})
        state.current_user = data.get("current_user")
        state.audio_buffer = data.get("audio_buffer")
        state.search_results = data.get("search_results")
        state.should_search = data.get("should_search", False)
        state.conversation_summary = data.get("conversation_summary")
        return state

class LangGraphConversation:
    """基於 LangGraph 的對話管理器"""
    
    def __init__(self, model_name="gpt-4o", api_key=None):
        """初始化對話管理器
        
        Args:
            model_name: 使用的模型名稱
            api_key: OpenAI API 密鑰，如果為 None，則從環境變數獲取
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("警告: 未找到 OpenAI API key。請在 .env 文件中設置 OPENAI_API_KEY。")
            self.initialized = False
            return
            
        try:
            # 初始化 LLM
            self.llm = ChatOpenAI(
                model=model_name,
                temperature=0.7,
                api_key=self.api_key
            )
            
            # 測試 LLM 連接
            test_response = self.llm.invoke([SystemMessage(content="回應 'OK' 來測試連接")])
            print(f"LLM 連接測試結果: {test_response.content}")
            
            # 創建對話圖
            self.graph = self._build_conversation_graph()
            
            self.initialized = True
            print(f"LangGraph 對話管理器初始化成功，使用模型: {model_name}")
        except Exception as e:
            print(f"LangGraph 對話管理器初始化失敗: {e}")
            import traceback
            traceback.print_exc()
            self.initialized = False
    
    def _build_conversation_graph(self) -> StateGraph:
        """構建對話圖
        
        Returns:
            StateGraph: LangGraph 狀態圖
        """
        # 創建狀態圖
        workflow = StateGraph(ConversationState)
        
        # 添加節點
        workflow.add_node("process_input", self._process_input)
        workflow.add_node("check_search_intent", self._check_search_intent)
        workflow.add_node("perform_search", self._perform_search)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("generate_audio", self._generate_audio)
        
        # 設置入口節點
        workflow.set_entry_point("process_input")
        
        # 連接節點
        workflow.add_edge("process_input", "check_search_intent")
        
        # 根據搜索意圖決定是否執行搜索
        workflow.add_conditional_edges(
            "check_search_intent",
            self._should_search,
            {
                True: "perform_search",
                False: "generate_response"
            }
        )
        
        # 搜索完成後生成回應
        workflow.add_edge("perform_search", "generate_response")
        
        # 生成回應後生成語音
        workflow.add_edge("generate_response", "generate_audio")
        
        # 語音生成後結束
        workflow.add_edge("generate_audio", END)
        
        # 編譯圖
        print("編譯對話圖...")
        return workflow.compile()
    
    @safe_execute
    def _process_input(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """處理用戶輸入
        
        Args:
            state: 當前狀態
            
        Returns:
            Dict[str, Any]: 更新後的狀態
        """
        state_obj = ConversationState.from_dict(state)
        
        # 如果沒有消息，添加一個系統消息以生成問候語
        if not state_obj.messages:
            print("沒有消息，添加系統消息以生成問候語")
            state_obj.messages.append({
                "role": "system",
                "content": "請生成一個簡短的問候語來開始對話。",
                "timestamp": datetime.now().isoformat()
            })
        
        return state_obj.to_dict()
    
    @safe_execute
    def _check_search_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """檢查用戶是否有搜索意圖
        
        Args:
            state: 當前狀態
            
        Returns:
            Dict[str, Any]: 更新後的狀態
        """
        state_obj = ConversationState.from_dict(state)
        
        if not state_obj.messages:
            return state_obj.to_dict()
            
        last_message = state_obj.messages[-1]
        if last_message["role"] != "user":
            return state_obj.to_dict()
            
        user_input = last_message["content"].lower()
        
        # 檢查是否包含搜索關鍵詞
        search_keywords = [
            "搜索", "查詢", "查找", "找一下", "幫我找", "搜尋", 
            "最新", "新聞", "資訊", "信息", "最近", "查看", 
            "了解", "告訴我關於", "什麼是", "怎麼樣"
        ]
        
        # 檢查是否包含時間關鍵詞，這通常表示需要最新信息
        time_keywords = ["今天", "昨天", "最近", "現在", "目前", "當前", "這幾天", "這個月"]
        
        # 檢查是否包含問題關鍵詞
        question_keywords = ["什麼", "如何", "為什麼", "怎麼", "哪裡", "誰", "何時", "多少"]
        
        # 檢查是否是一個問句
        is_question = "?" in user_input or "？" in user_input or any(kw in user_input for kw in question_keywords)
        
        # 檢查是否包含搜索關鍵詞或時間關鍵詞
        has_search_intent = any(kw in user_input for kw in search_keywords)
        has_time_intent = any(kw in user_input for kw in time_keywords)
        
        # 如果是問句並且包含搜索關鍵詞或時間關鍵詞，則認為有搜索意圖
        state_obj.should_search = is_question and (has_search_intent or has_time_intent)
        
        print(f"檢查搜索意圖: '{user_input}' -> {state_obj.should_search} (問句: {is_question}, 搜索關鍵詞: {has_search_intent}, 時間關鍵詞: {has_time_intent})")
        
        # 強制啟用搜索功能進行測試
        state_obj.should_search = True
        print("注意: 暫時強制啟用搜索功能進行測試")
        
        return state_obj.to_dict()
    
    @safe_execute
    def _should_search(self, state: Dict[str, Any]) -> bool:
        """決定是否應該執行搜索
        
        Args:
            state: 當前狀態
            
        Returns:
            bool: 是否應該搜索
        """
        return state.get("should_search", False)
    
    @safe_execute
    def _perform_search(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """執行網絡搜索
        
        Args:
            state: 當前狀態
            
        Returns:
            Dict[str, Any]: 更新後的狀態
        """
        from services.tavily_service import TavilyService
        
        state_obj = ConversationState.from_dict(state)
        
        if not state_obj.messages:
            return state_obj.to_dict()
            
        last_message = state_obj.messages[-1]
        if last_message["role"] != "user":
            return state_obj.to_dict()
            
        user_input = last_message["content"]
        
        # 使用 Tavily 進行搜索
        tavily_service = TavilyService()
        if not tavily_service.initialized:
            print("Tavily 服務未初始化，無法執行搜索")
            state_obj.search_results = "無法執行搜索，搜索服務未初始化。"
            return state_obj.to_dict()
            
        print(f"執行搜索: {user_input}")
        search_results = tavily_service.search(user_input)
        
        if search_results:
            # 格式化搜索結果
            formatted_results = tavily_service.format_search_results(search_results)
            state_obj.search_results = formatted_results
            print(f"搜索完成，找到 {len(search_results.get('results', []))} 個結果")
        else:
            state_obj.search_results = "搜索未返回任何結果。"
            print("搜索未返回任何結果")
        
        return state_obj.to_dict()
    
    @safe_execute
    def _generate_response(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成回應
        
        Args:
            state: 當前狀態
            
        Returns:
            Dict[str, Any]: 更新後的狀態
        """
        state_obj = ConversationState.from_dict(state)
        
        if not state_obj.messages:
            return state_obj.to_dict()
            
        # 構建提示
        system_prompt = self._build_system_prompt(state_obj)
        messages = [SystemMessage(content=system_prompt)]
        
        # 添加對話歷史
        for msg in state_obj.messages:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
            elif msg["role"] == "system" and len(state_obj.messages) == 1:
                # 如果只有一條系統消息，這是初始對話的情況
                # 不需要添加到 messages 中，因為我們已經有了 system_prompt
                pass
        
        # 如果有搜索結果，添加到提示中
        if state_obj.search_results:
            messages.append(SystemMessage(content=f"以下是關於用戶最近問題的搜索結果，請使用這些信息來幫助回答：\n\n{state_obj.search_results}"))
        
        print("生成回應...")
        response = self.llm.invoke(messages)
        
        # 添加助手回應到消息歷史
        state_obj.messages.append({
            "role": "assistant",
            "content": response.content,
            "timestamp": datetime.now().isoformat()
        })
        
        return state_obj.to_dict()
    
    @safe_execute
    def _generate_audio(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """生成語音
        
        Args:
            state: 當前狀態
            
        Returns:
            Dict[str, Any]: 更新後的狀態
        """
        # 注意：我們不在這裡播放音頻，而是返回音頻路徑，由主程序決定是否播放
        from utils.elevenlabs_tts import ElevenLabsTTS
        
        state_obj = ConversationState.from_dict(state)
        
        if not state_obj.messages:
            return state_obj.to_dict()
            
        last_message = state_obj.messages[-1]
        if last_message["role"] != "assistant":
            return state_obj.to_dict()
            
        assistant_response = last_message["content"]
        
        # 初始化 TTS 服務
        tts = ElevenLabsTTS()
        if not tts.initialized:
            print("ElevenLabs TTS 服務未初始化，無法生成語音")
            return state_obj.to_dict()
            
        # 生成語音
        print("生成語音...")
        audio_path = tts.synthesize_speech(assistant_response)
        if audio_path:
            print(f"語音生成成功: {audio_path}")
            state_obj.audio_buffer = audio_path
        else:
            print("語音生成失敗")
            
        return state_obj.to_dict()
    
    def _build_system_prompt(self, state: ConversationState) -> str:
        """構建系統提示
        
        Args:
            state: 當前狀態
            
        Returns:
            str: 系統提示
        """
        prompt = "你是 YCM 館長，一個友善、專業的智能助手。"
        
        # 添加用戶信息
        if state.current_user:
            prompt += f"\n\n你正在與 {state.current_user} 對話。"
            
        # 添加員工數據
        if state.employee_data:
            employee = state.employee_data
            prompt += f"\n\n以下是關於 {employee.get('name', '用戶')} 的資訊："
            
            if all(key in employee for key in ['name', 'chinese_name', 'department', 'position']):
                prompt += f"""
                
                基本資料：
                - 中文名字：{employee.get('chinese_name', '未提供')}
                - 部門：{employee.get('department', '未提供')}
                - 職位：{employee.get('position', '未提供')}
                - 工作年資：{employee.get('total_years_experience', '未提供')} 年
                """
                
                if 'technical_skills' in employee and employee['technical_skills']:
                    prompt += f"\n專業技能：\n{', '.join(employee['technical_skills'])}"
                    
                if 'interests' in employee and employee['interests']:
                    prompt += f"\n\n興趣愛好：\n{', '.join(employee['interests'])}"
                    
                if 'certificates' in employee and employee['certificates']:
                    prompt += "\n\n證書：\n"
                    prompt += "\n".join([f"- {cert['name']} (由 {cert['issuing_organization']} 頒發)" 
                                    for cert in employee['certificates']])
                    
                if 'work_experiences' in employee and employee['work_experiences']:
                    prompt += "\n\n工作經驗：\n"
                    prompt += "\n".join([f"- {exp['company_name']}: {exp['position']} ({exp['description']})" 
                                    for exp in employee['work_experiences']])
        
        # 添加對話指導
        prompt += """

        你應該：
        1. 保持專業但友善的態度
        2. 給出簡短的回應，不要太長
        3. 如果被問到不知道的問題，誠實地說不知道
        4. 如果有搜索結果，使用這些信息來提供最新、最準確的答案
        """
        
        # 添加對話摘要（如果有）
        if state.conversation_summary:
            prompt += f"\n\n對話摘要：\n{state.conversation_summary}"
            
        return prompt
    
    def add_user_message(self, state: Dict[str, Any], message: str) -> Dict[str, Any]:
        """添加用戶消息到狀態
        
        Args:
            state: 當前狀態
            message: 用戶消息
            
        Returns:
            Dict[str, Any]: 更新後的狀態
        """
        state_obj = ConversationState.from_dict(state)
        
        state_obj.messages.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        return state_obj.to_dict()
    
    @log_execution_time
    def process_message(self, message: str, employee_data: Dict[str, Any] = None, current_user: str = None, is_first_chat: bool = False) -> Tuple[str, Optional[str]]:
        """處理用戶消息並生成回應
        
        Args:
            message: 用戶消息
            employee_data: 員工數據
            current_user: 當前用戶
            is_first_chat: 是否是初始對話
            
        Returns:
            Tuple[str, Optional[str]]: 回應文本和語音文件路徑（如果有）
        """
        if not self.initialized:
            print("LangGraph 對話管理器未初始化，無法處理消息")
            return "抱歉，對話服務暫時不可用。", None
            
        try:
            print(f"處理消息: '{message}'")
            
            # 創建初始狀態
            state = ConversationState()
            state.employee_data = employee_data or {}
            state.current_user = current_user
            
            # 如果是初始對話，添加系統消息
            if is_first_chat:
                print("這是初始對話")
                # 不添加用戶消息，讓系統直接生成問候語
                pass
            else:
                # 添加用戶消息
                state = self.add_user_message(state.to_dict(), message)
                
            # 運行對話圖
            print("運行對話圖...")
            result = self.graph.invoke(state)
            
            # 提取回應和音頻
            result_obj = ConversationState.from_dict(result)
            
            if not result_obj.messages:
                return "抱歉，無法生成回應。", None
                
            # 獲取最後一個助手消息
            assistant_messages = [msg for msg in result_obj.messages if msg["role"] == "assistant"]
            if not assistant_messages:
                return "抱歉，無法生成回應。", None
                
            last_assistant_message = assistant_messages[-1]
            response = last_assistant_message["content"]
            
            return response, result_obj.audio_buffer
            
        except Exception as e:
            print(f"處理消息時出錯: {e}")
            import traceback
            traceback.print_exc()
            return f"處理消息時出錯: {e}", None
