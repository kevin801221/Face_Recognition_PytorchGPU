import os
import re

# 讀取文件
with open('D:/face_recognition_torch/main.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# 替換錯誤的代碼
content = content.replace('langgraph_conversation.ConversationState()', 'ConversationState()')

# 寫回文件
with open('D:/face_recognition_torch/main.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("文件已修復")
