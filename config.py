#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse

# API settings
API_BASE_URL = "https://inai-hr.jp.ngrok.io/api/employees/search/by-name"

# Resolution settings
RESOLUTION_MAP = {
    '360p': (640, 360),
    '480p': (640, 480),
    '720p': (1280, 720)
}

# Default settings
DEFAULT_MODEL = 'gpt4o'
DEFAULT_CPU_LIMIT = 80.0
DEFAULT_RESOLUTION = '480p'
DEFAULT_SKIP_FRAMES = 2

# Parse command-line arguments
def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='YCM 智能門禁系統')
    
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_MODEL,
        help=f'選擇要使用的 LLM 模型 (預設: {DEFAULT_MODEL})'
    )
    
    parser.add_argument(
        '--cpu-limit',
        type=float,
        default=DEFAULT_CPU_LIMIT,
        help=f'CPU 使用率限制 (預設: {DEFAULT_CPU_LIMIT}%)'
    )
    
    parser.add_argument(
        '--resolution',
        type=str,
        default=DEFAULT_RESOLUTION,
        choices=RESOLUTION_MAP.keys(),
        help=f'攝像頭解析度 (可選: {", ".join(RESOLUTION_MAP.keys())})'
    )
    
    parser.add_argument(
        '--skip-frames',
        type=int,
        default=DEFAULT_SKIP_FRAMES,
        help=f'跳過幀數量 (預設: 每{DEFAULT_SKIP_FRAMES+1}幀處理1幀)'
    )
    
    parser.add_argument(
        '--use-voice',
        action='store_true',
        help='啟用語音功能 (使用 ElevenLabs 進行語音合成)'
    )
    
    args = parser.parse_args()
    
    # Set camera dimensions based on resolution
    global CAMERA_WIDTH, CAMERA_HEIGHT
    CAMERA_WIDTH, CAMERA_HEIGHT = RESOLUTION_MAP.get(args.resolution, (640, 480))
    
    return args

# Initialize camera dimensions with default values
CAMERA_WIDTH, CAMERA_HEIGHT = RESOLUTION_MAP.get(DEFAULT_RESOLUTION, (640, 480))

# Face recognition settings
FACE_DETECTION_SIZE = (160, 160)  # Smaller detection size for better performance
FACE_SIMILARITY_THRESHOLD = 0.3   # Threshold for face matching

# Sleep mode settings
POSITION_THRESHOLD = 50        # Pixel threshold for position change detection
NO_FACE_THRESHOLD = 30         # Frame threshold for no face detection

# Speech recognition settings
SPEECH_TIMEOUT = 2.0           # Timeout for speech input (seconds)