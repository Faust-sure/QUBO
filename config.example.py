# -*- coding: utf-8 -*-
"""配置项模板。复制为 config.py 并填入你自己的密钥。"""
# 移动云五岳纪元平台（Kaiwu SDK）
ACCESS_KEY = "your_access_key_here"
SECRET_KEY = "your_secret_key_here"

# 玻色量子专用云平台（可选）
QBOSON_USER_ID = ""
QBOSON_SDK_CODE = ""

# 量子设备 ID
DEVICE_COHERENT_550  = "WuYue-QPU-Qboson-550"
DEVICE_COHERENT_1000 = "WuYue-QPU-Qboson-1000"
DEVICE_SUPERCONDUCTING = "WuYue-QPU-001"
DEVICE_CIM_SIM = "WuYue-QPUSim-CIMSim"
DEFAULT_DEVICE = DEVICE_COHERENT_550

# 精度级别
PRECISION_LEVELS = ["INT2", "INT3", "INT4", "INT6", "INT8", "FP16"]

# 本地路径
CHECKPOINT_DIR = "./checkpoints"
OUTPUT_DIR = "./output"
