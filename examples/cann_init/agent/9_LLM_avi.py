import os
import sys
import time
import requests
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()


def test_connection():
    # 1. 获取配置
    api_key = os.getenv("API_KEY")
    api_url = os.getenv("API_URL")
    model = os.getenv("MODEL")

    # 2. 检查配置是否存在
    if not api_key or not api_url:
        print("❌ 错误: .env 文件中未找到 API_KEY 或 API_URL")
        print("请检查文件内容是否正确。")
        return

    # 3. 显示当前配置（密钥脱敏）
    masked_key = api_key[:6] + "*" * 6 + api_key[-4:] if api_key else "None"
    print("-" * 40)
    print(f"📡 API 地址: {api_url}")
    print(f"🤖 模型名称: {model}")
    print(f"🔑 API 密钥: {masked_key}")
    print("-" * 40)
    print("🚀 正在发送测试请求 (Say '1')...")

    # 4. 构造请求头和数据 (OpenAI 兼容格式)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Test connection. Just reply with the number '1'."}
        ],
        "temperature": 0.1
    }

    # 5. 发送请求
    try:
        start_time = time.time()
        # 注意：verify=False 有时用于跳过某些代理的SSL证书问题，但在生产环境不建议
        response = requests.post(api_url, json=payload, headers=headers, timeout=30, verify=False)
        duration = time.time() - start_time

        # 6. 处理响应
        if response.status_code == 200:
            result = response.json()
            # 尝试解析不同厂商可能返回的略微不同的结构
            try:
                content = result['choices'][0]['message']['content']
                print(f"✅ 成功! (耗时: {duration:.2f}s)")
                print(f"💬 回复内容: {content}")
            except (KeyError, IndexError):
                print("⚠️ 连接成功，但无法解析返回内容的结构:")
                print(result)
        else:
            print(f"❌ 请求失败 (状态码: {response.status_code})")
            print("错误信息:", response.text)

    except requests.exceptions.ConnectionError:
        print("❌ 连接错误: 无法连接到服务器。")
        print("👉 如果你使用的是 Google API，请确认你的终端已开启系统代理/VPN。")
    except requests.exceptions.Timeout:
        print("❌ 请求超时: 服务器响应太慢。")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


if __name__ == "__main__":
    test_connection()