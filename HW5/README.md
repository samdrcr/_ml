# Nexus Agent Framework: Autonomous Task Execution & Security Proxy

**作者 (Author):** 范權榮  
**學號 (Student ID):** 111210557  
**課程 (Course):** 機器學習與人工智慧代理系統 

---

## 專案簡介 | Project Overview

本專案開發了一個基於 **ReAct (Reasoning-Act)** 架構的進階 AI 代理程式。不同於傳統的靜態聊天機器人，Nexus Agent 具備「系統感知」能力，能透過本地端大型語言模型（Ollama）進行複雜的邏輯推理，並在受控環境下執行系統層級的任務（如檔案自動編寫、環境建置與程式碼測試）。

核心設計哲學在於建立一個封閉的 **「思考—行動—觀察」(Think-Act-Observe)** 循環。為了平衡自動化的高效與系統安全性，本專案在 AI 邏輯層與作業系統之間建構了一層**嚴格的安全代理（Security Proxy）**，確保所有具備影響力的指令在執行前必須獲得人類授權，杜絕 AI 誤操作引發的風險。

This project implements a sophisticated **ReAct (Reasoning-Act)** AI Agent framework. Leveraging local LLMs via Ollama, the agent performs multi-step reasoning to execute terminal commands. The architecture maintains a "Human-in-the-Loop" security philosophy via a custom proxy layer, ensuring high autonomy without compromising system integrity.

---

## 核心技術特色 | Core Technical Features

### 1. 安全代理機制 (Security Proxy Layer)
本專案不直接調用系統原始的 `os` 或 `subprocess` 模組，而是透過自定義的 `SecurityProxy` 類別進行動態封裝：
* **指令攔截 (Interception)：** 任何涉及 `os.makedirs` 或 `subprocess.run` 的調用都會被攔截。
* **即時審核 (Real-time Audit)：** 系統會在終端機顯示 AI 意圖執行的完整指令，並強制等待使用者輸入 `[y/N]`。
* **拒絕反饋 (Refusal Feedback)：** 若使用者拒絕執行，程式會將「權限遭拒」的狀態回傳給 AI，引導其修正思路。

### 2. 強健的 XML 標記工具調用 (XML-Based Tool Calling)
傳統 Agent 常因 LLM 生成的 JSON 格式錯誤（如漏掉引號或括號）導致解析失敗。
* **語法定義：** 採用類似 XML 的 `<call name='...'>` 標籤，顯著提升模型遵循指令的準確度。
* **容錯解析：** 使用非同步正則表達式（Regex）提取指令內容，即使模型在標籤外產生冗餘文字，解析器仍能精準鎖定核心參數。

### 3. 非同步本地端 LLM 整合 (Local LLM Integration)
* **後端支持：** 完全相容 Ollama API，預設優化支援 `minimax-m2.5:cloud` 或 `llama3`。
* **高效 I/O：** 基於 `aiohttp` 實現非同步請求，確保在處理複雜推理任務時，主程式不會產生阻塞。

---

## 實作邏輯與流程 | Implementation Logic



代理程式的操作遵循以下標準化循環：

1.  **任務解析 (Task Injection):** 使用者輸入高階目標（例如：「建立一個支援 CRUD 的 Python API」）。
2.  **邏輯推理 (Reasoning):** LLM 分析目標，生成第一步指令（如 `mkdir`）。
3.  **安全審核 (Security Interception):** `SecurityProxy` 攔截指令並等待使用者 `y` 確認。
4.  **環境執行 (Action):** 指令在子程序中運行，擷取其標準輸出 (STDOUT) 與錯誤訊息 (STDERR)。

5.  **反饋觀察 (Observation):** 執行結果作為觀測數據回傳給 LLM。
6.  **遞迴優化 (Iteration):** LLM 根據反饋決定下一步動作，直到任務達成。

---

## 系統源碼 | Source Code

### agent_core.py
```python
import subprocess as _sp
import json
import os as _os
import asyncio
import aiohttp
import re
import sys

class SecurityProxy:
    def __init__(self, original_module, name):
        self.mod = original_module
        self.name = name

    def __getattr__(self, item):
        attr = getattr(self.mod, item)
        if callable(attr):
            def wrapper(*args, **kwargs):
                arg_str = ", ".join(map(str, args))
                print(f"\n[SECURITY ALERT] AI is attempting to call: {self.name}.{item}")
                print(f"Arguments: {arg_str}")
                confirm = input(f"Allow this action? [y/N]: ").lower()
                if confirm == 'y':
                    return attr(*args, **kwargs)
                raise PermissionError(f"Action '{item}' was blocked by the user.")
            return wrapper
        return attr

# Applying Security Layer
os = SecurityProxy(_os, "os")
subprocess = SecurityProxy(_sp, "subprocess")

# --- Configuration ---
BASE_DIR = os.path.expanduser("~/.nexus_agent")
MODEL_NAME = "minimax-m2.5:cloud"

async def query_llm(prompt, system=""):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": MODEL_NAME,
        "prompt": f"{system}\n\n{prompt}" if system else prompt,
        "stream": False
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            data = await resp.json()
            return data.get("response", "").strip()

def run_shell(command):
    try:
        proc = subprocess.run(
            command, 
            shell=True, 
            capture_output=True, 
            text=True, 
            timeout=45
        )
        return f"STDOUT: {proc.stdout}\nSTDERR: {proc.stderr}"
    except Exception as e:
        return f"Execution Error: {str(e)}"

async def process_task(user_input):
    system_msg = (
        "You are an autonomous developer named Nexus. You can interact with the system using:\n"
        "<call name='run_command'>{'command': '...'}</call>\n"
        "Think step-by-step. Provide a final summary when finished."
    )
    
    current_prompt = user_input
    for _ in range(5): # Iteration Limit
        raw_response = await query_llm(current_prompt, system_msg)
        print(f"\n[Nexus Reasoning]: {raw_response}")
        
        match = re.search(r"<call name='(.*?)'>(.*?)</call>", raw_response, re.DOTALL)
        if not match:
            return raw_response
            
        tool_name = match.group(1)
        try:
            # Safe JSON load for tool arguments
            tool_args = json.loads(match.group(2).replace("'", '"'))
            if tool_name == "run_command":
                observation = run_shell(tool_args['command'])
                print(f"[System Observation]: {observation}")
                current_prompt += f"\nObservation: {observation}"
        except Exception as e:
            current_prompt += f"\nInternal Tool Error: {e}"
            
    return "Task exceeded maximum allowed iterations."

def main():
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        
    print(f"--- Nexus Agent Framework Interface ---")
    print(f"Target Model: {MODEL_NAME} | Security Proxy: ENABLED")
    
    while True:
        try:
            cmd = input("\nEnter Task >>> ").strip()
            if cmd.lower() in ['exit', 'quit']: break
            if not cmd: continue
            asyncio.run(process_task(cmd))
        except (KeyboardInterrupt, EOFError):
            break

if __name__ == "__main__":
    main()