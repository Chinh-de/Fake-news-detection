"""
LLM handler orchestrator.
Supports both External API LLM (default to save RAM/VRAM) and Local LLM (HuggingFace Transformers).
Model configuration is loaded from config.py and environment variables.
"""

import os
import time
import requests

from src.config import (
    LLM_MODEL_NAME,
    LLM_ENDPOINT,
    LLM_API_KEY,
    LLM_MAX_NEW_TOKENS,
    LLM_TEMPERATURE,
    LLM_TOP_P,
)
from src.llm.base import BaseLLM


class ExternalLLM(BaseLLM):
    """
    LLM handler calling an external API (vLLM / OpenAI compatible)
    to save local RAM/VRAM and prevent OOM.
    """

    def __init__(self, model_name: str = None, endpoint: str = None, api_key: str = None):
        self.model_name = model_name or LLM_MODEL_NAME
        self.endpoint = endpoint or LLM_ENDPOINT or os.environ.get("LLM_ENDPOINT", "http://localhost:8000/v1")
        self.api_key = api_key or LLM_API_KEY or os.environ.get("LLM_API_KEY", "EMPTY")

        # Đảm bảo endpoint kết thúc bằng /v1 hoặc /chat/completions phù hợp
        if self.endpoint and not self.endpoint.endswith("/v1") and not self.endpoint.endswith("/v1/"):
            if not self.endpoint.endswith("/"):
                self.endpoint += "/"
            self.endpoint += "v1"

        print(f"Initialized ExternalLLM: endpoint={self.endpoint}, model={self.model_name}")

    def generate_text(
        self, prompt: str, max_output_tokens: int = LLM_MAX_NEW_TOKENS
    ) -> str:
        """
        Gửi prompt đến external API. 
        Điều chỉnh max output tokens dựa trên loại prompt để tăng tốc độ phản hồi.
        """
        prompt_len = len(prompt)
        # Nhận diện prompt phân loại/kiểm chứng có context dài (thường chứa các thẻ XML hoặc văn bản rất dài)
        is_heavy_prompt = "<VERIFIED_REPORTS>" in prompt or "<ENTITY_DEFINITIONS>" in prompt or prompt_len > 1500

        if is_heavy_prompt:
            # Đối với prompt nặng, set max_output_tokens nhỏ lại để tăng tốc phản hồi (phân loại chỉ cần "Thật"/"Giả")
            max_tokens = min(max_output_tokens, 15)
        else:
            max_tokens = max_output_tokens

        headers = {
            "Content-Type": "application/json"
        }
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }

        url = f"{self.endpoint}/chat/completions"
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=120)
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            if content is None:
                return ""
            return content.strip()
        except Exception as e:
            print(f"Error calling external LLM at {url}: {e}")
            raise RuntimeError(f"External LLM call failed: {e}")


# ============================================================
# Singleton Accessor
# ============================================================
_current_llm = None


def get_llm(model_name: str = None, endpoint: str = None, api_key: str = None) -> BaseLLM:
    """
    Lấy hoặc tạo mới singleton Global LLM (chỉ sử dụng ExternalLLM).
    Hỗ trợ truyền động model_name, endpoint, api_key để ghi đè cấu hình.
    """
    global _current_llm
    if _current_llm is None:
        _current_llm = ExternalLLM(
            model_name=model_name,
            endpoint=endpoint,
            api_key=api_key
        )
    return _current_llm