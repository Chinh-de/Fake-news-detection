"""
Abstract base class for LLM backends.
"""

from abc import ABC, abstractmethod

from src.config import LLM_MAX_NEW_TOKENS


class BaseLLM(ABC):
    """Abstract base class for Large Language Model backends."""

    @abstractmethod
    def generate_text(
        self, prompt: str, max_output_tokens: int = LLM_MAX_NEW_TOKENS
    ) -> str:
        """
        Sinh văn bản từ prompt (Phương thức trừu tượng).
        
        Flow triển khai (dự kiến trong lớp con):
        1. Nhận prompt văn bản và số lượng token tối đa.
        2. Tokenize prompt và chuyển dữ liệu sang thiết bị xử lý (CPU/GPU).
        3. Thực hiện quá trình suy luận (inference) để sinh ra dãy token kết quả.
        4. Giải mã dãy token thành chuỗi văn bản.
        5. Trả về chuỗi văn bản đã sinh ra.
        
        Args:
            prompt: Chuỗi prompt đầu vào.
            max_output_tokens: Số lượng token tối đa cần sinh.
            
        Returns:
            Vân bản đã được sinh ra.
        """
        pass
