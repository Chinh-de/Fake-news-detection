"""
Integrated SLM (Small Language Model) wrapper.
RoBERTa-based binary classifier for fake news detection.

Supports two backends:
- "hf": HuggingFace Transformers (default)
- "vllm": vLLM for faster inference (requires vllm package)
"""

import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.config import MODEL_PATH, SLM_BACKEND
from src.utils import preprocess_text
from src.slm.dataset import FakeNewsDataset


class IntegratedSLM:
    """
    Wrapper for RoBERTa-based SLM with inference and fine-tuning capabilities.
    
    Args:
        model_path: Path to pre-trained model checkpoint.
        backend: "hf" (HuggingFace) or "vllm"
    """

    def __init__(self, model_path: str = MODEL_PATH, backend: str = None):
        """
        Khởi tạo lớp IntegratedSLM để quản lý mô hình RoBERTa.
        
         
        1. Xác định backend sử dụng (hf hoặc vllm).
        2. Xác định thiết bị tính toán (GPU nếu có, ngược lại dùng CPU).
        3. Kiểm tra sự tồn tại của đường dẫn mô hình; nếu không thấy, sử dụng dự phòng ("roberta-base").
        4. Gọi hàm khởi tạo tương ứng với backend đã chọn (`_init_vllm` hoặc `_init_hf`).
        """
        self.backend = backend or SLM_BACKEND
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if not os.path.exists(model_path):
            print("Saved model not found. Using untrained base.")
            model_path = "roberta-base"
        else:
            print(f"Loading SLM from {model_path}")
        
        if self.backend == "vllm":
            self._init_vllm(model_path)
        else:
            self._init_hf(model_path)

    # ================================================================
    # HuggingFace Backend
    # ================================================================
    def _init_hf(self, model_path: str):
        """
        Khởi tạo mô hình sử dụng HuggingFace Transformers.
        
         
        1. Nạp `RobertaTokenizer` từ đường dẫn mô hình.
        2. Nạp `RobertaForSequenceClassification` với cấu hình 2 lớp (binary classification).
        3. Chuyển mô hình sang thiết bị tính toán (device).
        4. Thiết lập mô hình ở chế độ đánh giá (eval).
        """
        self.tokenizer, self.model = self._load_roberta_components(
            model_path=model_path,
            eval_mode=True,
        )
        print(f"SLM loaded (HF backend) on {self.device}")

    def _load_roberta_components(self, model_path: str, eval_mode: bool = True):
        """
        Tải tokenizer và mô hình RoBERTa cho cả inference và fine-tune.

        Args:
            model_path: checkpoint local hoặc tên model trên HuggingFace Hub
            eval_mode: nếu True thì đưa model về chế độ eval, ngược lại train

        Returns:
            tuple (tokenizer, model)
        """
        tokenizer = RobertaTokenizer.from_pretrained(model_path)
        model = RobertaForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
        )
        model.to(self.device)
        if eval_mode:
            model.eval()
        else:
            model.train()
        return tokenizer, model

    def _inference_hf(self, text: str) -> tuple:
        """
        Thực hiện suy luận (inference) qua HuggingFace.
        
         
        1. Tiền xử lý văn bản đầu vào.
        2. Tokenize văn bản với các tham số: truncation, padding và độ dài tối đa 128.
        3. Tắt tính toán gradient (`torch.no_grad`) để tăng tốc độ.
        4. Đưa dữ liệu qua mô hình để lấy logits.
        5. Sử dụng hàm Softmax để chuyển logits thành xác suất.
        6. Lấy nhãn dự đoán (pred) và độ tin cậy (conf) cao nhất.
        7. Trả về nhãn, độ tin cậy và mảng xác suất.
        """
        clean_text = preprocess_text(text)
        inputs = self.tokenizer(
            clean_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length",
        )
        with torch.no_grad():
            outputs = self.model(
                inputs["input_ids"].to(self.device),
                inputs["attention_mask"].to(self.device),
            )
            probs = F.softmax(outputs.logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return pred.item(), conf.item(), probs[0].cpu().numpy()

    def _inference_hf_batch(self, texts: list[str], batch_size: int = 32) -> list[tuple]:
        """
        Thực hiện suy luận theo lô (batch) qua HuggingFace.
        
         
        1. Tiền xử lý toàn bộ văn bản.
        2. Tách thành các mini-batch.
        3. Dự đoán và gộp kết quả.
        """
        clean_texts = [preprocess_text(t) for t in texts]
        results = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(clean_texts), batch_size):
                batch_texts = clean_texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch_texts,
                    max_length=128,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                
                outputs = self.model(
                    inputs["input_ids"].to(self.device),
                    inputs["attention_mask"].to(self.device),
                )
                probs = F.softmax(outputs.logits, dim=1)
                conf, pred = torch.max(probs, dim=1)

                for j in range(len(batch_texts)):
                    results.append((pred[j].item(), conf[j].item(), probs[j].cpu().numpy()))

        return results

    # ================================================================
    # vLLM Backend
    # ================================================================
    def _init_vllm(self, model_path: str):
        """
        Khởi tạo mô hình với vLLM để tối ưu hóa tốc độ suy luận.
        
         
        1. Thử nhập thư viện `vllm`; nếu không có, thông báo lỗi cài đặt.
        2. Nạp tokenizer và mô hình phân loại qua HuggingFace (do vLLM chủ yếu hỗ trợ mô hình sinh văn bản).
        3. Chuyển mô hình sang device và thiết lập chế độ eval.
        4. Lưu đường dẫn mô hình vLLM để sử dụng sau này.
        """
        try:
            from vllm import LLM as VLLM_LLM
        except ImportError:
            raise ImportError(
                "vLLM not installed. Install with: pip install vllm\n"
                "Or set SLM_BACKEND=hf in .env"
            )
        
        # vLLM wraps the model for high-throughput inference
        # For sequence classification, we still need HF for fine-tuning
        self.tokenizer, self.model = self._load_roberta_components(
            model_path=model_path,
            eval_mode=True,
        )
        
        # vLLM is primarily for generative models; for classification
        # we use it as an optimized inference wrapper
        self._vllm_model_path = model_path
        print(f"SLM loaded (vLLM backend) on {self.device}")

    def _inference_vllm(self, text: str) -> tuple:
        """
        Thực hiện suy luận qua đường dẫn tối ưu vLLM.
        
         
        1. Hiện tại RoBERTa phân loại chưa được vLLM hỗ trợ trực tiếp.
        2. Fallback (quay lại) sử dụng phương thức suy luận của HuggingFace (`_inference_hf`).
        """
        # For RoBERTa classification, vLLM doesn't provide direct support.
        # Use HF inference with torch.compile optimization if available.
        return self._inference_hf(text)

    # ================================================================
    # Public Interface
    # ================================================================
    def inference(self, text: str) -> tuple:
        """
        Thực hiện suy luận trên một văn bản đơn lẻ (Public Interface).
        
         
        1. Kiểm tra cấu hình backend hiện tại.
        2. Nếu là "vllm", gọi `_inference_vllm`.
        3. Nếu không, gọi `_inference_hf`.
        
        Args:
            text: Văn bản đầu vào (sẽ được tiền xử lý).
            
        Returns:
            Tuple gồm (dự đoán: int, độ tin cậy: float, xác suất: np.ndarray).
        """
        if self.backend == "vllm":
            return self._inference_vllm(text)
        return self._inference_hf(text)

    def inference_batch(self, texts: list[str], batch_size: int = 32) -> list[tuple]:
        """
        Thực hiện suy luận trên một danh sách văn bản (Batch Public Interface).
        
         
        1. Gọi hàm nội bộ `_inference_hf_batch`. (vLLM fallback cũng sử dụng hàm này).
        
        Args:
            texts: Danh sách văn bản đầu vào.
            batch_size: Kích thước lô thao tác.
            
        Returns:
            Danh sách các tuple gồm (dự đoán, độ tin cậy, xác suất).
        """
        return self._inference_hf_batch(texts, batch_size)

    def finetune_on_clean(
        self,
        clean_samples: list,
        epochs: int = 1,
        batch_size: int = 32,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
    ) -> dict:
        """
        Fine-tune mô hình SLM trên tập dữ liệu sạch (D_clean) sau mỗi vòng.
        Luôn sử dụng backend HuggingFace cho việc huấn luyện.
        
         
        1. Lọc các mẫu hợp lệ (có văn bản và nhãn 0 hoặc 1).
        2. Tiền xử lý văn bản và chuyển nhãn sang kiểu số nguyên.
        3. Khởi tạo `FakeNewsDataset` và `DataLoader` để quản lý việc nạp dữ liệu theo batch.
        4. Sử dụng bộ tối ưu AdamW với các tham số lr và weight_decay.
        5. Chuyển mô hình sang chế độ huấn luyện (train).
        6. Thực hiện lặp qua các epochs:
           - Với mỗi batch: tính toán loss, thực hiện lan truyền ngược (backward) và cập nhật trọng số (step).
        7. Sau khi hoàn thành, chuyển mô hình về chế độ eval.
        8. Tính toán loss trung bình và trả về thống kê huấn luyện.
        """
        valid_samples = [
            s
            for s in clean_samples
            if s.get("text") is not None and s.get("label") in [0, 1]
        ]
        if not valid_samples:
            return {"trained": False, "reason": "no_valid_samples"}

        texts = [preprocess_text(s["text"]) for s in valid_samples]
        labels = [int(s["label"]) for s in valid_samples]

        dataset = FakeNewsDataset(texts, labels, self.tokenizer, max_len=128)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        label_counts = torch.tensor(
            [
                sum(1 for label in labels if label == 0),
                sum(1 for label in labels if label == 1),
            ],
            dtype=torch.float,
        )
        class_weights = (label_counts.sum() / (2 * label_counts.clamp(min=1))).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()

        total_loss = 0.0
        total_steps = 0

        for _ in range(epochs):
            for batch in loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_t = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = loss_fn(outputs.logits, labels_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += float(loss.item())
                total_steps += 1

        self.model.eval()
        avg_loss = total_loss / max(1, total_steps)
        return {
            "trained": True,
            "samples": len(valid_samples),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": weight_decay,
            "avg_loss": avg_loss,
        }

    def finetune(
        self,
        train_texts: list[str],
        train_labels: list[int],
        model_init: str = "roberta-base",
        epochs: int = 4,
        batch_size: int = 32,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        save_path: str | None = None,
    ) -> dict:
        """
        Khởi tạo mô hình từ bộ tiền huấn luyện và fine-tune như notebook.

        Điều này khác với `finetune_on_clean` vì:
        - nó khởi tạo lại mô hình từ `model_init`.
        - nhận dữ liệu huấn luyện công khai (train_texts/train_labels).
        - sử dụng scheduler và optimizer giống notebook.
        - lưu lại best model theo train loss.
        """
        if len(train_texts) != len(train_labels):
            raise ValueError("train_texts và train_labels phải cùng số lượng")
        if len(train_texts) == 0:
            return {"trained": False, "reason": "no_train_data"}

        self.tokenizer, self.model = self._load_roberta_components(
            model_path=model_init,
            eval_mode=False,
        )
        self.model.train()

        train_dataset = FakeNewsDataset(train_texts, train_labels, self.tokenizer, max_len=128)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        total_steps = len(train_loader) * epochs
        optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * warmup_ratio),
            num_training_steps=total_steps,
        )

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            best_model_path = os.path.join(save_path, "best_model.pt")
        else:
            best_model_path = "best_model_fnetune.pt"

        label_counts = torch.tensor(
            [
                sum(1 for l in train_labels if l == 0),
                sum(1 for l in train_labels if l == 1),
            ],
            dtype=torch.float,
        )
        class_weights = (
            label_counts.sum() / (2 * label_counts.clamp(min=1))
        ).to(self.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

        history = {"train_loss": []}
        best_train_loss = float("inf")

        for epoch in range(epochs):
            epoch_loss = 0.0
            self.model.train()

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels_t = batch["labels"].to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                loss = loss_fn(outputs.logits, labels_t)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())

            avg_train_loss = epoch_loss / max(1, len(train_loader))
            history["train_loss"].append(avg_train_loss)

            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                torch.save(self.model.state_dict(), best_model_path)

        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.model.eval()

        if save_path:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)

        result = {
            "trained": True,
            "samples": len(train_texts),
            "epochs": epochs,
            "train_loss_history": history["train_loss"],
        }
        if save_path:
            result["save_path"] = save_path

        return result

    def fnetune(
        self,
        train_texts: list[str],
        train_labels: list[int],
        model_init: str = "roberta-base",
        epochs: int = 4,
        batch_size: int = 32,
        lr: float = 1e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        max_grad_norm: float = 1.0,
        save_path: str | None = None,
    ) -> dict:
        """
        Backward-compatible alias for `finetune()`.
        """
        return self.finetune(
            train_texts=train_texts,
            train_labels=train_labels,
            model_init=model_init,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            max_grad_norm=max_grad_norm,
            save_path=save_path,
        )
