# Fake News Detection using MRCD (Multi-Round Collaborative Detection)

Dự án này triển khai khung làm việc **MRCD (Multi-Round Collaborative Detection)** giúp phát hiện tin giả tiếng Việt thông qua sự phối hợp nhiều vòng giữa mô hình ngôn ngữ lớn (LLM) đóng vai trò người đánh giá/nhãn và mô hình ngôn ngữ nhỏ (SLM - PhoBERT) đóng vai trò dự đoán nhanh. Đồng thời kết hợp RAG/Wikipedia để tìm kiếm bằng chứng (evidence) và định nghĩa (knowledge context).

---

## 📌 Cấu trúc thư mục dự án

```text
├── dataset/                     # Thư mục chứa dữ liệu tiếng Việt (train.csv, test.csv, vifactcheck_all.csv)
├── notebooks/                   # Jupyter Notebooks thực nghiệm và hướng dẫn từng bước
│   └── slm-fine-tuning-from-database.ipynb
├── src/                         # Mã nguồn cốt lõi của dự án (Python Modules)
│   ├── config.py                # Cấu hình tập trung thông số pipeline và siêu tham số SLM/LLM
│   ├── utils.py                 # Hàm tiện ích xử lý văn bản, ghi log CSV
│   ├── prompts.py               # Prompt templates cho LLM đánh giá tin thật/giả
│   ├── llm/                     # Xử lý kết nối và gọi API LLM (Qwen, Llama...)
│   ├── retrieval/               # RAG và Wikipedia retrieval engine
│   ├── slm/                     # Định nghĩa mô hình PhoBERT, Dataset, Dataloader
│   └── pipeline/                # Luồng phối hợp đa vòng (runner, selection, finetune, evidence)
├── .env.example                 # File cấu hình biến môi trường mẫu
├── pyproject.toml               # Định nghĩa package metadata và dependencies bổ sung
├── requirements.txt             # Danh sách thư viện Python cần thiết
└── README.md                    # Hướng dẫn này
```

---

## 🛠 Hướng dẫn Cài đặt

### 1. Chuẩn bị môi trường ảo
Khuyến nghị sử dụng Python `3.10` trở lên. Đầu tiên, hãy khởi tạo và kích hoạt môi trường ảo:

**Trên Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Trên Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Cài đặt thư viện (Dependencies)
Bạn có thể cài đặt trực tiếp qua file `requirements.txt`:
```bash
pip install -r requirements.txt
```

Hoặc cài đặt dưới dạng gói cục bộ kèm các nhóm thư viện bổ sung (khuyên dùng khi chạy notebook):
```bash
# Cài đặt chế độ editable cùng các gói hỗ trợ Jupyter Notebook
pip install -e .[notebook]
```

*Nếu bạn có công cụ `uv` (Trình quản lý package siêu nhanh), hãy dùng:*
```bash
uv pip install -r requirements.txt
# Hoặc
uv sync --extra notebook
```

### 3. Cấu hình biến môi trường
Tạo file `.env` từ file mẫu `.env.example`:
```bash
cp .env.example .env
```

Mở file `.env` lên và cấu hình các trường thông tin cần thiết:
*   `HF_TOKEN`: Token HuggingFace của bạn (cần thiết để tải PhoBERT hoặc Llama 3 nếu chúng ở dạng gated).
*   `LLM_MODEL_NAME`: Tên mô hình LLM sử dụng (Ví dụ: `Qwen/Qwen2.5-7B-Instruct` có hỗ trợ tiếng Việt rất tốt).
*   `LLM_ENDPOINT` và `LLM_API_KEY`: Địa chỉ endpoint và API Key nếu bạn sử dụng LLM qua API dịch vụ (như RunPod, TogetherAI, OpenAI...).

---

## 🚀 Hướng dẫn thực thi mã nguồn

Dự án hỗ trợ chạy thông qua cả **Jupyter Notebook** (phù hợp thử nghiệm, fine-tune và quan sát kết quả) lẫn **Python Scripts (src)** (phù hợp chạy tự động hóa hoặc deploy).

### Cách 1: Chạy bằng Jupyter Notebook (notebooks/)
1. Chạy lệnh mở Jupyter Notebook trong thư mục dự án:
   ```bash
   jupyter notebook
   ```
2. Truy cập vào file: `notebooks/slm-fine-tuning-from-database.ipynb`
3. Tiến hành chạy tuần tự từng cell để:
   *   Tải dữ liệu từ thư mục `dataset/`
   *   Cài đặt và thiết lập LLM Handler + SLM (PhoBERT)
   *   Khởi chạy RAG Retrieval thu thập bằng chứng từ Wikipedia & Google Search (nếu cấu hình)
   *   Chạy luồng MRCD Pipeline lặp qua các vòng (định nghĩa trong `src/pipeline/runner.py`)
   *   Fine-tune SLM trên tập dữ liệu được gán nhãn sạch (D_clean) thu được từ mỗi vòng và đánh giá độ chính xác.

### Cách 2: Chạy trực tiếp qua Python Scripts (src/)
Bạn có thể gọi trực tiếp hàm xử lý pipeline từ code Python của mình. Ví dụ, tạo một file `run.py` ở thư mục gốc:

```python
import os
import pandas as pd
from src.pipeline import run_mrcd_pipeline
from src.slm.model import IntegratedSLM
from src.config import MODEL_PATH

# 1. Khởi tạo Mô hình Ngôn ngữ Nhỏ (SLM)
print("Initializing SLM...")
slm = IntegratedSLM(model_name_or_path=MODEL_PATH, num_labels=2)

# 2. Chuẩn bị danh sách bài viết cần thẩm định tin tức
sample_events = [
    "Thủ tướng Chính phủ vừa ký quyết định ban hành kế hoạch phát triển kinh tế xã hội mới.",
    "Báo động giả: Phát hiện sinh vật ngoài hành tinh khổng lồ đổ bộ xuống trung tâm thành phố Hồ Chí Minh sáng nay."
]

# 3. Thực thi Multi-Round Collaborative Detection Pipeline
results = run_mrcd_pipeline(
    events=sample_events,
    slm=slm,
    max_rounds=3,
    confidence_threshold=0.8,
    knowledge_mode="full" # Hoặc "wiki_only"
)

# 4. In kết quả dự đoán
for item in results["results"]:
    status = "Thật" if item["label"] == 0 else "Giả"
    print(f"\nBài viết: {item['text'][:100]}...")
    print(f"-> Phán quyết: {status} (Độ tin cậy SLM: {item['conf_slm']:.4f} ở Vòng {item['round']})")
```

Sau đó chạy lệnh:
```bash
python run.py
```

Kết quả dự đoán chi tiết sẽ được ghi nhận vào file `results.csv` và lịch sử từng vòng được lưu vào file `trace.csv` (định nghĩa theo thiết lập trong `src/config.py`).
