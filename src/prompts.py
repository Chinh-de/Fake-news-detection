"""Prompt builders for MRCD Framework (Vietnamese Language Support).

Supports two classification prompt modes:
- wiki_only: chỉ dùng K_wiki (định nghĩa thực thể)
- full: dùng K_wiki + K_fact (các báo cáo đã xác minh)

LLM chỉ trả về "Thật" (Real) hoặc "Giả" (Fake).
"""

import datetime

# ============================================================
# 1. ENTITY & SEARCH QUERY EXTRACTION PROMPTS
# ============================================================

ENTITY_EXTRACTION_SYSTEM_PROMPT = (
    "Bạn là chuyên gia Trích xuất Kiểm chứng Sự kiện cấp cao. Nhiệm vụ của bạn là xử lý văn bản tin tức thô "
    "và tạo ra hai kết quả đồng thời phục vụ cho Hệ thống Truy xuất thông tin (RAG).\n\n"
    
    "NHIỆM VỤ 1: TRUY VẤN TÌM KIẾM (Để Tìm kiếm Bài viết Đối chiếu)\n"
    "- Tạo một truy vấn tìm kiếm duy nhất, ngắn gọn (từ 4 đến 8 từ) tập trung vào cốt lõi sự việc.\n"
    "- QUY TẮC GIỮ TỪ KHÓA CỐT LÕI: BẮT BUỘC phải giữ lại các neo định vị dữ liệu bao gồm: Danh từ riêng (tên đối tượng cụ thể), mốc thời gian, hoặc địa danh xuất hiện trong văn bản gốc. KHÔNG ĐƯỢC lược bỏ vì chúng là chìa khóa để tìm kiếm đối chiếu.\n"
    "- QUY TẮC LỌC: LOẠI BỎ toàn bộ các từ biểu đạt cảm xúc, từ giật gân, phóng đại, từ nối dông dài hoặc các trạng từ thừa.\n"
    "- Sử dụng tiếng Việt có dấu, KHÔNG dùng dấu ngoặc kép (\") hoặc toán tử tìm kiếm.\n\n"
    
    "NHIỆM VỤ 2: CÁC THỰC THỂ WIKIPEDIA CHIẾN LƯỢC (Tối đa 3 thực thể)\n"
    "- Hãy trích xuất các danh từ riêng đại diện cho các thực thể nền tảng xuất hiện trong văn bản: bao gồm Chủ thể (Cơ quan, tổ chức, pháp nhân, nhân vật) hoặc **TÊN RIÊNG CỦA CÁC SỰ KIỆN / BIẾN CỐ / CỘT MỐC THỜI SỰ VÀ LỊCH SỬ**.\n"
    "- QUY TẮC TRÍCH XUẤT ĐỐI CHIẾU:\n"
    "  1. Bắt buộc trích xuất nếu thực thể đó là nguồn phát ngôn, đối tượng hành động, hoặc chịu trách nhiệm chính của thông tin.\n"
    "  2. VẪN TRÍCH XUẤT các thực thể phụ trợ (thương hiệu, hiệp hội, tên sự kiện được viện dẫn) nếu chúng chứa đựng thông tin cốt lõi, đóng vai trò là 'bằng chứng danh tính' hoặc 'neo logic' để hệ thống tra cứu từ điển xem thông tin có bị mâu thuẫn mốc thời gian, địa điểm hoặc sai lệch bối cảnh thực tế hay không.\n"
    "  3. Tuyệt đối không bốc tên các cá nhân đơn lẻ không có tầm ảnh hưởng xã hội. Nếu cá nhân đó không có khả năng sở hữu trang hồ sơ riêng trên Wikipedia, việc trích xuất chắc chắn sẽ gây lỗi tra cứu sai lệch sang một thực thể trùng tên khác.\n"
    "  4. LOẠI BỎ các danh từ chung chung, mang tính đại chúng không có trang định nghĩa bối cảnh riêng trên các hệ thống từ điển tri thức.\n\n"
    
    "ĐỊNH DẠNG ĐẦU RA (QUY TẮC BẮT BUỘC):\n"
    "- Chỉ trả về một đối tượng JSON duy nhất, KHÔNG bao bọc trong các thẻ markdown (như ```json), không giải thích gì thêm.\n"
    "- BẮT BUỘC trường 'query' phải xuất hiện trước trường 'entities'.\n\n"
    
    "Cấu trúc đích bắt buộc:\n"
    '{"query": "chuỗi_truy_vấn_ngắn", "entities": ["thực_thể_1", "thực_thể_2"]}'
)


def build_dual_extraction_prompt(text: str) -> str:
    """
    Xây dựng prompt để trích xuất thực thể và tạo truy vấn tìm kiếm (tiếng Việt).
    """
    return f"{ENTITY_EXTRACTION_SYSTEM_PROMPT}\n\nVăn bản đầu vào: {text}"


def build_entity_extraction_prompt(text: str) -> str:
    """
    Xây dựng prompt CHỈ trích xuất thực thể cho chế độ wiki_only.
    """
    system_prompt = (
        "Bạn là chuyên gia Trích xuất Kiểm chứng Sự kiện cấp cao.\n\n"

        "NHIỆM VỤ: CÁC THỰC THỂ WIKIPEDIA CHIẾN LƯỢC (Tối đa 3 thực thể)\n"
        "- Hãy trích xuất các danh từ riêng đại diện cho các thực thể nền tảng xuất hiện trong văn bản: "
        "bao gồm Chủ thể (Cơ quan, tổ chức, pháp nhân, nhân vật) hoặc TÊN RIÊNG CỦA CÁC SỰ KIỆN / BIẾN CỐ / "
        "CỘT MỐC THỜI SỰ VÀ LỊCH SỬ.\n"

        "- QUY TẮC TRÍCH XUẤT ĐỐI CHIẾU:\n"
        "  1. Bắt buộc trích xuất nếu thực thể đó là nguồn phát ngôn, đối tượng hành động, hoặc chịu trách nhiệm chính của thông tin.\n"
        "  2. VẪN TRÍCH XUẤT các thực thể phụ trợ (thương hiệu, hiệp hội, tên sự kiện được viện dẫn) nếu chúng chứa đựng thông tin cốt lõi, đóng vai trò là 'bằng chứng danh tính' hoặc 'neo logic' để hệ thống tra cứu từ điển xem thông tin có bị mâu thuẫn mốc thời gian, địa điểm hoặc sai lệch bối cảnh thực tế hay không.\n"
        "  3. Tuyệt đối không bốc tên các cá nhân đơn lẻ không có tầm ảnh hưởng xã hội. Nếu cá nhân đó không có khả năng sở hữu trang hồ sơ riêng trên Wikipedia, việc trích xuất chắc chắn sẽ gây lỗi tra cứu sai lệch sang một thực thể trùng tên khác.\n"
        "  4. LOẠI BỎ các danh từ chung chung, mang tính đại chúng không có trang định nghĩa bối cảnh riêng trên các hệ thống từ điển tri thức.\n\n"

        "ĐỊNH DẠNG ĐẦU RA:\n"
        "Chỉ trả về một đối tượng JSON hợp lệ. "
        "KHÔNG bao bọc trong các thẻ markdown, không giải thích gì thêm.\n\n"

        'Cấu trúc đích bắt buộc:\n'
        '{"entities": ["thực_thể_1", "thực_thể_2"]}'
    )

    return f"{system_prompt}\n\nVăn bản đầu vào: {text}"


# ============================================================
# 2. FINAL CLASSIFICATION PROMPTS
# ============================================================

FINAL_CLASSIFICATION_SYSTEM_PROMPT = (
    "Bạn là một chuyên gia phân tích và kiểm chứng tin tức có tư duy sắc bén. "
    "Nhiệm vụ của bạn là đánh giá tính xác thực của <BÀI_VIẾT>.\n\n"
    
    "QUY TRÌNH TƯ DUY 2 BẬC (BẮT BUỘC THEO THỨ TỰ):\n"
    "BẬC 1: ĐỐI CHIẾU BẰNG CHỨNG (RAG)\n"
    "- Đầu tiên, hãy xem kỹ dữ liệu trong <VERIFIED_REPORTS> và <ENTITY_DEFINITIONS>.\n"
    "- Nếu dữ liệu RAG có liên quan và cung cấp đủ thông tin, bạn PHẢI dựa hoàn toàn vào đó để kết luận 'Thật' (Real) hoặc 'Giả' (Fake).\n\n"
    
    "BẬC 2: SUY LUẬN LOGIC & ĐỘ LỆCH THỜI GIAN (Khi RAG thiếu/tin quá mới)\n"
    "- Nếu dữ liệu đối chiếu (RAG) trống hoặc chưa cập nhật kịp các sự kiện mới diễn ra gần đây, hãy tự phân tích bằng logic.\n"
    "- QUY TẮC CẤM ĐOÁN BỪA: Bộ nhớ và kiến thức cũ của bạn có thể đã lỗi thời đối với những thông tin có thể thay đổi theo thời gian (Ví dụ: chức vụ, nhân sự mới được bổ nhiệm, hoặc công nghệ mới vừa ra mắt). TUYỆT ĐỐI không được kết luận bài viết là 'Giả' (Fake) chỉ vì thông tin đó không có trong trí nhớ cũ của bạn khi dữ liệu RAG bị thiếu. Nếu văn phong bài viết nghiêm túc, chính thống, hãy dựa vào tính logic tổng thể hoặc kết luận là cần kiểm chứng thêm.\n\n"
    "- Ví dụ: Một tin tức nói về 'công nghệ bất tử người bằng nước muối' - dù RAG trống, logic thông thường của bạn vẫn phải khẳng định đây là tin 'Giả' (Fake).\n\n"
    
    "ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:\n"
    "Chỉ trả về MỘT từ duy nhất: \"Thật\" (đối với tin thật / Real) hoặc \"Giả\" (đối với tin giả / Fake). Không giải thích, không thêm dấu câu hay bất kỳ ký tự nào khác."
)

FINAL_CLASSIFICATION_USER_PROMPT_TEMPLATE = """MỐC THỜI GIAN HỆ THỐNG HIỆN TẠI: {current_time}
(Lưu ý: Luôn đối chiếu mốc thời gian của bài viết với mốc thời gian hiện tại này để tránh nhầm lẫn về mặt sự kiện thời sự).

DỮ LIỆU ĐỐI CHIẾU (NẾU CÓ):
{knowledge_k}

BÀI VIẾT CẦN PHÂN LOẠI:
Nội dung: "{text_input}"

CÁC VÍ DỤ MẪU ĐỂ HỌC TẬP (VÍ DỤ THAM KHẢO):
{demo_text}

Kết luận (Chỉ ghi "Thật" hoặc "Giả" ):"""


def build_classification_prompt_wiki_only(text: str, knowledge_k: str, demos: list) -> str:
    """
    Xây dựng prompt phân loại chỉ dùng định nghĩa thực thể (wiki_only) cho các bài test.
    """
    return build_classification_prompt(text, knowledge_k, demos, mode="wiki_only")


def build_classification_prompt_full(text: str, knowledge_k: str, demos: list) -> str:
    """
    Xây dựng prompt phân loại dùng cả wiki và fact reports (full) cho các bài test.
    """
    return build_classification_prompt(text, knowledge_k, demos, mode="full")


def build_classification_prompt(
    text: str,
    knowledge_k: str,
    demos: list,
    mode: str = "full",
    round_id: int = 1,
) -> str:
    """
    Xây dựng prompt phân loại bằng tiếng Việt (Round-Aware).
    Hỗ trợ cả mode wiki_only và full để tương thích với các bài test.
    """
    now = datetime.datetime.now()
    current_time_str = now.strftime("Thứ %w, ngày %d/%m/%Y lúc %H:%M:%S")

    # Format background knowledge
    if mode == "wiki_only":
        # Check if already has tags
        if "<ENTITY_DEFINITIONS>" in knowledge_k:
            knowledge_section = knowledge_k
        else:
            knowledge_section = f"<ENTITY_DEFINITIONS>\n{knowledge_k}\n</ENTITY_DEFINITIONS>"
    else:
        # If knowledge_k is already formatted as XML, use it directly
        if "<VERIFIED_REPORTS>" in knowledge_k or "<ENTITY_DEFINITIONS>" in knowledge_k:
            knowledge_section = knowledge_k
        else:
            # Phân tách knowledge_k thô thành VERIFIED_REPORTS và ENTITY_DEFINITIONS dựa trên cấu trúc dòng
            lines = knowledge_k.split("\n")
            verified_reports_blocks = []
            entity_definitions_blocks = []
            
            current_block = []
            current_type = None  # 'report' hoặc 'entity'
            
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("- Title:"):
                    if current_block and current_type:
                        if current_type == 'report':
                            verified_reports_blocks.append("\n".join(current_block))
                        else:
                            entity_definitions_blocks.append("\n".join(current_block))
                    current_block = [line]
                    current_type = 'report'
                elif stripped.startswith("- Entity:"):
                    if current_block and current_type:
                        if current_type == 'report':
                            verified_reports_blocks.append("\n".join(current_block))
                        else:
                            entity_definitions_blocks.append("\n".join(current_block))
                    current_block = [line]
                    current_type = 'entity'
                else:
                    if current_type is not None:
                        current_block.append(line)
                    else:
                        if stripped:
                            current_block.append(line)
            
            if current_block and current_type:
                if current_type == 'report':
                    verified_reports_blocks.append("\n".join(current_block))
                else:
                    entity_definitions_blocks.append("\n".join(current_block))
            
            reports_text = "\n\n".join(verified_reports_blocks).strip()
            entity_text = "\n\n".join(entity_definitions_blocks).strip()
            
            if not reports_text and not entity_text:
                # Fallback nếu không khớp bất kỳ cấu trúc nào
                knowledge_section = f"<VERIFIED_REPORTS>\n{knowledge_k}\n</VERIFIED_REPORTS>"
            else:
                sections = []
                if reports_text:
                    sections.append(f"<VERIFIED_REPORTS>\n{reports_text}\n</VERIFIED_REPORTS>")
                if entity_text:
                    sections.append(f"<ENTITY_DEFINITIONS>\n{entity_text}\n</ENTITY_DEFINITIONS>")
                knowledge_section = "\n\n".join(sections)

    # Cắt ngắn bớt phần tri thức nếu quá dài 10k kí tự
    if len(knowledge_section) > 10000:
        truncated = knowledge_section[:10000] + "..."
        
        # Đóng các thẻ XML nếu chúng được mở nhưng chưa được đóng trong phần bị cắt
        open_tags = []
        if "<VERIFIED_REPORTS>" in truncated and "</VERIFIED_REPORTS>" not in truncated:
            open_tags.append("VERIFIED_REPORTS")
        if "<ENTITY_DEFINITIONS>" in truncated and "</ENTITY_DEFINITIONS>" not in truncated:
            open_tags.append("ENTITY_DEFINITIONS")
            
        for tag in reversed(open_tags):
            truncated += f"</{tag}>"
            
        knowledge_section = truncated

    # 1. FEW-SHOT DEMOS
    if not demos:
        demo_text = "(Không có ví dụ / No examples provided)"
    else:
        demo_text = ""
        for i, demo in enumerate(demos, start=1):
            label_str = demo.get("label", "Chưa xác định")
            text_demo = demo.get("text", "")[:1000].strip()
            source = demo.get("source", "")
            
          

            demo_text += f'\n[Ví dụ {i} / Example {i}]\nNội dung: "{text_demo}..."\nKết luận: {label_str}\n'

    # Sanitize input claim
    sanitized_input = text.replace('"', '\\"').replace('\n', ' ')

    user_prompt = FINAL_CLASSIFICATION_USER_PROMPT_TEMPLATE.format(
        current_time=current_time_str,
        knowledge_k=knowledge_section,
        demo_text=demo_text.strip(),
        text_input=sanitized_input.strip()
    )

    return f"{FINAL_CLASSIFICATION_SYSTEM_PROMPT}\n\n{user_prompt}"
