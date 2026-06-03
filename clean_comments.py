import os, glob, re

target_dir = r'd:/Study_space/Ki8/PBL7/MRCD/src'

for filepath in glob.glob(os.path.join(target_dir, '**', '*.py'), recursive=True):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    if ' ' in content:
        print(f'Found in {filepath}')
        lines = content.split('\n')
        new_lines = []
        skip_mode = False
        empty_lines = 0
        for line in lines:
            if ' ' in line:
                skip_mode = True
                continue
            
            if skip_mode:
                if re.match(r'^\s*$', line):
                    # End skip mode on empty line after Flow
                    skip_mode = False
                    continue
                elif re.match(r'^\s*(Args|Returns|Yields|Raises):', line) or line.strip() == '"""':
                    skip_mode = False
                elif re.match(r'^\s*\d+\.', line) or re.match(r'^\s*-', line):
                    continue
                elif re.match(r'^\s*[a-zA-Z]', line):
                    # Sometimes flow steps wrap, let's just skip anything that looks like text during skip mode.
                    continue
                else:
                    skip_mode = False
            
            if not skip_mode:
                new_lines.append(line)
        
        cleaned_content = '\n'.join(new_lines)
        
        # Fixing any messy quotes
        cleaned_content = re.sub(r'\n\s*\n\s*"""', r'\n    """', cleaned_content)
        cleaned_content = re.sub(r'"""\n    """', r'"""', cleaned_content)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
        print(f'Cleaned {filepath}')


import re
import unicodedata

def normalize_unicode(text: str) -> str:
    """Chuẩn hóa Unicode về dạng tổ hợp (NFC)."""
    return unicodedata.normalize('NFC', text)

# ---------- Regex patterns ----------
URL_PATTERN = r"http\S+|www\S+"
TAG_PATTERN = r"\[.*?\]"          # [hook], [quote], [img]...
HTML_PATTERN = r"<.*?>"
MENTION_PATTERN = r"@\w+"
HASHTAG_PATTERN = r"#(\w+)"

# Tối ưu lại EMOJI_PATTERN để bao quát hơn (tránh sót các icon rác)
EMOJI_PATTERN = re.compile(
    r"[\U0001F600-\U0001F64F"  # emoticons
    r"\U0001F300-\U0001F5FF"  # symbols & pictographs
    r"\U0001F680-\U0001F6FF"  # transport & map symbols
    r"\U0001F1E0-\U0001F1FF"  # flags
    r"\U000E0000-\U000E007F"  # tags
    r"\u2600-\u27BF"          # Miscellaneous Symbols / Dingbats (❤️, ✨, ✔️...)
    r"]+",
    flags=re.UNICODE
)

REPEAT_PATTERN = r'(.)\1{2,}' 
def reduce_repeated_chars(text: str) -> str:
    """Giảm lặp ký tự (giữ tối đa 2 lần)."""
    return re.sub(REPEAT_PATTERN, r'\1\1', text)


def clean_text_transformer(text: str) -> str:
    """Tiền xử lý tổng hợp cho các mô hình (PhoBERT). Không xóa dấu câu hợp lệ."""
    if not isinstance(text, str):
        return ""

    # 1. Chuẩn hóa Unicode
    text = normalize_unicode(text)

    # 2. XÓA DẤU CÂU NỘI TỪ (Fix vụ "co.n m.ẹ", "b-ố l-á-o")
    # Chỉ xóa dấu câu (. , - * _ /) nếu nó bị kẹp giữa 2 chữ cái tiếng Việt/English
    text = re.sub(r'(?<=[a-zA-Zà-ỹÀ-Ỹ])([.,\-*_\/])(?=[a-zA-Zà-ỹÀ-Ỹ])', '', text)

    # 3. Loại bỏ mention
    text = re.sub(MENTION_PATTERN, "", text)

    # 4. Xử lý hashtag: giữ phần từ, bỏ #
    text = re.sub(HASHTAG_PATTERN, r'\1', text)

    # 5. Loại bỏ URL, tag, HTML
    text = re.sub(URL_PATTERN, "", text)
    text = re.sub(TAG_PATTERN, "", text)
    text = re.sub(HTML_PATTERN, "", text)

    # 6. Loại bỏ emoji (Đã nâng cấp pattern để sạch hơn)
    text = EMOJI_PATTERN.sub("", text)

    # [BỔ SUNG VÀO BƯỚC 7] XỬ LÝ DẤU CÂU HỢP LỆ (Thay vì xóa sạch)
    # 7.1. Gom các dấu câu lặp lại (Biến "!!!!" thành "!", "???" thành "?")
    text = re.sub(r'([.!?,])\1+', r'\1', text)
    
    # 7.2. Tách dấu câu dính liền chữ (Biến "được.Xong" thành "được. Xong", "học,chơi" thành "học, chơi")
    text = re.sub(r'([.,!?])(?=[^\s])', r'\1 ', text)

    # 8. Giảm lặp kí tự (Ví dụ: "ngonnnn" -> "ngon")
    text = reduce_repeated_chars(text)

    # 9. Chuẩn hóa khoảng trắng phát sinh do các bước xóa ở trên
    text = re.sub(r"\s+", " ", text).strip()

    return text

