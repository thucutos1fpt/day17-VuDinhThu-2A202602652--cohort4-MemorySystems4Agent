# Hướng dẫn chạy bài Day 17

## 1. Chuẩn bị môi trường

Yêu cầu Python 3.11 trở lên. Tại thư mục gốc của project, chạy:

```bash
python -m venv .venv
```

Kích hoạt môi trường ảo:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Cài các thư viện cần thiết:

```bash
pip install pytest tabulate python-dotenv
```

Chế độ offline vẫn chạy được test và benchmark. Nếu muốn dùng LLM thật, cài thêm provider phù hợp và cấu hình API key trong file `.env`.

## 2. Cách chạy

Chạy toàn bộ test:

```bash
pytest src/test_agents.py -v
```

Chạy benchmark để so sánh hai agent:

```bash
python src/benchmark.py
```

## 3. Ý tưởng chính

- `BaselineAgent` chỉ giữ lịch sử trong một `thread_id`, nên sang phiên mới sẽ không nhớ thông tin cũ.
- `AdvancedAgent` lưu các thông tin ổn định của người dùng vào `state/profiles/<user>/User.md`, vì vậy có thể nhớ qua nhiều phiên.
- Khi hội thoại dài, `CompactMemoryManager` tóm tắt phần lịch sử cũ và giữ lại các tin nhắn gần nhất để giảm lượng context phải gửi vào prompt.

Benchmark gồm phần hội thoại thông thường và phần stress context dài. Kết quả cho thấy advanced agent có lợi thế về recall xuyên phiên; compact memory đặc biệt hữu ích khi lịch sử hội thoại lớn vì giúp giảm `Prompt tokens processed`.

## 4. Lưu ý

Thư mục `state/` được tạo khi chạy chương trình và không cần commit. File `User.md` chỉ nên lưu fact rõ ràng, có tính ổn định; khi người dùng đính chính thông tin, giá trị mới sẽ thay thế giá trị cũ.
