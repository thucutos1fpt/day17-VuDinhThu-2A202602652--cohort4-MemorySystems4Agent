# Phân tích kết quả benchmark

`AdvancedAgent` có cross-session recall tốt hơn vì các fact ổn định được trích xuất
và ghi vào `state/profiles/<user>/User.md`. Fact được lưu theo key, vì vậy một câu
đính chính như chuyển từ backend engineer sang MLOps engineer sẽ thay thế giá trị
cũ thay vì tạo hai fact mâu thuẫn. `BaselineAgent` chỉ có lịch sử gắn với một
`thread_id`, nên không thể trả lời các câu hỏi recall ở thread mới.

Ở benchmark chuẩn, Advanced có thể xử lý nhiều prompt token hơn. Đây là chi phí của
việc luôn kèm profile và một phần context ngắn hạn; với hội thoại ngắn, lợi ích
compact chưa đủ để bù chi phí này.

Ở stress benchmark, compact memory tóm tắt phần lịch sử cũ và chỉ giữ một số message
gần nhất. Vì vậy Advanced giảm đáng kể `Prompt tokens processed` so với Baseline,
trong khi persistent profile vẫn bảo toàn tên, nghề nghiệp, nơi ở và style trả lời.
Compact chủ yếu tối ưu token context, không nhất thiết làm giảm số token sinh trực
tiếp trong câu trả lời.

`User.md` tăng kích thước theo số fact bền vững. Lợi ích là recall xuyên phiên;
rủi ro là profile phình to hoặc chứa fact sai. Bài này giảm rủi ro bằng trích xuất
theo pattern có độ chắc chắn cao, bỏ qua câu đùa/nhiễu rõ ràng, và upsert theo key.
Trong hệ thống production, nên bổ sung confidence score, nguồn gốc fact và cơ chế
decay/duyệt lại các fact cũ.
