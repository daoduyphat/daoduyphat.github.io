# Ẩm Thực Chay Hà Nội

Website tĩnh hướng dẫn nấu món chay chuẩn nhà hàng và giới thiệu các quán chay uy tín tại Hà Nội.

## Cấu trúc thư mục

```
├── index.html                # Trang chủ
├── recipes.html              # Danh sách món chay
├── recipe-detail.html        # Trang chi tiết món chay
├── restaurants.html          # Danh sách quán chay
├── temples.html              # Danh sách chùa
├── contact.html              # Trang liên hệ
├── assets/
│   ├── icons/                # Icon SVG, logo
│   ├── images/               # Hình ảnh món chay, quán chay
│   └── sounds/               # Âm thanh mõ chay
├── css/
│   └── style.css             # File CSS chính
├── js/
│   ├── script.js             # Logic chính: load dữ liệu, render, event, modal
│   ├── animation.js          # Hiệu ứng động khi cuộn trang
│   └── mo.js                 # Logic mõ chay mini
├── data/
│   ├── recipes.json          # Dữ liệu món chay
│   ├── restaurants.json      # Dữ liệu quán chay
│   └── temples.json          # Dữ liệu chùa
```

## Chức năng chính
- Xem công thức món chay chi tiết, hình ảnh, nguyên liệu, các bước, video hướng dẫn.
- Tìm kiếm, lọc món chay theo tên, nguyên liệu, loại món.
- Xem danh sách quán chay, lọc theo quận, tìm kiếm, xem bản đồ.
- Xem danh sách chùa nổi bật tại Hà Nội.
- Gõ mõ chay mini tích đức, có hiệu ứng và âm thanh.
- Gửi liên hệ qua Formspree.

## Hướng dẫn chạy website
1. **Yêu cầu:** Máy tính có Python hoặc bất kỳ phần mềm chạy web server tĩnh.
2. **Chạy server:**
   - Mở PowerShell tại thư mục dự án.
   - Chạy lệnh:
     ```powershell
     python -m http.server 8000
     ```
   - Truy cập trình duyệt: `http://localhost:8000/index.html`
3. **Lưu ý:**
   - Nếu mở trực tiếp file HTML bằng `file://`, một số trình duyệt sẽ chặn truy vấn dữ liệu JSON. Luôn chạy qua server.
   - Nếu gặp lỗi không tải được dữ liệu, kiểm tra debug banner trên trang hoặc mở DevTools (F12).

## Tác giả
- Dao Duy Phat
- Email: daoduyphat066@gmail.com
- Trường Đại học Quốc tế - ĐHQGHN (AIT, VNUIS)

## License
MIT
