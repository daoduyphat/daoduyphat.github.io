<div align="center">

# 🌿 Ẩm Thực Chay Hà Nội

![Banner](assets/images/banner.jpg)

**Website hướng dẫn nấu món chay chuẩn nhà hàng, chi tiết từng bước, hình ảnh minh họa và địa chỉ quán chay uy tín tại Hà Nội.**

[![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/HTML)
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[🌐 Demo Live](#) • [📖 Tài liệu](#-tính-năng-nổi-bật) • [🐛 Báo lỗi](https://github.com/daoduyphat/thanh-tinh-web/issues)

</div>

---

## 📋 Tính năng nổi bật

### 🍜 Món Chay
- ✅ **12+ công thức món chay** đa dạng: món nước, món khô, món xào, món chiên, món tráng miệng
- 🖼️ **Hình ảnh chất lượng cao** cho từng món ăn
- 📝 **Công thức chi tiết** với nguyên liệu, các bước thực hiện có đánh số
- ⏱️ **Thời gian chuẩn bị và nấu nướng** rõ ràng
- 🔍 **Tìm kiếm & lọc** theo tên món, nguyên liệu, loại món
- 🎬 **Video hướng dẫn** cho từng món (YouTube embed)
- ✔️ **Checklist nguyên liệu** tương tác để theo dõi

### 🏪 Quán Chay
- 🗺️ **12+ quán chay uy tín** tại Hà Nội
- ⭐ **Đánh giá sao** (hệ thống hiển thị sao đầy/nửa/rỗng)
- 📍 **Địa chỉ chi tiết** với link Google Maps trực tiếp
- 📞 **Thông tin liên hệ**: số điện thoại, link Facebook page
- 💰 **Khoảng giá** và giờ mở cửa
- 🔎 **Tìm kiếm & lọc theo quận** (29 quận/huyện Hà Nội)
- 📱 **Responsive design** với các card cố định, thẳng hàng

### 🎨 Trải nghiệm người dùng
- 🎵 **Nhạc nền tự động** với icon đĩa xoay
- 🔔 **Mõ chay mini** có hiệu ứng âm thanh và animation
- 📱 **Yêu cầu xoay ngang** trên thiết bị di động/tablet (≤1024px)
- ⬆️ **Nút scroll to top** mượt mà
- 🎭 **Animations** tinh tế cho cards và interactions
- 🌈 **Gradient theme** xanh lá (#3BAF4A) và nâu gỗ (#C8A27D)

### 📞 Liên hệ
- 📧 **Form liên hệ** tích hợp
- 👤 **Thông tin developer** với avatar và social links
- 🔗 **Facebook, Instagram, GitHub** links

---

## 🚀 Cài đặt & Chạy

### Yêu cầu
- Trình duyệt web hiện đại (Chrome, Firefox, Safari, Edge)
- Web server tĩnh (tùy chọn cho development)

### Cách 1: Chạy trực tiếp
```bash
# Clone repository
git clone https://github.com/daoduyphat/thanh-tinh-web.git

# Mở file index.html bằng trình duyệt
cd thanh-tinh-web
# Double click index.html hoặc kéo thả vào trình duyệt
```

### Cách 2: Dùng Local Server (Khuyến nghị)

#### Python
```bash
# Python 3
python -m http.server 8000

# Python 2
python -m SimpleHTTPServer 8000
```

#### Node.js
```bash
# Cài http-server
npm install -g http-server

# Chạy server
http-server -p 8000
```

#### VS Code Live Server
1. Cài extension **Live Server**
2. Right-click `index.html` → **Open with Live Server**

Truy cập: `http://localhost:8000`

---

## 📁 Cấu trúc dự án

```
thanh-tinh-web/
├── 📄 index.html              # Trang chủ
├── 📄 recipes.html            # Danh sách món chay
├── 📄 recipe-detail.html      # Chi tiết món chay
├── 📄 restaurants.html        # Danh sách quán chay
├── 📄 restaurant-detail.html  # Chi tiết quán chay
├── 📄 temples.html            # Danh sách chùa (tương lai)
├── 📄 contact.html            # Liên hệ
│
├── 📂 assets/
│   ├── 📂 images/            # Hình ảnh món ăn, quán, banner
│   │   ├── banner.jpg
│   │   ├── avatar.jpg
│   │   ├── pho-chay.jpg
│   │   ├── bun-rieu-chay.jpg
│   │   └── ... (12+ món, 12+ quán)
│   ├── 📂 icons/             # SVG icons
│   │   ├── leaf.svg         # Logo
│   │   ├── disc.svg         # Music player
│   │   ├── mo.svg           # Bell icon
│   │   ├── facebook.svg
│   │   ├── instagram.svg
│   │   └── github.svg
│   └── 📂 sounds/
│       ├── nhacnen.mp3      # Background music
│       └── gomo.mp3         # Bell sound effect
│
├── 📂 css/
│   └── style.css             # Main stylesheet (~1400+ lines)
│
├── 📂 js/
│   ├── script.js             # Main logic (~512 lines)
│   ├── animation.js          # Animation effects
│   └── mo.js                 # Bell mini interaction
│
├── 📂 data/
│   ├── recipes.json          # 12 món chay data
│   ├── restaurants.json      # 12 quán chay data
│   └── temples.json          # Chùa data (tương lai)
│
└── 📄 README.md              # Bạn đang đọc file này
```

---

## 🎨 Stack công nghệ

| Công nghệ | Mô tả |
|-----------|-------|
| **HTML5** | Semantic markup, accessibility (ARIA) |
| **CSS3** | Flexbox, CSS Grid, Animations, Custom Properties |
| **JavaScript (ES6+)** | Vanilla JS, Fetch API, DOM manipulation |
| **JSON** | Data storage cho recipes & restaurants |
| **SVG** | Icons và illustrations |
| **Google Fonts** | Playfair Display (headings), Inter (body) |

### CSS Features
- ✨ CSS Grid Layout cho card system đồng bộ
- 🎭 CSS Animations & Transitions
- 📱 Responsive với media queries
- 🎨 CSS Custom Properties (Variables)
- 🔄 Flexbox cho layout phức tạp

### JavaScript Features
- 🔍 Search & Filter functionality
- 📊 Dynamic rendering từ JSON
- 🎵 Audio control với Web Audio API
- 🖱️ Interactive effects (hover, click)
- 📱 Orientation detection cho mobile

---

## 💾 Data Structure

### Recipes (data/recipes.json)
```json
{
  "id": 1,
  "name": "Phở chay Hà Nội",
  "image": "assets/images/pho-chay.jpg",
  "short_description": "Phở chay thanh đạm...",
  "tags": ["nấm", "đậu phụ", "món nước"],
  "time": "Chuẩn bị 30 phút, nấu 45 phút",
  "ingredients": ["500g nấm hương", "..."],
  "steps": ["**Bước 1:** Làm sạch nấm...", "..."],
  "video_url": "https://www.youtube.com/embed/..."
}
```

### Restaurants (data/restaurants.json)
```json
{
  "id": 1,
  "name": "Nhà hàng chay Tâm Đức",
  "image": "assets/images/tamduc.jpg",
  "address": "14 Dịch Vọng, Cầu Giấy, Hà Nội",
  "district": "Cầu Giấy",
  "phone": "024 3783 1333",
  "page_url": "https://www.facebook.com/...",
  "rating": 4.8,
  "price_range": "50.000đ - 150.000đ",
  "opening_hours": "7:00 - 22:00"
}
```

---

## 🎯 Tính năng đặc biệt

### 📱 Orientation Lock
Website **yêu cầu xoay ngang** trên thiết bị có màn hình ≤1024px (điện thoại, iPad) để đảm bảo trải nghiệm tốt nhất:
- 📵 Portrait mode: Hiển thị overlay với animation xoay điện thoại
- 📱 Landscape mode: Hiển thị website bình thường
- 💻 Desktop (>1024px): Luôn hiển thị bình thường

### ⭐ Rating System
Hệ thống đánh giá sao thông minh:
- ★ Sao đầy (≥0.5)
- ⯨ Nửa sao (0.25-0.49)
- ☆ Sao rỗng
- Hiển thị: "★★★★⯨ 4.4/5"

### 🎵 Music Player
- Icon đĩa xoay khi phát nhạc
- Tự động tiếp tục khi chuyển tab
- Control button với animation

### 🔔 Mõ Chay Mini
- Animation halo khi click
- Âm thanh gõ mõ thực tế
- Số điểm tích đức hiển thị (+1)
- Có thể kéo thả vị trí

---

## 🛠️ Customization

### Thay đổi màu chủ đạo
Chỉnh sửa trong `css/style.css`:
```css
:root {
  --primary-green: #3BAF4A;  /* Xanh lá chủ đạo */
  --wood-brown: #C8A27D;     /* Nâu gỗ */
  --white: #FFFEF9;          /* Trắng kem */
}
```

### Thêm món chay mới
Thêm vào `data/recipes.json`:
```json
{
  "id": 13,
  "name": "Tên món mới",
  "image": "assets/images/mon-moi.jpg",
  ...
}
```

### Thêm quán chay mới
Thêm vào `data/restaurants.json` với đầy đủ thông tin.

---

## 📊 Statistics

- 📄 **7 HTML pages** (index, recipes, restaurants, contact, details...)
- 🎨 **1 CSS file** (~1400+ lines)
- 📜 **3 JavaScript files** (~650+ lines tổng)
- 🍜 **12 món chay** với công thức chi tiết
- 🏪 **12 quán chay** tại Hà Nội
- 🖼️ **30+ images** tối ưu hóa
- 🎵 **2 audio files** (nhạc nền, âm thanh mõ)

---

## 📱 Responsive Breakpoints

| Breakpoint | Thiết bị | Behavior |
|------------|----------|----------|
| **>1024px** | Desktop/Laptop | Full layout, tất cả tính năng |
| **768px - 1024px** | iPad, Tablet | Yêu cầu xoay ngang |
| **<768px** | Mobile | Yêu cầu xoay ngang |
| **Landscape <500px height** | Mobile ngang | Hiển thị bình thường |

---

## 🐛 Known Issues & Roadmap

### Known Issues
- [ ] Video embed cần mạng để load
- [ ] Music autoplay bị block ở một số trình duyệt
- [ ] Lazy loading images chưa tối ưu hoàn toàn

### Roadmap
- [ ] Thêm trang temples.html (danh sách chùa)
- [ ] Dark mode toggle
- [ ] PWA support (offline mode)
- [ ] Multi-language (EN/VI)
- [ ] Backend integration (Node.js/Firebase)
- [ ] User authentication
- [ ] Recipe rating & comments
- [ ] Recipe bookmarking
- [ ] Print-friendly recipe view

---

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Vui lòng:

1. **Fork** repository
2. Tạo **branch** mới (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. Tạo **Pull Request**

### Hướng dẫn đóng góp
- 🐛 Báo lỗi qua [Issues](https://github.com/daoduyphat/thanh-tinh-web/issues)
- 💡 Đề xuất tính năng mới
- 📝 Thêm món chay/quán chay mới
- 🎨 Cải thiện UI/UX
- 📖 Cập nhật documentation

---

## 📄 License

Dự án được phát hành dưới giấy phép **MIT License**. Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 👨‍💻 Tác giả

<div align="center">

### **Dao Duy Phat**

*Developer & Handsome* 😎

[![Email](https://img.shields.io/badge/Email-daoduyphat066%40gmail.com-red?style=flat&logo=gmail)](mailto:daoduyphat066@gmail.com)
[![Facebook](https://img.shields.io/badge/Facebook-dphat20-blue?style=flat&logo=facebook)](https://www.facebook.com/dphat20/)
[![Instagram](https://img.shields.io/badge/Instagram-dao.dphat-E4405F?style=flat&logo=instagram)](https://www.instagram.com/dao.dphat/)
[![GitHub](https://img.shields.io/badge/GitHub-daoduyphat-black?style=flat&logo=github)](https://github.com/daoduyphat)

---

**© 2025 Dao Duy Phat. All rights reserved.**

*Made with ❤️ and ☕ in Hanoi, Vietnam*

</div>
