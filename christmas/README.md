# 🎄 Premium Christmas Concert QR Giveaway

> A beautiful, production-ready landing page for a Christmas concert QR ticket giveaway.  
> Pure HTML/CSS/JavaScript - No framework, no build tools, no backend required.

[![Live Demo](https://img.shields.io/badge/demo-live-success?style=for-the-badge)](.)
[![License](https://img.shields.io/badge/license-MIT-blue?style=for-the-badge)](LICENSE)

---

## ✨ Features

- 🎨 **Premium Design** - Glassmorphism UI with smooth animations
- ❄️ **Snowfall Effect** - Animated canvas snowfall for Christmas atmosphere
- 📱 **Fully Responsive** - Works perfectly on mobile, tablet, and desktop
- 🎁 **Interactive Gift Box** - Animated gift box with sound effects
- 🎫 **QR Code Distribution** - Automatic ticket assignment with QR codes
- 🔒 **Anti-Cheat System** - Browser fingerprinting to prevent duplicate claims
- 🚀 **Zero Dependencies** - Pure vanilla JavaScript, runs anywhere
- 💾 **LocalStorage** - Persistent data without a database

---

## 🚀 Quick Start

### Option 1: Open Directly (Simplest)
1. Download or clone this repository
2. Open `index.html` in any modern web browser
3. Done! The website is ready to use

### Option 2: Using Live Server (Recommended for Development)
1. Install [Live Server extension](https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer) in VS Code
2. Right-click on `index.html`
3. Select "Open with Live Server"

### Option 3: Deploy to GitHub Pages
1. Push this repository to GitHub
2. Go to Settings → Pages
3. Select branch `main` and folder `/root`
4. Your site will be live at `https://username.github.io/repository-name`

---

## 📁 Project Structure

```
.
├── index.html              # Main HTML file (contains all code)
├── README.md               # This file
└── public/
    ├── qr/                 # QR code images for tickets
    │   ├── qr1.png
    │   ├── qr2.png
    │   └── qr3.png
    ├── audio/
    │   └── open-gift.mp3   # Sound effect when opening gift
    └── images/
        └── bg-christmas.jpg # Background image (optional)
```

---

## 🎯 User Flow

### 1. Landing Page (`/`)
- Fullscreen hero with Christmas theme
- Animated snowfall effect
- "Open My Gift" call-to-action button

### 2. Claim Page (`/#claim`)
- **Input Fields:**
  - Nickname (your name)
  - Secret Code (default: `NOEL2025`)
- **Validation:**
  - Wrong code → Shake animation + error message
  - Correct code → Proceed to gift box

### 3. Gift Box Animation
- Interactive animated gift box
- Click to open
- Sound effect plays
- Snow burst animation
- Auto-assign ticket based on browser fingerprint

### 4. Ticket Display (`/#ticket`)
- Shows personalized QR code
- Concert details
- Owner's nickname
- Download QR code button

### 5. Sold Out Page (`/#sold-out`)
- Displayed when all tickets are claimed
- Graceful message with holiday wishes

---

## ⚙️ Configuration

### 1. Change Secret Code
Open `index.html` and find this line (~line 290):
```javascript
const SECRET_CODE = 'NOEL2025';
```
Change `'NOEL2025'` to your desired code.

### 2. Update Concert Information
Find the `tickets` array (~line 291):
```javascript
const tickets = [
  {
    id: 1,
    qr: 'public/qr/qr1.png',
    concert: 'Winter Melody Concert',
    date: '24 Dec 2025',
    location: 'Ho Chi Minh City'
  },
  // Add more tickets here...
];
```
Modify the concert details as needed.

### 3. Add or Remove Tickets
- **Add tickets:** Add more objects to the `tickets` array
- **Remove tickets:** Delete objects from the array
- **Important:** Make sure you have corresponding QR images in `public/qr/`

### 4. Replace QR Codes
1. Generate your QR codes (use [QR Code Generator](https://www.qr-code-generator.com/))
2. Save as PNG files: `qr1.png`, `qr2.png`, etc.
3. Place them in `public/qr/` folder
4. Update the `qr` path in the `tickets` array if needed

### 5. Change Background Image
Replace `public/images/bg-christmas.jpg` with your own image (recommended: 1920x1080px or higher).

### 6. Change Sound Effect
Replace `public/audio/open-gift.mp3` with your own MP3 file.

---

## 🛡️ Anti-Cheat System

The website uses a simple browser fingerprinting system to prevent users from claiming multiple tickets:

**How it works:**
1. Creates a fingerprint from: `User Agent + Screen Width + Screen Height`
2. Hashes the fingerprint with the user's nickname
3. Assigns a ticket based on the hash value
4. Stores the fingerprint in `localStorage`
5. Prevents duplicate claims from the same browser

**Limitations:**
- Can be bypassed by clearing browser data
- Different browsers = different fingerprints
- This is frontend-only and not cryptographically secure
- Suitable for friendly giveaways, not high-security applications

**Reset for Testing:**
Open browser console and run:
```javascript
localStorage.clear();
location.reload();
```

---

## 🎨 Customization

### Colors
The main colors are defined in CSS. Search for:
- `#C81D25` - Christmas Red
- `#FFD700` - Gold
- `#FF6B6B` - Light Red

### Fonts
The project uses **Poppins** from Google Fonts. To change:
1. Find `@import url('https://fonts.googleapis.com/css2?family=Poppins:...')` in `<style>` tag
2. Replace with your preferred Google Font
3. Update `font-family: 'Poppins'` throughout the CSS

### Animations
All animations are defined in the `<style>` section:
- `fadeInUp` - Page entrance animation
- `shake` - Error shake animation
- `bounce` - Gift box bounce
- `pulse` - Pulsing effect
- `shimmer` - Text shimmer

---

## 📱 Browser Support

- ✅ Chrome/Edge (90+)
- ✅ Firefox (88+)
- ✅ Safari (14+)
- ✅ Mobile browsers (iOS Safari, Chrome Mobile)

---

## 🐛 Troubleshooting

### Images not loading
- Check that the `public/` folder path is correct
- Verify image file names match exactly (case-sensitive)

### Sound not playing
- Some browsers block autoplay audio
- User must interact with the page first (click the gift box)
- Check that `public/audio/open-gift.mp3` exists

### Snowfall not showing
- Check browser console for JavaScript errors
- Ensure the `<canvas>` element is present
- Try a different browser

### LocalStorage not working
- Check if browser allows localStorage (not in incognito/private mode)
- Clear localStorage and try again: `localStorage.clear()`

---

## 📝 License

MIT License - Feel free to use this for personal or commercial projects!

---

## 🎁 Credits

Created with ❤️ for the Christmas season 🎄

**Technologies Used:**
- Pure HTML5
- Vanilla JavaScript (ES6+)
- CSS3 (Flexbox, Animations, Glassmorphism)
- Canvas API (for snowfall)

---

## 🌟 Tips for Success

1. **Test on multiple devices** before going live
2. **Generate unique QR codes** for each ticket
3. **Share the secret code** only with intended recipients
4. **Promote on social media** for maximum reach
5. **Monitor claims** by checking localStorage in browser console

---

## 📞 Support

If you encounter any issues or have questions:
1. Check the [Troubleshooting](#-troubleshooting) section
2. Open an issue on GitHub
3. Review the code - it's fully commented and easy to understand!

---

**Merry Christmas! 🎄✨**

