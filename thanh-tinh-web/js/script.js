// Nút tắt/mở nhạc nền với localStorage
window.toggleMusic = function() {
  var audio = document.getElementById('musicBg');
  var disc = document.getElementById('musicDiscIcon');
  var status = document.getElementById('musicDiscStatus');
  if (!audio) return;
  if (audio.paused) {
    audio.removeAttribute('data-user-paused');
    audio.play();
    if (disc) disc.classList.add('spinning');
    if (status) {
      status.textContent = 'Đang phát';
      status.classList.add('show');
    }
    localStorage.setItem('musicPlaying', 'true');
    localStorage.setItem('musicCurrentTime', '0');
  } else {
    audio.setAttribute('data-user-paused', 'true');
    audio.pause();
    if (disc) {
      disc.classList.remove('spinning');
      disc.style.animation = 'none';
    }
    if (status) {
      status.textContent = 'Mở nhạc';
      status.classList.add('show');
    }
    localStorage.setItem('musicPlaying', 'false');
  }
}

// Tự động phát nhạc khi vào trang nếu đang bật
document.addEventListener('DOMContentLoaded', function() {
  var audio = document.getElementById('musicBg');
  
  if (audio) {
    // Kiểm tra trạng thái nhạc từ localStorage
    var musicPlaying = localStorage.getItem('musicPlaying');
    var musicTime = parseFloat(localStorage.getItem('musicCurrentTime') || '0');
    
    // UI đã được set bởi inline script, chỉ cần phát nhạc
    if (musicPlaying === 'true') {
      audio.currentTime = musicTime;
      audio.play().catch(function(err) {
        console.log('Autoplay bị chặn:', err);
        // Nếu autoplay bị chặn, reset UI
        var disc = document.getElementById('musicDiscIcon');
        var status = document.getElementById('musicDiscStatus');
        if (disc) {
          disc.classList.remove('spinning');
          disc.style.animation = 'none';
        }
        if (status) {
          status.textContent = 'Mở nhạc';
          status.classList.add('show');
        }
      });
    }
    
    // Lưu thời gian hiện tại của nhạc
    audio.addEventListener('timeupdate', function() {
      if (!audio.paused) {
        localStorage.setItem('musicCurrentTime', audio.currentTime.toString());
      }
    });
    
    // Ngăn trình duyệt tự động dừng nhạc khi chuyển tab
    audio.addEventListener('pause', function(e) {
      if (!audio.hasAttribute('data-user-paused')) {
        setTimeout(function() {
          if (audio.paused && !audio.hasAttribute('data-user-paused')) {
            audio.play().catch(function() {});
          }
        }, 100);
      }
    });
  }
});
// Load restaurant detail page
function loadRestaurantDetail() {
  const params = new URLSearchParams(window.location.search);
  const id = params.get('id');
  
  if (!id) {
    window.location.href = 'restaurants.html';
    return;
  }

  fetch('data/restaurants.json')
    .then(res => res.json())
    .then(data => {
      const container = document.getElementById('restaurantDetailContent');
      const restaurant = data.find(r => String(r.id) === String(id));
      
      if (!restaurant) {
        container.innerHTML = '<h2>Không tìm thấy quán chay này.</h2><p>Vui lòng chọn lại từ danh sách quán chay.</p>';
        return;
      }
      
      container.innerHTML = `
        <img src="${restaurant.image}" alt="${restaurant.name}" loading="lazy" style="width:100%;max-width:320px;border-radius:10px;margin-bottom:16px;display:block;margin-left:auto;margin-right:auto;">
        <h2 style="margin-top:0">${restaurant.name}</h2>
        <div class="address">${restaurant.address}</div>
        <div class="price">${restaurant.price_range}</div>
        <div class="desc">${restaurant.description}</div>
        <div class="hours">Giờ mở cửa: ${restaurant.opening_hours}</div>
        <div style="display:flex;justify-content:center;margin-top:24px;gap:12px;">
          <a href="restaurants.html" class="btn btn-secondary" style="min-width:180px;">Quay lại danh sách quán chay</a>
          <a href="https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(restaurant.address)}" target="_blank" rel="noopener" class="btn btn-primary" style="min-width:180px;">Xem bản đồ</a>
        </div>
      `;
      lazyLoadImages();
    })
    .catch(err => {
      const container = document.getElementById('restaurant-detail');
      container.innerHTML = `
        <h2>Không thể tải dữ liệu quán chay</h2>
        <p>Vui lòng thử lại sau.</p>
      `;
    });
}

// script.js - Xử lý logic, load dữ liệu, bộ lọc, tìm kiếm, lazy load, scroll-to-top

// Lazy load images
function lazyLoadImages() {
  const images = document.querySelectorAll('img[loading="lazy"]');
  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries, obs) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          const img = entry.target;
          img.src = img.dataset.src || img.src;
          obs.unobserve(img);
        }
      });
    });
    images.forEach(img => observer.observe(img));
  }
}

document.addEventListener('DOMContentLoaded', () => {
  lazyLoadImages();
  handleScrollToTop();
  if (document.getElementById('featured-recipes-list')) {
    loadFeaturedRecipes();
  }
  if (document.getElementById('featured-restaurants-list')) {
    loadFeaturedRestaurants();
  }
  if (document.getElementById('recipesGrid') && !document.getElementById('searchInput')) {
    loadRecipesAndRestaurants();
  } else {
    if (document.getElementById('recipesGrid') && document.getElementById('searchInput')) {
      loadRecipesPage();
    }
    if (document.getElementById('recipe-detail')) {
      loadRecipeDetail();
    }
    if (document.getElementById('restaurant-detail')) {
      loadRestaurantDetail();
    }
    if (document.getElementById('restaurantsList')) {
      loadRestaurantsPage();
    }
    if (document.getElementById('templesGrid')) {
      loadTemplesPage();
    }
  }
});
// Hàm render card quán chay gọn, có data-id
function renderRestaurantCard(restaurant) {
  const fullStars = Math.floor(restaurant.rating);
  const hasHalfStar = (restaurant.rating % 1) >= 0.5;
  const emptyStars = 5 - fullStars - (hasHalfStar ? 1 : 0);
  
  let starsHtml = '★'.repeat(fullStars);
  if (hasHalfStar) starsHtml += '⯨';
  starsHtml += '☆'.repeat(emptyStars);
  
  const mapUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(restaurant.address)}`;
  const phone = restaurant.phone || 'Đang cập nhật';
  const hours = restaurant.opening_hours.replace(/ & /g, '<br>&');
  
  return `<div class="restaurant-card" data-id="${restaurant.id}">
    <div class="restaurant-card-content">
      <img src="${restaurant.image}" alt="${restaurant.name}" loading="lazy">
      <h3>${restaurant.name}</h3>
      <div class="rating" style="color:#FFD700;font-size:1rem;margin:0.3rem 0;">${starsHtml} <span style="color:#666;font-size:0.9rem;">${restaurant.rating}/5</span></div>
      <div class="address" style="font-size:0.95rem;color:#666;margin:0.3rem 0;display:flex;align-items:center;gap:0.3rem;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg> ${restaurant.address}</div>
      <div class="phone" style="font-size:0.95rem;color:#666;margin:0.3rem 0;display:flex;align-items:center;gap:0.3rem;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path></svg> ${phone}</div>
      <div class="price" style="font-weight:600;color:var(--primary-green);margin:0.3rem 0;display:flex;align-items:center;gap:0.3rem;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><path d="M16 8h-6a2 2 0 1 0 0 4h4a2 2 0 1 1 0 4H8"></path><path d="M12 18V6"></path></svg> ${restaurant.price_range}</div>
      <div class="hours" style="font-size:0.9rem;color:#888;margin:0.3rem 0;display:flex;align-items:center;gap:0.3rem;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg> ${hours}</div>
    </div>
    <div class="restaurant-buttons">
      <a href="${restaurant.page_url}" target="_blank" rel="noopener" class="btn btn-primary" style="display:flex;align-items:center;gap:0.3rem;text-decoration:none;justify-content:center;flex:1;padding:0.5rem 0.5rem;font-size:0.85rem;white-space:nowrap;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M18 2h-3a5 5 0 0 0-5 5v3H7v4h3v8h4v-8h3l1-4h-4V7a1 1 0 0 1 1-1h3z"></path></svg> Trang</a>
      <a href="${mapUrl}" target="_blank" rel="noopener" class="btn btn-primary" style="display:flex;align-items:center;gap:0.3rem;text-decoration:none;justify-content:center;flex:1;padding:0.5rem 0.5rem;font-size:0.85rem;white-space:nowrap;"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path><circle cx="12" cy="10" r="3"></circle></svg> Bản đồ</a>
    </div>
  </div>`;
}

// Scroll to top button
function handleScrollToTop() {
  const btn = document.getElementById('scrollToTopBtn');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 300) {
      btn.classList.add('show');
    } else {
      btn.classList.remove('show');
    }
  });
  btn.addEventListener('click', () => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });
}

// Load featured recipes for homepage
function loadFeaturedRecipes() {
  // Render first 4 featured recipes (deterministic, no auto-rotation)
  fetch('data/recipes.json')
    .then(res => res.json())
    .then(data => {
      const list = document.getElementById('featured-recipes-list');
      list.innerHTML = '';
      const items = data.slice(0, 4);
      items.forEach(recipe => list.innerHTML += renderRecipeCard(recipe));
      // attach click handlers so featured cards open modal or navigate
      const cards = list.querySelectorAll('.recipe-card');
      cards.forEach(card => {
        const rid = card.getAttribute('data-id');
        card.addEventListener('click', (e) => {
          const rec = data.find(r => String(r.id) === String(rid));
          if (typeof window.openRecipeModal === 'function' && document.getElementById('recipeModal')) {
            e.preventDefault();
            window.openRecipeModal(rec);
          } else {
            window.location.href = 'recipe-detail.html?id=' + encodeURIComponent(rid);
          }
        });
      });
      lazyLoadImages();
    })
    .catch(err => {
      console.error('Failed to load featured recipes:', err);
    });
}

// Render featured restaurants randomly and rotate every 12s
function loadFeaturedRestaurants() {
  fetch('data/restaurants.json')
    .then(res => res.json())
    .then(data => {
      const list = document.getElementById('featured-restaurants-list');
      let prevIndexes = [];
      function pickRandomSet() {
        const indexes = [];
        const max = data.length;
        const count = Math.min(4, max);
        while (indexes.length < count) {
          const idx = Math.floor(Math.random() * max);
          if (!indexes.includes(idx)) indexes.push(idx);
        }
        if (arraysEqual(indexes, prevIndexes)) return pickRandomSet();
        prevIndexes = indexes;
        return indexes;
      }
      function arraysEqual(a, b) {
        if (!b || a.length !== b.length) return false;
        for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
        return true;
      }
      function renderSet() {
        const idxs = pickRandomSet();
        list.classList.add('hidden-anim');
        setTimeout(() => {
          list.innerHTML = '';
          idxs.forEach(i => {
            const r = data[i];
            list.innerHTML += renderRestaurantCard(r);
          });
          lazyLoadImages();
          requestAnimationFrame(() => {
            list.classList.remove('hidden-anim');
            list.classList.add('show-anim');
            setTimeout(() => list.classList.remove('show-anim'), 800);
          });
        }, 300);
      }
      renderSet();
      setInterval(renderSet, 12000);
    });
}

// Load recipes page with search & filter
function loadRecipesPage() {
  let recipesData = [];
  const grid = document.getElementById('recipesGrid');
  const searchInput = document.getElementById('searchInput');
  const typeFilter = document.getElementById('typeFilter');

  grid.innerHTML = '<p style="grid-column:1/-1;text-align:center;font-size:1.1rem;color:#666;padding:2rem;">Đang tải...</p>';

  fetch('data/recipes.json')
    .then(res => {
      if (!res.ok) throw new Error('Không thể tải dữ liệu');
      return res.json();
    })
    .then(data => {
      recipesData = data;
      renderRecipes(recipesData);
    })
    .catch(err => {
      console.error('Failed to load recipes:', err);
      grid.innerHTML = '<p style="grid-column:1/-1;text-align:center;font-size:1.1rem;color:#e74c3c;padding:2rem;">⚠️ Không thể tải dữ liệu món chay. Vui lòng thử lại sau.</p>';
    });

  function renderRecipes(data) {
    grid.innerHTML = '';
    if (data.length === 0) {
      grid.innerHTML = '<p style="grid-column:1/-1;text-align:center;font-size:1.1rem;color:#666;padding:2rem;">Không tìm thấy món chay phù hợp.</p>';
      return;
    }
    data.forEach(recipe => {
      grid.innerHTML += renderRecipeCard(recipe);
    });
    lazyLoadImages();
    // Attach click handlers to open modal when available, otherwise navigate
    const cards = grid.querySelectorAll('.recipe-card');
    cards.forEach(card => {
      const rid = card.getAttribute('data-id');
      card.addEventListener('click', (e) => {
        const rec = recipesData.find(r => String(r.id) === String(rid));
        // Luôn chuyển sang trang chi tiết nếu không có modal
        if (window.recipeModalAvailable && document.getElementById('recipeModal')) {
          e.preventDefault();
          window.openRecipeModal(rec);
        } else {
          window.location.href = 'recipe-detail.html?id=' + encodeURIComponent(rid);
        }
      });
    });
  }

  function filterRecipes() {
    let keyword = searchInput.value.trim().toLowerCase();
    let type = typeFilter.value;
    let filtered = recipesData.filter(recipe => {
      let matchName = recipe.name.toLowerCase().includes(keyword);
      let matchIngredient = recipe.ingredients.some(ing => ing.toLowerCase().includes(keyword));
      let matchType = type ? recipe.tags.includes(type) : true;
      return (matchName || matchIngredient) && matchType;
    });
    renderRecipes(filtered);
  }

  let debounceTimer;
  searchInput.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(filterRecipes, 300);
  });
  typeFilter.addEventListener('change', filterRecipes);
}

// Render recipe card
function renderRecipeCard(recipe) {
  return `<div class="recipe-card" data-id="${recipe.id}">
    <div class="recipe-card-content">
      <img src="${recipe.image}" alt="${recipe.name}" loading="lazy">
      <div class="tags">${recipe.tags.map(tag => `<span class='tag'>${tag}</span>`).join('')}</div>
      <h3>${recipe.name}</h3>
      <p>${recipe.short_description}</p>
    </div>
    <div class="recipe-time-wrapper">
      <div class="time">⏱ ${recipe.time}</div>
    </div>
  </div>`;
}

// Load recipe detail page
function loadRecipeDetail() {
  const params = new URLSearchParams(window.location.search);
  const id = params.get('id');
  
  if (!id) {
    window.location.href = 'recipes.html';
    return;
  }

  fetch('data/recipes.json')
    .then(res => res.json())
    .then(data => {
      const container = document.getElementById('recipeDetailContent');
      const recipe = data.find(r => String(r.id) === String(id));
      
      if (!recipe) {
        container.innerHTML = '<h2>Không tìm thấy món chay này.</h2><p>Vui lòng chọn lại từ danh sách món chay.</p>';
        return;
      }
      
      const longDescriptions = {
        1: "Phở chay Hà Nội là sự kết hợp tinh tế giữa nước dùng rau củ ngọt thanh, nấm hương thơm đặc trưng và đậu phụ chiên giòn. Món ăn mang hương vị truyền thống, thanh đạm, thích hợp cho mọi lứa tuổi. Bánh phở mềm, nước dùng trong, topping đa dạng, ăn kèm rau thơm, chanh, ớt giúp tăng vị giác. Đây là lựa chọn lý tưởng cho bữa sáng hoặc những ngày cần thanh lọc cơ thể.",
        2: "Bún riêu chay nổi bật với màu sắc hấp dẫn từ cà chua, đậu phụ, nấm rơm và nước dùng ngọt thanh. Món ăn giữ trọn vị riêu truyền thống nhưng hoàn toàn từ thực vật, phù hợp cho người ăn chay và muốn đổi vị. Rau sống tươi mát, bún mềm, nước dùng đậm đà, ăn kèm chanh, ớt tạo nên trải nghiệm ẩm thực đặc biệt.",
        3: "Cơm rang thập cẩm chay là sự hòa quyện của nhiều loại rau củ, nấm, đậu phụ và hạt điều, tạo nên món ăn đầy màu sắc, bổ dưỡng. Cơm tơi, không dính, vị đậm đà, topping giòn ngon, thích hợp cho bữa trưa hoặc tối. Món này giúp bổ sung chất xơ, vitamin và protein thực vật, tốt cho sức khỏe.",
        4: "Đậu phụ sốt cà chua là món ăn quen thuộc, dễ làm, với vị chua ngọt hài hòa từ cà chua tươi và đậu phụ mềm. Sốt cà chua thấm đều vào từng miếng đậu, ăn cùng cơm trắng rất ngon miệng. Món này giàu protein, vitamin, thích hợp cho cả gia đình, đặc biệt vào những ngày ăn chay.",
        5: "Gỏi cuốn chay thanh mát, nhiều rau củ, đậu phụ chiên, ăn kèm nước chấm đặc biệt. Món ăn giàu chất xơ, vitamin, dễ tiêu hóa, thích hợp làm món khai vị hoặc ăn nhẹ. Gỏi cuốn có thể biến tấu với nhiều loại rau củ theo mùa, giúp thực đơn chay thêm phong phú.",
        6: "Canh nấm đậu phụ là món canh thanh đạm, bổ dưỡng, vị ngọt tự nhiên từ nấm và rau củ. Nước canh trong, nấm dai ngon, đậu phụ mềm, ăn nóng rất hợp vào những ngày trời mát. Món này giúp thanh lọc cơ thể, bổ sung vitamin và khoáng chất cần thiết.",
        7: "Bánh xèo chay giòn rụm, nhân rau củ, nấm, ăn kèm rau sống và nước chấm chay. Vỏ bánh vàng giòn, nhân đậm đà, rau sống tươi mát, nước chấm hài hòa. Món này thích hợp cho bữa xế hoặc tiệc chay, giúp thực đơn thêm đa dạng và hấp dẫn.",
        8: "Chè đậu xanh chay ngọt thanh, giải nhiệt, ăn kèm nước cốt dừa béo nhẹ. Đậu xanh mềm, nước chè trong, vị ngọt vừa phải, thích hợp cho mùa hè hoặc làm món tráng miệng sau bữa ăn. Món này giúp thanh nhiệt, bổ sung chất xơ và vitamin.",
        9: "Súp bí đỏ chay mịn, thơm, giàu dinh dưỡng, thích hợp cho trẻ nhỏ và người ăn kiêng. Bí đỏ vàng cam, vị ngọt tự nhiên, kết hợp cùng sữa hạt tạo độ béo nhẹ, ăn nóng rất ngon. Món này tốt cho mắt, tiêu hóa và giúp bổ sung năng lượng.",
        10: "Đậu phụ chiên giòn vàng, lớp vỏ xù hấp dẫn, ăn kèm nước chấm chay cay nhẹ. Đậu phụ mềm bên trong, giòn bên ngoài, thích hợp làm món ăn chơi hoặc ăn kèm cơm. Món này giàu protein, dễ chế biến, phù hợp mọi lứa tuổi.",
        11: "Nấm xào thập cẩm chay là sự kết hợp của nhiều loại nấm, rau củ, vị ngọt tự nhiên, màu sắc bắt mắt. Nấm dai ngon, rau củ giòn, gia vị vừa miệng, thích hợp cho bữa cơm gia đình hoặc tiệc chay. Món này bổ sung chất xơ, vitamin và khoáng chất.",
        12: "Bánh chuối hấp chay mềm thơm, vị ngọt tự nhiên, ăn kèm nước cốt dừa béo nhẹ. Chuối chín, bột năng tạo độ dẻo, nước cốt dừa béo, thích hợp làm món tráng miệng hoặc ăn nhẹ. Món này tốt cho tiêu hóa, bổ sung năng lượng và vitamin."
      };
      
      container.innerHTML = `
        <div style="max-width:1400px;margin:0 auto;padding:0 20px;">
          <img src="${recipe.image}" alt="${recipe.name}" loading="lazy" style="width:100%;max-width:320px;border-radius:10px;margin-bottom:16px;display:block;margin-left:auto;margin-right:auto;">
          <div style="display:flex;flex-direction:column;align-items:center;gap:12px;margin-bottom:12px;">
            <button id="videoBtn" class="btn btn-primary" style="margin:1.1rem 0 1.2rem 0;min-width:180px;">Xem video hướng dẫn</button>
          </div>
          <h2 style="margin-top:0;text-align:center;">${recipe.name}</h2>
          <div class="time" style="text-align:center;">⏱ ${recipe.time}</div>
          <p style="max-width:900px;margin-left:auto;margin-right:auto;text-align:left;">${longDescriptions[recipe.id] || recipe.short_description}</p>
          <div class="ingredients">
            <h3>Nguyên liệu</h3>
            ${recipe.ingredients.map(ing => `<label><input type="checkbox"> ${ing}</label>`).join('')}
          </div>
        </div>
        <div class="steps">
          <h3 style="font-family:'Playfair Display',serif;font-size:1.8rem;color:#3BAF4A;text-align:center;margin:2.5rem 0 2rem 0;">Các bước thực hiện</h3>
          ${recipe.steps.map((step, i) => {
            // Tách **Tiêu đề:** trước
            const titleMatch = step.match(/\*\*([^*]+?):\*\*/);
            let title = '';
            let remainingText = step;
            
            if (titleMatch) {
              title = titleMatch[1].trim();
              remainingText = step.replace(/\*\*[^*]+?:\*\*/, '').trim();
            }
            
            // Tách *Mẹo/Lưu ý/Tip:* 
            const tipMatch = remainingText.match(/\*([^*]+?):\*/);
            let content = remainingText;
            let tipText = '';
            
            if (tipMatch) {
              const fullTip = tipMatch[0]; // e.g., "*Mẹo:*"
              const tipLabel = tipMatch[1].trim(); // e.g., "Mẹo"
              
              // Tìm text sau dấu *Label:*
              const tipStartIndex = remainingText.indexOf(fullTip);
              if (tipStartIndex !== -1) {
                content = remainingText.substring(0, tipStartIndex).trim();
                tipText = remainingText.substring(tipStartIndex + fullTip.length).trim();
              }
            }
            
            return `
              <div class="step">
                <div style='display:flex;align-items:center;gap:0.8rem;margin-bottom:0.8rem;'>
                  <span style='display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;background:#3BAF4A;color:#fff;border-radius:50%;font-weight:700;font-size:0.9rem;flex-shrink:0;'>${i+1}</span>
                  ${title ? `<strong style='color:#C8A27D;font-size:1.15rem;font-weight:700;'>${title}</strong>` : ''}
                </div>
                <div style='font-size:1.05rem;line-height:1.75;color:#444;margin-left:40px;'>${content}</div>
                ${tipText ? `<div style='margin-top:1rem;margin-left:40px;padding:0.7rem 1rem;background:#fffef5;border-left:3px solid #C8A27D;border-radius:6px;'><span style='color:#C8A27D;font-weight:600;font-size:0.95rem;'>Mẹo:</span> <em style='color:#666;font-size:0.98rem;'>${tipText}</em></div>` : ''}
              </div>
            `;
          }).join('')}
        </div>
        <div style="max-width:1400px;margin:0 auto;padding:0 20px;">
          <div style="display:flex;justify-content:center;margin-top:24px;">
            <a href="recipes.html" class="btn btn-secondary" style="min-width:180px;">Quay lại danh sách món chay</a>
          </div>
        </div>
      `;
      
      const vb = document.getElementById('videoBtn');
      if (vb) {
        vb.onclick = function() {
          if (recipe.video_url && recipe.video_url.trim() !== "") {
            window.open(recipe.video_url, '_blank');
          } else {
            alert('Chưa có video hướng dẫn cho món này.');
          }
        };
      }
      lazyLoadImages();
    })
    .catch(err => {
      const container = document.getElementById('recipe-detail');
      container.innerHTML = `
        <h2>Không thể tải dữ liệu công thức</h2>
        <p>Vui lòng thử lại sau.</p>
      `;
    });
}

// Load restaurants page with search & filter
function loadRestaurantsPage() {
  let restaurantsData = [];
  const grid = document.getElementById('restaurantsList');
  const searchInput = document.getElementById('searchInput');
  const districtFilter = document.getElementById('districtFilter');

  if (!grid) return;

  grid.innerHTML = '<p style="grid-column:1/-1;text-align:center;font-size:1.1rem;color:#666;padding:2rem;">Đang tải...</p>';

  fetch('data/restaurants.json')
    .then(res => {
      if (!res.ok) throw new Error('Không thể tải dữ liệu');
      return res.json();
    })
    .then(data => {
      restaurantsData = data;
      renderRestaurants(restaurantsData);
    })
    .catch(err => {
      console.error('Failed to load restaurants:', err);
      grid.innerHTML = '<p style="grid-column:1/-1;text-align:center;font-size:1.1rem;color:#e74c3c;padding:2rem;">⚠️ Không thể tải dữ liệu quán chay. Vui lòng thử lại sau.</p>';
    });

  function renderRestaurants(data) {
    grid.innerHTML = '';
    if (data.length === 0) {
      const selectedDistrict = districtFilter ? districtFilter.value : '';
      if (selectedDistrict) {
        grid.innerHTML = `<p style="grid-column:1/-1;text-align:center;font-size:1.1rem;color:#666;padding:2rem;">Chưa cập nhật thông tin quán chay ở khu vực <strong>${selectedDistrict}</strong>.<br>Vui lòng chọn quận khác hoặc xem tất cả.</p>`;
      } else {
        grid.innerHTML = '<p style="grid-column:1/-1;text-align:center;font-size:1.1rem;color:#666;padding:2rem;">Không tìm thấy quán chay phù hợp.</p>';
      }
      return;
    }
    data.forEach(restaurant => {
      grid.innerHTML += renderRestaurantCard(restaurant);
    });
    lazyLoadImages();
  }

  function filterRestaurants() {
    let keyword = searchInput.value.trim().toLowerCase();
    let district = districtFilter.value;
    let filtered = restaurantsData.filter(restaurant => {
      let matchName = restaurant.name.toLowerCase().includes(keyword);
      let matchAddress = restaurant.address.toLowerCase().includes(keyword);
      let matchDistrict = district ? restaurant.district === district : true;
      return (matchName || matchAddress) && matchDistrict;
    });
    renderRestaurants(filtered);
  }

  if (searchInput) {
    let debounceTimer;
    searchInput.addEventListener('input', function() {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(filterRestaurants, 300);
    });
  }
  if (districtFilter) districtFilter.addEventListener('change', filterRestaurants);
}

// Load temples page with search & filter
function loadTemplesPage() {
  let templesData = [];
  const grid = document.getElementById('templesGrid');
  const searchInput = document.getElementById('searchInput');
  const districtFilter = document.getElementById('districtFilter');

  if (!grid) return;

  grid.innerHTML = '<p style="grid-column:1/-1;text-align:center;font-size:1.1rem;color:#666;padding:2rem;">Đang tải...</p>';

  fetch('data/temples.json')
    .then(res => {
      if (!res.ok) throw new Error('Không thể tải dữ liệu');
      return res.json();
    })
    .then(data => {
      templesData = data;
      renderTemples(templesData);
    })
    .catch(err => {
      console.error('Failed to load temples:', err);
      grid.innerHTML = '<p style="grid-column:1/-1;text-align:center;font-size:1.1rem;color:#e74c3c;padding:2rem;">⚠️ Không thể tải dữ liệu chùa. Vui lòng thử lại sau.</p>';
    });

  function renderTemples(data) {
    grid.innerHTML = '';
    if (data.length === 0) {
      const selectedDistrict = districtFilter ? districtFilter.value : '';
      if (selectedDistrict) {
        grid.innerHTML = `<p style="grid-column:1/-1;text-align:center;font-size:1.1rem;color:#666;padding:2rem;">Chưa cập nhật thông tin chùa ở khu vực <strong>${selectedDistrict}</strong>.<br>Vui lòng chọn quận khác hoặc xem tất cả.</p>`;
      } else {
        grid.innerHTML = '<p style="grid-column:1/-1;text-align:center;font-size:1.1rem;color:#666;padding:2rem;">Không tìm thấy chùa phù hợp.</p>';
      }
      return;
    }
    data.forEach(temple => {
      grid.innerHTML += renderTempleCard(temple);
    });
    lazyLoadImages();
  }

  function renderTempleCard(temple) {
    return `
      <div class="temple-card">
        <img src="${temple.image}" alt="${temple.name}" loading="lazy">
        <div class="temple-info">
          <h3>${temple.name}</h3>
          <p class="address"><strong>Địa chỉ:</strong> ${temple.address}</p>
          <p class="historical"><strong>Lịch sử:</strong> ${temple.historical_info}</p>
          <p class="features"><strong>Đặc điểm:</strong> ${temple.features}</p>
          ${temple.phone ? `<p class="contact"><strong>Điện thoại:</strong> <a href="tel:${temple.phone}">${temple.phone}</a></p>` : ''}
          ${temple.email ? `<p class="contact"><strong>Email:</strong> <a href="mailto:${temple.email}">${temple.email}</a></p>` : ''}
          ${temple.website ? `<p class="contact"><strong>Website:</strong> <a href="${temple.website}" target="_blank" rel="noopener">Xem trang</a></p>` : ''}
        </div>
      </div>
    `;
  }

  function filterTemples() {
    let keyword = searchInput.value.trim().toLowerCase();
    let district = districtFilter.value;
    let filtered = templesData.filter(temple => {
      let matchName = temple.name.toLowerCase().includes(keyword);
      let matchAddress = temple.address.toLowerCase().includes(keyword);
      let matchDistrict = district ? temple.district === district : true;
      return (matchName || matchAddress) && matchDistrict;
    });
    renderTemples(filtered);
  }

  if (searchInput) {
    let debounceTimer;
    searchInput.addEventListener('input', function() {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(filterTemples, 300);
    });
  }
  if (districtFilter) districtFilter.addEventListener('change', filterTemples);
}

// Xóa hamburger menu vì chỉ hỗ trợ PC/Laptop
document.addEventListener('DOMContentLoaded', function() {
  // Lotus Intro Animation
  var lotusIntro = document.getElementById('lotusIntro');
  if (lotusIntro) {
    // Kiểm tra xem đã xem intro chưa (trong session này)
    var hasSeenIntro = sessionStorage.getItem('hasSeenIntro');
    
    if (hasSeenIntro) {
      // Đã xem rồi, ẩn ngay
      lotusIntro.style.display = 'none';
    } else {
      // Chưa xem, hiển thị intro
      setTimeout(function() {
        lotusIntro.style.opacity = '0';
        setTimeout(function() {
          lotusIntro.style.display = 'none';
          lotusIntro.remove(); // Xóa hoàn toàn khỏi DOM
          sessionStorage.setItem('hasSeenIntro', 'true');
        }, 600);
      }, 1500); // Hiển thị 1.5s rồi mờ dần (0.6s) = tổng 2.1s
    }
  }
});
