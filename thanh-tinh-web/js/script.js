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
  if (document.getElementById('recipesGrid') && document.getElementById('restaurantsList')) {
    loadRecipesAndRestaurants();
  } else {
    if (document.getElementById('recipesGrid')) {
      loadRecipesPage();
    }
    if (document.getElementById('recipe-detail')) {
      loadRecipeDetail();
    }
    if (document.getElementById('restaurantsGrid')) {
      loadRestaurantsPage();
    }
    // If a page has only `restaurantsList` (static container), also initialize restaurants page behaviors
    if (document.getElementById('restaurantsList') && !document.getElementById('restaurantsGrid')) {
      loadRestaurantsPage();
    }
    if (document.getElementById('templesGrid')) {
      loadTemplesPage();
    }
  }
  // Contact form handler (Formspree-ready). Replace CONTACT_ENDPOINT with your Formspree endpoint.
  if (document.getElementById('contactForm')) {
    const CONTACT_ENDPOINT = 'https://formspree.io/f/YOUR_FORM_ID'; // <- replace this with your Formspree form URL
    const form = document.getElementById('contactForm');
    const statusDiv = document.getElementById('contactStatus');
    form.addEventListener('submit', (e) => {
      e.preventDefault();
      statusDiv.textContent = '';
      const name = document.getElementById('name').value.trim();
      const email = document.getElementById('email').value.trim();
      const message = document.getElementById('message').value.trim();
      if (!name || !email || !message) {
        statusDiv.textContent = 'Vui lòng điền đầy đủ thông tin.';
        statusDiv.style.color = '#c0392b';
        return;
      }
      statusDiv.textContent = 'Đang gửi...';
      statusDiv.style.color = '#333';
      // POST JSON to Formspree endpoint
      fetch(CONTACT_ENDPOINT, {
        method: 'POST',
        headers: { 'Accept': 'application/json', 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, email, message })
      }).then(r => {
        if (r.ok) {
          statusDiv.textContent = 'Cảm ơn! Tin nhắn của bạn đã được gửi.';
          statusDiv.style.color = '#27ae60';
          form.reset();
        } else {
          return r.json().then(data => { throw data; });
        }
      }).catch(err => {
        console.error('Contact form error', err);
        statusDiv.textContent = 'Gửi thất bại. Bạn có thể thử lại hoặc gửi email trực tiếp tới daoduyphat066@gmail.com';
        statusDiv.style.color = '#c0392b';
      });
    });
  }
  // Initialize any static map buttons added in HTML (set href from data-address)
  function initStaticMapButtons() {
    const mapBtns = document.querySelectorAll('.btn-map[data-address]');
    mapBtns.forEach(btn => {
      const addr = btn.getAttribute('data-address') || '';
      const url = 'https://www.google.com/maps/search/?api=1&query=' + encodeURIComponent(addr);
      btn.setAttribute('href', url);
      btn.setAttribute('target', '_blank');
      btn.setAttribute('rel', 'noopener');
    });
  }
  initStaticMapButtons();
  // Inline recipe modal elements (if present on the page)
  const recipeModal = document.getElementById('recipeModal');
  const recipeModalBody = document.getElementById('recipeModalBody');
  const recipeModalClose = document.getElementById('recipeModalClose');
  function escapeHtml(str) {
    if (!str) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }
  function openRecipeModal(recipe) {
    if (!recipeModal || !recipeModalBody) return;
    const html = [];
    html.push(`<h2 id="modalTitle">${escapeHtml(recipe.name || recipe.title || '')}</h2>`);
    if (recipe.image) html.push(`<img src="${escapeHtml(recipe.image)}" alt="${escapeHtml(recipe.name || '')}" style="max-width:100%;height:auto;margin-bottom:12px;">`);
    if (recipe.description) html.push(`<p>${escapeHtml(recipe.description)}</p>`);
    if (recipe.ingredients && recipe.ingredients.length) {
      html.push('<h3>Nguyên liệu</h3><ul>');
      for (const ing of recipe.ingredients) html.push(`<li>${escapeHtml(ing)}</li>`);
      html.push('</ul>');
    }
    if (recipe.steps && recipe.steps.length) {
      html.push('<h3>Cách làm</h3><ol>');
      for (const s of recipe.steps) html.push(`<li>${escapeHtml(s)}</li>`);
      html.push('</ol>');
    }
    if (recipe.video_url) {
      html.push(`<p><a class="btn btn-primary" href="${escapeHtml(recipe.video_url)}" target="_blank" rel="noopener">Xem video hướng dẫn</a></p>`);
    }
    recipeModalBody.innerHTML = html.join('\n');
    recipeModal.classList.add('show');
    recipeModal.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';
  }
  function closeRecipeModal() {
    if (!recipeModal) return;
    recipeModal.classList.remove('show');
    recipeModal.setAttribute('aria-hidden', 'true');
    if (recipeModalBody) recipeModalBody.innerHTML = '';
    document.body.style.overflow = '';
  }
  if (recipeModalClose) recipeModalClose.addEventListener('click', closeRecipeModal);
  if (recipeModal) recipeModal.addEventListener('click', function (e) { if (e.target === recipeModal) closeRecipeModal(); });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') closeRecipeModal(); });
  // expose modal controls globally so other modules can open/close
  window.openRecipeModal = openRecipeModal;
  window.closeRecipeModal = closeRecipeModal;
  window.recipeModalAvailable = !!recipeModal;
  // Prevent selection highlight on the floating widget (mousedown guard as extra safety)
  try {
    const moEls = document.querySelectorAll('#moMiniWrap, #moMini, #moLabel, #moMiniWrap *');
    moEls.forEach(el => el.addEventListener('mousedown', (ev) => ev.preventDefault()));
  } catch (e) { /* silent */ }
});
// Hiển thị danh sách món chay và quán chay trên restaurants.html
function loadRecipesAndRestaurants() {
  // Danh sách món chay
  if (document.getElementById('recipesGrid')) {
    fetch('data/recipes.json')
      .then(res => res.json())
      .then(data => {
        const grid = document.getElementById('recipesGrid');
        grid.innerHTML = '';
        data.forEach(recipe => {
          grid.innerHTML += renderRecipeCard(recipe);
        });
        // attach click handlers for recipe cards
        const cards = grid.querySelectorAll('.recipe-card');
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
      });
  }
  // Danh sách quán chay
  if (document.getElementById('restaurantsList')) {
    fetch('data/restaurants.json')
      .then(res => res.json())
      .then(data => {
        const list = document.getElementById('restaurantsList');
        list.className = 'restaurants-grid';
        list.innerHTML = '';
        data.forEach(restaurant => {
          const mapUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(restaurant.address)}`;
          list.innerHTML += `
            <div class="restaurant-card" onclick="this.classList.toggle('card-active')">
              <img src="${restaurant.image}" alt="${restaurant.name}" loading="lazy">
              <h3>${restaurant.name}</h3>
              <div class="address">${restaurant.address}</div>
              <div class="price">${restaurant.price_range}</div>
              <div class="desc">${restaurant.description}</div>
              <div class="hours">Giờ mở cửa: ${restaurant.opening_hours}</div>
              <a href="${mapUrl}" target="_blank" rel="noopener" onclick="event.stopPropagation()" class="btn btn-secondary" style="margin-top:8px">Xem bản đồ</a>
            </div>
          `;
        });
        lazyLoadImages();
      });
  }
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
            const mapUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(r.address)}`;
            list.innerHTML += `
              <div class="restaurant-card">
                <img src="${r.image}" alt="${r.name}" loading="lazy">
                <h3>${r.name}</h3>
                <div class="address">${r.address}</div>
                <div class="price">${r.price_range}</div>
                <div class="desc">${r.description}</div>
                <div class="hours">Giờ mở cửa: ${r.opening_hours}</div>
                <a href="${mapUrl}" target="_blank" rel="noopener" onclick="event.stopPropagation()" class="btn btn-secondary" style="margin-top:8px">Xem bản đồ</a>
              </div>
            `;
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

  fetch('data/recipes.json')
    .then(res => res.json())
    .then(data => {
      recipesData = data;
      renderRecipes(recipesData);
    });

  function renderRecipes(data) {
    grid.innerHTML = '';
    if (data.length === 0) {
      grid.innerHTML = '<p>Không tìm thấy món chay phù hợp.</p>';
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
        if (typeof window.openRecipeModal === 'function' && document.getElementById('recipeModal')) {
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

  searchInput.addEventListener('input', filterRecipes);
  typeFilter.addEventListener('change', filterRecipes);
}

// Render recipe card
function renderRecipeCard(recipe) {
  return `<div class="recipe-card" data-id="${recipe.id}">
    <img src="${recipe.image}" alt="${recipe.name}" loading="lazy">
    <div class="tags">${recipe.tags.map(tag => `<span class='tag'>${tag}</span>`).join('')}</div>
    <h3>${recipe.name}</h3>
    <p>${recipe.short_description}</p>
    <div class="time">⏱ ${recipe.time}</div>
  </div>`;
}

// Load recipe detail page
function loadRecipeDetail() {
  const params = new URLSearchParams(window.location.search);
  const id = params.get('id');
  console.log('[site] loadRecipeDetail start, id=', id);
  // add lightweight on-page debug banner to show fetch status (helps users without DevTools)
  try {
    const container = document.getElementById('recipe-detail');
    if (container) {
      let dbg = document.getElementById('recipeDebugBanner');
      if (!dbg) {
        dbg = document.createElement('div');
        dbg.id = 'recipeDebugBanner';
        dbg.style.cssText = 'font-size:1.05rem;padding:10px 16px;border-radius:10px;background:#fffbe8;color:#c0392b;border:2px solid #f0dca8;margin-bottom:18px;text-align:center;font-weight:600;';
        container.insertAdjacentElement('afterbegin', dbg);
      }
      dbg.style.display = '';
      dbg.textContent = 'Đang tải dữ liệu công thức...';
      let help = document.getElementById('recipeErrorHelp');
      if (help) help.style.display = 'none';
      let fallback = document.getElementById('recipeFallbackContent');
      if (fallback) fallback.style.display = 'none';
    }
  } catch (e) { /* ignore banner errors */ }

  fetch('data/recipes.json')
    .then(res => {
      const dbg = document.getElementById('recipeDebugBanner');
      if (dbg) {
        dbg.style.display = '';
        dbg.textContent = res.ok ? 'Tải dữ liệu công thức thành công.' : `Lỗi tải dữ liệu: ${res.status} ${res.statusText}`;
      }
      let help = document.getElementById('recipeErrorHelp');
      if (!res.ok && help) {
        help.style.display = '';
        help.innerHTML = `<b>Lỗi tải dữ liệu công thức!</b><br>Hãy kiểm tra lại đường dẫn <code>data/recipes.json</code> hoặc chạy máy chủ HTTP:<br><pre>python -m http.server 8000</pre><br>Truy cập lại: <code>http://localhost:8000/recipe-detail.html?id=${id}</code>`;
      } else if (help) {
        help.style.display = 'none';
      }
      return res.json();
    })
    .then(data => {
      const container = document.getElementById('recipe-detail');
      const dbg = document.getElementById('recipeDebugBanner');
      let help = document.getElementById('recipeErrorHelp');
      let fallback = document.getElementById('recipeFallbackContent');
      try {
        const recipe = data.find(r => String(r.id) === String(id));
        if (!recipe) {
          if (dbg) dbg.textContent = 'Không tìm thấy món chay này trong dữ liệu.';
          if (help) {
            help.style.display = '';
            help.innerHTML = `<b>Không tìm thấy món chay này!</b><br>Vui lòng kiểm tra lại <code>id</code> trên đường dẫn hoặc chọn lại từ danh sách món chay.`;
          }
          if (fallback) fallback.style.display = '';
          container.innerHTML = '<h2>Không tìm thấy món chay này.</h2><p>Vui lòng chọn lại từ danh sách món chay.</p>';
          return;
        }
        if (dbg) dbg.textContent = 'Hiển thị công thức thành công.';
        if (help) help.style.display = 'none';
        if (fallback) fallback.style.display = 'none';
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
          <img src="${recipe.image}" alt="${recipe.name}" loading="lazy" style="width:100%;max-width:320px;border-radius:10px;margin-bottom:16px;display:block;margin-left:auto;margin-right:auto;">
          <div style="display:flex;flex-direction:column;align-items:center;gap:12px;margin-bottom:12px;">
            <button id="videoBtn" class="btn btn-primary" style="margin:1.1rem 0 1.2rem 0;min-width:180px;">Xem video hướng dẫn</button>
            <a href="recipes.html" class="btn btn-secondary" style="min-width:180px;">Quay lại danh sách món chay</a>
          </div>
          <h2 style="margin-top:0">${recipe.name}</h2>
          <div class="time">⏱ ${recipe.time}</div>
          <p>${longDescriptions[recipe.id] || recipe.short_description}</p>
          <div class="ingredients">
            <h3>Nguyên liệu</h3>
            ${recipe.ingredients.map(ing => `<label style='display:block;margin-bottom:4px;'><input type="checkbox"> ${ing}</label>`).join('')}
          </div>
          <div class="steps">
            <h3>Các bước nấu</h3>
            ${recipe.steps.map((step, i) => `<div class="step" style='margin-bottom:6px;'><strong>Bước ${i+1}:</strong> ${step}</div>`).join('')}
          </div>
        `;
        // Xử lý nút video (guarded)
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
      } catch (ex) {
        if (dbg) dbg.textContent = 'Lỗi khi hiển thị công thức: ' + ex;
        if (help) {
          help.style.display = '';
          help.innerHTML = `<b>Lỗi khi hiển thị công thức!</b><br>Mở DevTools (F12) để xem chi tiết lỗi trong Console.`;
        }
        if (fallback) fallback.style.display = '';
        container.innerHTML = `<h2>Lỗi khi hiển thị công thức</h2><p>Mở DevTools (F12) để xem chi tiết trong Console.</p>`;
      }
    })
    .catch(err => {
      const container = document.getElementById('recipe-detail');
      const dbg = document.getElementById('recipeDebugBanner');
      let help = document.getElementById('recipeErrorHelp');
      if (dbg) dbg.style.display = '';
      if (dbg) dbg.textContent = 'Lỗi fetch dữ liệu: ' + err;
      if (help) {
        help.style.display = '';
        help.innerHTML = `<b>Lỗi fetch dữ liệu công thức!</b><br>Hãy kiểm tra lại đường dẫn <code>data/recipes.json</code> hoặc chạy máy chủ HTTP:<br><pre>python -m http.server 8000</pre><br>Truy cập lại: <code>http://localhost:8000/recipe-detail.html?id=${id}</code>`;
      }
      container.innerHTML = `
        <h2>Không thể tải dữ liệu công thức</h2>
        <p>Hệ thống không thể tải dữ liệu công thức từ <code>data/recipes.json</code>.</p>
        <p>Nếu bạn đang mở trang trực tiếp bằng <code>file://</code>, một số trình duyệt chặn truy vấn tới file JSON. Hãy chạy một máy chủ HTTP đơn giản (ví dụ dùng Python) và mở trang lại:</p>
        <pre>python -m http.server 8000</pre>
        <p>Rồi truy cập: <code>http://localhost:8000/recipe-detail.html?id=${id}</code></p>
      `;
    });
}

// Load restaurants page
function loadRestaurantsPage() {
  // Load restaurants and allow filtering by district via buttons shown inside each card
  fetch('data/restaurants.json')
    .then(res => res.json())
    .then(data => {
      const grid = document.getElementById('restaurantsGrid') || document.getElementById('restaurantsList');
      let restaurantsData = data;
      let currentDistrictFilter = null;

        // Build search/filter/sort controls above the grid
        function createControls() {
          const controlsId = 'restaurantsControls';
          if (document.getElementById(controlsId)) return;
          const controls = document.createElement('div');
          controls.id = controlsId;
          controls.style.maxWidth = '1200px';
          controls.style.margin = '0 auto 1rem auto';
          controls.style.display = 'flex';
          controls.style.gap = '0.75rem';
          controls.style.alignItems = 'center';
          controls.style.justifyContent = 'space-between';
          controls.style.padding = '0 1rem';

          controls.innerHTML = `
            <div style="display:flex;gap:0.75rem;flex:1 1 60%;align-items:center">
              <input id="restaurantSearch" type="search" placeholder="Tìm theo tên hoặc địa chỉ" style="flex:1;padding:0.6rem 0.9rem;border-radius:12px;border:1px solid #ddd;font-size:1rem;">
              <select id="restaurantDistrict" style="padding:0.6rem 0.9rem;border-radius:12px;border:1px solid #ddd;font-size:1rem;min-width:160px">
                <option value="">Tất cả quận</option>
              </select>
            </div>
            <div style="display:flex;gap:0.6rem;align-items:center">
              <select id="restaurantSort" style="padding:0.6rem 0.9rem;border-radius:12px;border:1px solid #ddd;font-size:1rem">
                <option value="">Sắp xếp</option>
                <option value="price-asc">Giá: thấp → cao</option>
                <option value="price-desc">Giá: cao → thấp</option>
              </select>
              <button id="clearRestaurantFilters" class="btn btn-secondary">Xóa</button>
            </div>
          `;
          grid.insertAdjacentElement('beforebegin', controls);

          // populate district select
          const districtSelect = document.getElementById('restaurantDistrict');
          const districts = Array.from(new Set(restaurantsData.map(r => (r.district || 'Chưa cập nhật')))).sort();
          districts.forEach(d => {
            const opt = document.createElement('option'); opt.value = d; opt.textContent = d; districtSelect.appendChild(opt);
          });

          // event listeners
          document.getElementById('restaurantSearch').addEventListener('input', applyFilters);
          districtSelect.addEventListener('change', applyFilters);
          document.getElementById('restaurantSort').addEventListener('change', applyFilters);
          document.getElementById('clearRestaurantFilters').addEventListener('click', () => {
            document.getElementById('restaurantSearch').value = '';
            districtSelect.value = '';
            document.getElementById('restaurantSort').value = '';
            applyFilters();
          });
        }

        function parsePriceRange(str) {
          if (!str) return NaN;
          // remove currency symbols and spaces, handle formats like "50.000 - 150.000đ"
          try {
            const s = str.replace(/đ|\s/g,'').replace(/\./g,'');
            if (s.indexOf('-') !== -1) {
              const parts = s.split('-').map(p => parseInt(p,10)).filter(n => !isNaN(n));
              if (parts.length === 2) return (parts[0] + parts[1]) / 2;
            }
            const n = parseInt(s,10);
            return isNaN(n) ? NaN : n;
          } catch (e) { return NaN; }
        }

        function applyFilters() {
          const q = (document.getElementById('restaurantSearch') && document.getElementById('restaurantSearch').value || '').trim().toLowerCase();
          const districtSel = document.getElementById('restaurantDistrict') ? document.getElementById('restaurantDistrict').value : '';
          const sortOpt = document.getElementById('restaurantSort') ? document.getElementById('restaurantSort').value : '';

          let filtered = restaurantsData.filter(r => {
            const name = (r.name || '').toLowerCase();
            const addr = (r.address || '').toLowerCase();
            const matchQ = !q || name.includes(q) || addr.includes(q);
            const matchDistrict = !districtSel || ((r.district || 'Chưa cập nhật') === districtSel);
            return matchQ && matchDistrict;
          });

          if (sortOpt === 'price-asc' || sortOpt === 'price-desc') {
            filtered.sort((a,b) => {
              const pa = parsePriceRange(a.price_range || '');
              const pb = parsePriceRange(b.price_range || '');
              if (isNaN(pa) && isNaN(pb)) return 0;
              if (isNaN(pa)) return 1;
              if (isNaN(pb)) return -1;
              return sortOpt === 'price-asc' ? pa - pb : pb - pa;
            });
          }

          // update filter bar (show selected district)
          renderFilterBar(districtSel || null);
          renderRestaurants(filtered);
        }

        // expose applyFilters so other handlers can call it
        window.applyRestaurantFilters = applyFilters;

        // create the controls and render initial list
        createControls();
        applyFilters();
      })
      .catch(err => {
        // If JSON fetch fails (e.g., opened via file://), fall back to scanning static DOM restaurant cards
        console.warn('Could not fetch restaurants.json, falling back to DOM scan', err);
        const grid = document.getElementById('restaurantsGrid') || document.getElementById('restaurantsList');
        let restaurantsData = [];
        const cardEls = grid ? Array.from(grid.querySelectorAll('.restaurant-card')) : [];
        cardEls.forEach(el => {
          const name = (el.querySelector('h3') && el.querySelector('h3').innerText) || '';
          const address = (el.querySelector('.address') && el.querySelector('.address').innerText) || '';
          const price_range = (el.querySelector('.price') && el.querySelector('.price').innerText) || '';
          const description = (el.querySelector('.desc') && el.querySelector('.desc').innerText) || '';
          const hoursText = (el.querySelector('.hours') && el.querySelector('.hours').innerText) || '';
          const opening_hours = hoursText.replace(/^Giờ mở cửa:\s*/i, '');
          const imgEl = el.querySelector('img');
          const image = imgEl ? imgEl.getAttribute('src') : '';
          const district = (el.querySelector('.district-badge') && el.querySelector('.district-badge').innerText) || 'Chưa cập nhật';
          restaurantsData.push({ name, address, price_range, description, opening_hours, image, district });
        });
        // continue with controls and filters using scanned data
        if (typeof createControls === 'function') createControls();
        if (typeof applyFilters === 'function') applyFilters();
      });

      function renderFilterBar(district) {
        const id = 'restaurantsFilterBar';
        let bar = document.getElementById(id);
        const gridEl = document.getElementById('restaurantsGrid') || document.getElementById('restaurantsList');
        if (!bar) {
          if (gridEl) gridEl.insertAdjacentHTML('beforebegin', `<div id="${id}" style="max-width:1200px;margin:0 auto 1rem auto;padding:0 1rem;display:flex;gap:1rem;align-items:center;justify-content:flex-end"></div>`);
          bar = document.getElementById(id);
        }
        if (bar) {
          if (district) {
            bar.innerHTML = `<div style="font-weight:600;color:#3BAF4A;margin-right:auto">Đang lọc: Quận ${district}</div><button class='btn btn-secondary' onclick="clearRestaurantFilter()">Bỏ lọc</button>`;
          } else {
            bar.innerHTML = '';
          }
        }
      }

      function renderRestaurants(listData) {
        const gridEl = document.getElementById('restaurantsGrid') || document.getElementById('restaurantsList');
        if (!gridEl) return;
        gridEl.innerHTML = '';
        if (!listData || listData.length === 0) {
          gridEl.innerHTML = '<p>Không tìm thấy quán chay cho bộ lọc này.</p>';
          return;
        }
        listData.forEach(restaurant => {
          const mapUrl = `https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(restaurant.address)}`;
          const districtLabel = restaurant.district || 'Chưa cập nhật';
          gridEl.innerHTML += `<div class="restaurant-card" onclick="this.classList.toggle('card-active')">
            <div class="district-badge">${districtLabel}</div>
            <img src="${restaurant.image}" alt="${restaurant.name}" loading="lazy">
            <h3>${restaurant.name}</h3>
            <div class="address">${restaurant.address}</div>
            <div class="price">${restaurant.price_range}</div>
            <div class="desc">${restaurant.description}</div>
            <div class="hours">Giờ mở cửa: ${restaurant.opening_hours}</div>
            <a href="${mapUrl}" target="_blank" rel="noopener" onclick="event.stopPropagation()" class="btn btn-secondary" style="margin-top:8px">Xem bản đồ</a>
            <button class="btn btn-primary district-btn" onclick="event.stopPropagation(); filterByDistrict('${escapeQuotes(districtLabel)}')">Quận ${districtLabel}</button>
          </div>`;
        });
        lazyLoadImages();
      }

      // Expose filter functions globally so inline handlers can call them
      window.filterByDistrict = function(district) {
        // Set district control and apply filters
        const sel = document.getElementById('restaurantDistrict');
        if (sel) sel.value = district;
        if (typeof applyFilters === 'function') applyFilters();
        window.scrollTo({ top: 200, behavior: 'smooth' });
      };
      window.clearRestaurantFilter = function() {
        const search = document.getElementById('restaurantSearch');
        const sel = document.getElementById('restaurantDistrict');
        const sort = document.getElementById('restaurantSort');
        if (search) search.value = '';
        if (sel) sel.value = '';
        if (sort) sort.value = '';
        if (typeof applyFilters === 'function') applyFilters();
      };

      // helper to escape single quotes in district strings when injected into inline handlers
      function escapeQuotes(s) {
        if (s == null) return '';
        return String(s).replace(/'/g, "\\'");
      }
      renderFilterBar(null);
    }
}
