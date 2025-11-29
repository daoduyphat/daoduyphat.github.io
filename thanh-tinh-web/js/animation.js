// animation.js - Hiệu ứng fade-in, slide-up khi cuộn trang bằng JS thuần

document.addEventListener('DOMContentLoaded', () => {
  const animatedEls = document.querySelectorAll('.recipe-card, .restaurant-card, .temple-card, .category-card, .step');
  animatedEls.forEach(el => {
    el.classList.add('hidden-anim');
  });
  window.addEventListener('scroll', revealOnScroll);
  revealOnScroll();
});

function revealOnScroll() {
  const animatedEls = document.querySelectorAll('.hidden-anim');
  animatedEls.forEach(el => {
    const rect = el.getBoundingClientRect();
    if (rect.top < window.innerHeight - 60) {
      el.classList.add('show-anim');
      el.classList.remove('hidden-anim');
    }
  });
}

// CSS animation classes sẽ được định nghĩa trong style.css
