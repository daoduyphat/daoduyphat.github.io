var DohaHeader = (function () {
  'use strict';

  var header = null;
  var isScrolled = false;
  var threshold = 20;

  function handleScroll() {
    var scrolled = window.scrollY > threshold;
    if (scrolled === isScrolled) return;
    isScrolled = scrolled;
    if (isScrolled) {
      header.classList.add('is-scrolled');
    } else {
      header.classList.remove('is-scrolled');
    }
  }

  function init() {
    header = document.getElementById('site-header');
    if (!header) return;
    window.addEventListener('scroll', handleScroll, { passive: true });
    handleScroll();
  }

  return { init: init };
}());
