var DohaScrollReveal = (function () {
  'use strict';

  function init() {
    var elements = document.querySelectorAll('.reveal');
    if (!elements.length) return;

    if (typeof IntersectionObserver === 'undefined') {
      elements.forEach(function (el) {
        el.classList.add('is-visible');
      });
      return;
    }

    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            entry.target.classList.add('is-visible');
            observer.unobserve(entry.target);
          }
        });
      },
      {
        threshold: 0.12,
        rootMargin: '0px 0px -60px 0px'
      }
    );

    elements.forEach(function (el) {
      observer.observe(el);
    });
  }

  return { init: init };
}());
