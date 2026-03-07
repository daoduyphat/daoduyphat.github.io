var DohaMobileMenu = (function () {
  'use strict';

  var toggle = null;
  var nav = null;
  var links = null;
  var isOpen = false;

  function open() {
    isOpen = true;
    toggle.classList.add('is-open');
    toggle.setAttribute('aria-expanded', 'true');
    nav.classList.add('is-open');
    nav.setAttribute('aria-hidden', 'false');
  }

  function close() {
    isOpen = false;
    toggle.classList.remove('is-open');
    toggle.setAttribute('aria-expanded', 'false');
    nav.classList.remove('is-open');
    nav.setAttribute('aria-hidden', 'true');
  }

  function onKeydown(e) {
    if (e.key === 'Escape' && isOpen) {
      close();
    }
  }

  function init() {
    toggle = document.getElementById('nav-toggle');
    nav = document.getElementById('mobile-nav');
    if (!toggle || !nav) return;

    links = nav.querySelectorAll('.mobile-nav-link');

    toggle.addEventListener('click', function () {
      if (isOpen) {
        close();
      } else {
        open();
      }
    });

    links.forEach(function (link) {
      link.addEventListener('click', close);
    });

    document.addEventListener('keydown', onKeydown);
  }

  return { init: init };
}());
