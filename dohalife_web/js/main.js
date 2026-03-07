(function () {
  'use strict';

  function init() {
    DohaHeader.init();
    DohaScrollReveal.init();
    DohaMobileMenu.init();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
}());
