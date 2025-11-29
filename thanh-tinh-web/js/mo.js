// js/mo.js - Logic mõ chay mini, dùng chung cho mọi trang
(function() {
  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }
  ready(function() {
    var moMini = document.getElementById('moMini');
    var moMiniWrap = document.getElementById('moMiniWrap');
    var moSound = document.getElementById('moSound');
    var moHalo = document.getElementById('moHalo');
    var moPlus = document.getElementById('moPlus');
    if (!(moMini && moMiniWrap && moSound && moHalo && moPlus)) return;
    // Drag
    var isDragging = false, offsetX = 0, offsetY = 0;
    moMini.addEventListener('mousedown', function(e) {
      isDragging = true;
      offsetX = e.clientX - moMiniWrap.getBoundingClientRect().left;
      offsetY = e.clientY - moMiniWrap.getBoundingClientRect().top;
      moMini.style.cursor = 'grabbing';
    });
    document.addEventListener('mousemove', function(e) {
      if(isDragging) {
        moMiniWrap.style.right = '';
        moMiniWrap.style.bottom = '';
        moMiniWrap.style.left = (e.clientX - offsetX) + 'px';
        moMiniWrap.style.top = (e.clientY - offsetY) + 'px';
      }
    });
    document.addEventListener('mouseup', function() {
      if(isDragging) {
        isDragging = false;
        moMini.style.cursor = 'grab';
      }
    });
    // Click: play sound + hiệu ứng
    var lastClick = 0, clickCount = 0, clickTimeout;
    moMini.addEventListener('click', function(e) {
      if(isDragging) return;
      // Phát âm thanh: tạo Audio mới mỗi lần để đảm bảo có thể phát liên tiếp
      (function playOne() {
        try {
          var src = (moSound && (moSound.currentSrc || moSound.src)) ? (moSound.currentSrc || moSound.src) : 'assets/sounds/gomo.mp3';
          var s = new Audio(src);
          s.preload = 'auto';
          s.volume = 0.95;
          var p = s.play();
          if (p !== undefined) {
            p.catch(function(err) {
              // retry once after short delay
              setTimeout(function() { s.play().catch(function(){ console.warn('Không phát được âm thanh mõ (retry):', err); }); }, 200);
            });
          }
        } catch (e) {
          console.warn('Không phát được âm thanh mõ:', e);
        }
      })();
      // Hiệu ứng rung
      moMini.animate([
        { transform: 'scale(1) rotate(0deg)' },
        { transform: 'scale(1.15) rotate(-10deg)' },
        { transform: 'scale(0.95) rotate(10deg)' },
        { transform: 'scale(1) rotate(0deg)' }
      ], { duration: 400, easing: 'ease' });
      // Hiệu ứng hào quang
      moHalo.innerHTML = '';
      for(var i=0;i<10;i++){
        var ray = document.createElement('div');
        ray.style.position = 'absolute';
        ray.style.width = '6px';
        ray.style.height = '32px';
        ray.style.background = 'linear-gradient(180deg,#FFD700 60%,rgba(255,255,255,0) 100%)';
        ray.style.left = '50%';
        ray.style.top = '50%';
        ray.style.transform = 'translate(-50%,-100%) rotate('+(i*36)+'deg)';
        ray.style.borderRadius = '3px';
        ray.style.opacity = '0.8';
        ray.style.pointerEvents = 'none';
        moHalo.appendChild(ray);
        setTimeout((function(ray){ return function(){ ray.style.opacity='0'; }; })(ray),350);
      }
      // Hiệu ứng +1/+2
      var now = Date.now();
      if(now-lastClick<500){ clickCount++; } else { clickCount=1; }
      lastClick=now;
      moPlus.textContent = '+'+clickCount;
      moPlus.style.opacity = '1';
      clearTimeout(clickTimeout);
      clickTimeout = setTimeout(function(){moPlus.style.opacity='0';},700);
    });
    // Reset vị trí nếu không drag 10s
    var lastDrag = Date.now();
    document.addEventListener('mousemove', function() {
      if(isDragging) lastDrag = Date.now();
    });
    setInterval(function() {
      if(Date.now() - lastDrag > 10000 && !isDragging) {
        moMiniWrap.style.left = '';
        moMiniWrap.style.top = '';
        moMiniWrap.style.right = '2rem';
        moMiniWrap.style.bottom = '2rem';
      }
    }, 2000);
  });
})();
