// --- Snowfall effect ---
const snow = document.querySelector('.snow');
const ctx = snow.getContext('2d');
let W = window.innerWidth, H = window.innerHeight;
snow.width = W; snow.height = H;
let particles = [];
for(let i=0;i<120;i++){
  particles.push({x:Math.random()*W,y:Math.random()*H,r:Math.random()*3+1,d:Math.random()*W});
}
function drawSnow(){
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = 'rgba(255,255,255,0.8)';
  ctx.beginPath();
  for(let i=0;i<particles.length;i++){
    let p=particles[i];
    ctx.moveTo(p.x,p.y);
    ctx.arc(p.x,p.y,p.r,0,Math.PI*2,true);
  }
  ctx.fill();
  updateSnow();
}
let angle=0;
function updateSnow(){
  angle+=0.01;
  for(let i=0;i<particles.length;i++){
    let p=particles[i];
    p.y+=Math.cos(angle+p.d)+1+p.r/2;
    p.x+=Math.sin(angle)*2;
    if(p.x>W+5||p.x<-5||p.y>H){
      particles[i]={x:Math.random()*W,y:-10,r:p.r,d:p.d};
    }
  }
}
setInterval(drawSnow,33);
window.addEventListener('resize',()=>{W=window.innerWidth;H=window.innerHeight;snow.width=W;snow.height=H;});

// --- App State ---
window.ticketCodes = {};
window.validCodes = [];
async function loadTicketCodes() {
  try {
    const res = await fetch('data.json');
    const data = await res.json();
    window.ticketCodes = data;
    window.validCodes = Object.keys(data);
  } catch (e) {
    console.error('Failed to load ticket codes:', e);
    window.ticketCodes = {};
    window.validCodes = [];
  }
}
// Load ticket codes on startup
loadTicketCodes();
function getFingerprint(){
  return navigator.userAgent+screen.width+screen.height;
}
function hash(str){
  let h=5381;for(let i=0;i<str.length;i++){h=(h*33)^str.charCodeAt(i);}return Math.abs(h>>>0);
}
window.assignTicket = function assignTicket(code){
  if(!window.ticketCodes[code]){
    return 'invalid';
  }
  const claimedCodes = JSON.parse(localStorage.getItem('claimedCodes')||'[]');
  if(claimedCodes.includes(code)){
    return 'claimed';
  }
  claimedCodes.push(code);
  localStorage.setItem('claimedCodes',JSON.stringify(claimedCodes));
  localStorage.setItem('claimedTicketId',String(window.ticketCodes[code].id));
  localStorage.setItem('claimedCode',code);
  localStorage.setItem('nickname',code);
  return window.ticketCodes[code].id;
}
function clearApp(){
  document.getElementById('app').innerHTML='';
}
function showLanding(){
  clearApp();
  const app = document.getElementById('app');
    app.innerHTML = `
      <div class="center">
        <h1 style="font-weight:800;margin-bottom:0.5rem;text-align:center;display:flex;align-items:center;justify-content:center;gap:0.75rem;">
          <span style="display:inline-flex;align-items:center;justify-content:center;width:2.2rem;height:2.2rem;font-size:2.2rem;">🎄</span>
          Merry Christmas
        </h1>
        <p style="font-style:italic;margin-bottom:3rem;text-shadow:0 2px 10px rgba(0,0,0,0.5);text-align:center;padding:0 1rem;">A special secret is waiting for you</p>
        <button class="btn" onclick="window.location.href='claim.html'">
          <span style="display:inline-flex;align-items:center;justify-content:center;width:2rem;height:2rem;vertical-align:middle;">
            <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' width='28' height='28'><rect x='6' y='12' width='20' height='14' rx='4' fill='#FFD700'/><rect x='10' y='6' width='12' height='8' rx='2' fill='#C81D25'/><rect x='15' y='2' width='2' height='8' rx='1' fill='#FFD700'/></svg>
          </span>
          Open My Gift
        </button>
      </div>
    `;
}
function showClaim(){
  clearApp();
  document.getElementById('app').innerHTML = `
    <div class="center">
      <form class="glass" id="claimForm">
          <h2 style="text-align:center;margin-bottom:2rem;display:flex;align-items:center;justify-content:center;gap:0.75rem;white-space:nowrap;font-size:2rem;font-weight:800;">
            <span style="display:inline-flex;align-items:center;justify-content:center;width:2.2rem;height:2.2rem;font-size:2.2rem;">🎄</span>
            Claim Your Gift
          </h2>
        <input class="input" placeholder="Enter your code" id="nickname" required autocomplete="off" />
        <input class="input" placeholder="Re-enter code to confirm" id="code" required autocomplete="off" />
          <button class="btn" type="submit" style="width:100%;margin-top:0.5rem;">
            <span style="display:inline-flex;align-items:center;justify-content:center;width:1.5rem;height:1.5rem;vertical-align:middle;">
              <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' width='24' height='24'><rect x='6' y='12' width='20' height='14' rx='4' fill='#FFD700'/><rect x='10' y='6' width='12' height='8' rx='2' fill='#C81D25'/><rect x='15' y='2' width='2' height='8' rx='1' fill='#FFD700'/></svg>
            </span>
            Verify & Continue
          </button>
          <div id="claimError" style="color:#FFD700;text-align:center;margin-top:1rem;display:none;font-weight:600;">
            <span style="display:inline-flex;align-items:center;justify-content:center;width:1.2rem;height:1.2rem;vertical-align:middle;">
              <svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' width='18' height='18'><circle cx='16' cy='16' r='16' fill='#FFD700'/><path d='M10 10l12 12M22 10l-12 12' stroke='#C81D25' stroke-width='3' stroke-linecap='round'/></svg>
            </span>
            Invalid code or nickname!
          </div>
      </form>
    </div>
  `;
  const form = document.getElementById('claimForm');
  if(form) {
    form.onsubmit = async function(e){
      e.preventDefault();
      await loadTicketCodes();
      const nickname = document.getElementById('nickname').value.trim().toUpperCase();
      const code = document.getElementById('code').value.trim().toUpperCase();
      if(!window.validCodes.includes(code) || nickname !== code){
        const errorMsg = document.getElementById('claimError');
        form.classList.add('shake');
        errorMsg.textContent = '❌ Invalid code! Please check again.';
        errorMsg.style.display='block';
        setTimeout(()=>{form.classList.remove('shake');errorMsg.style.display='none';},600);
        return;
      }
      showGiftBox(code);
    };
  }
}
function showGiftBox(code){
  clearApp();
  document.getElementById('app').innerHTML = `
    <div class="center">
      <div id="giftBox" class="gift-box" style="margin-bottom:2rem;">
        <div style="width:160px;height:160px;background:linear-gradient(135deg,#C81D25,#FF6B6B,#FFD700);border-radius:1.5rem;box-shadow:0 8px 32px rgba(200,29,37,0.5);position:relative;display:flex;align-items:end;justify-content:center;">
          <div style="position:absolute;top:-20px;left:50%;transform:translateX(-50%);width:100px;height:40px;background:#FFD700;border-radius:1.5rem 1.5rem 0 0;border:5px solid #C81D25;box-shadow:0 4px 16px rgba(0,0,0,0.3);"></div>
          <div style="position:absolute;top:10px;left:50%;transform:translateX(-50%);width:40px;height:40px;background:#C81D25;border-radius:50%;border:3px solid #FFD700;"></div>
          <div style="position:absolute;top:20px;left:50%;transform:translateX(-50%);width:10px;height:40px;background:#FFD700;border-radius:5px;"></div>
        </div>
      </div>
      <p style="margin-top:2.5rem;font-size:1.3rem;font-weight:600;text-shadow:0 2px 10px rgba(0,0,0,0.5);text-align:center;word-break:keep-all;max-width:320px;margin-left:auto;margin-right:auto;">✨ Click the box to open your Christmas secret! ✨</p>
    </div>
  `;
  const giftBox = document.getElementById('giftBox');
  if(giftBox) {
    giftBox.onclick = function(){
      // Audio removed
      giftBox.classList.remove('gift-box');
      giftBox.innerHTML = '<div style="width:160px;height:160px;opacity:0.7;background:linear-gradient(135deg,#C81D25,#FFD700);border-radius:1.5rem;position:relative;animation:pulse 0.6s ease;"></div><div style="position:absolute;top:0;left:0;width:160px;height:160px;display:flex;align-items:center;justify-content:center;font-size:5rem;animation:fadeInUp 0.8s ease;">🎫</div>';
      setTimeout(()=>{
        const result = assignTicket(code);
        if(result==='claimed'){alert('This code has already been used!');window.location.href='claim.html';}
        else if(result==='invalid'){alert('Invalid code!');window.location.href='claim.html';}
        else{window.location.href='ticket.html';}
      },1200);
    };
  }
}
function showTicket(){
  const id = localStorage.getItem('claimedTicketId');
  const code = localStorage.getItem('claimedCode');
  if(!id||!code){
    location.hash = '';
    return;
  }
  const ticket = ticketCodes[code];
  clearApp();
  document.getElementById('app').innerHTML = `
    <div class="center">
      <div class="ticket-card">
        <h2 style="margin-bottom:1.5rem;font-size:2rem;">🎉 Congratulations!</h2>
        <img src="${ticket.qr}" alt="QR Code" />
        <h3 style="font-size:1.5rem;font-weight:700;margin-bottom:0.75rem;">${ticket.concert}</h3>
        <div style="font-size:1.2rem;margin-bottom:0.5rem;">📅 ${ticket.date}</div>
        <div style="font-size:1.1rem;margin-bottom:1.5rem;">📍 ${ticket.location}</div>
        <div style="padding:1.5rem;background:rgba(255,255,255,0.2);border-radius:1rem;margin-bottom:1.5rem;">
          <div style="font-size:1.2rem;margin-bottom:0.5rem;">This ticket belongs to:</div>
          <div style="font-size:1.5rem;font-weight:700;color:#FFD700;">${code}</div>
        </div>
        <button class="btn" id="saveQR" style="width:100%;">💾 Save QR Code</button>
      </div>
    </div>
  `;
  const saveBtn = document.getElementById('saveQR');
  if(saveBtn) {
    saveBtn.onclick = function(){
      const link = document.createElement('a');
      link.href = ticket.qr;
      link.download = `ticket-${code}.jpg`;
      link.click();
    };
  }
}
function showSoldOut(){
  clearApp();
  document.getElementById('app').innerHTML = `
    <div class="center">
      <div class="glass" style="text-align:center;max-width:500px;">
        <div style="font-size:4rem;margin-bottom:1rem;">🎫</div>
        <h2 style="font-size:2.2rem;font-weight:800;color:#FFD700;margin-bottom:1.5rem;">All Tickets Claimed!</h2>
        <p style="margin:1.5rem 0;font-size:1.2rem;line-height:1.8;">Sorry, all Christmas concert tickets have been claimed.<br/><br/>Wishing you a magical holiday season! 🎄✨</p>
        <button class="btn" id="backHome" style="width:100%;margin-top:1rem;">🏠 Back to Home</button>
      </div>
    </div>
  `;
  const backBtn = document.getElementById('backHome');
  if(backBtn) {
    backBtn.onclick = function() {
      window.location.href = 'index.html';
    };
  }
}
function route(){
  if(location.hash==='#claim'){showClaim();}
  else if(location.hash==='#ticket'){showTicket();}
  else if(location.hash==='#sold-out'){showSoldOut();}
  else{showLanding();}
}
window.addEventListener('hashchange',route);
route();
