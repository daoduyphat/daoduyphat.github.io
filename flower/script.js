/**
 * =========================================================
 * 20/10 MAGICAL SUNFLOWER & ENCHANTED GARDEN EXPERIENCE
 * Full Procedural Blooming Canvas Animation
 * =========================================================
 */

(function () {
  'use strict';

  // ---------------------------------------------------------
  // 🌻 CẤU HÌNH TÊN NGƯỜI NHẬN (HARD-CODE)
  // Bạn có thể điền tên vào giữa 2 dấu ngoặc kép bên dưới:
  // Ví dụ: const HARDCODED_NAME = "Hà An";
  // ---------------------------------------------------------
  const HARDCODED_NAME = "Hà An";

  // ---------------------------------------------------------
  // 1. CONFIGURATION & CONSTANTS
  // ---------------------------------------------------------
  const GOLDEN_ANGLE = 137.507764 * (Math.PI / 180);

  const THEMES = {
    night: {
      name: 'night',
      icon: '🌙',
      skyTop: '#1a103c',
      skyBottom: '#050211',
      starAlpha: 0.8,
      sunGlow: 'rgba(251, 191, 36, 0.25)',
      sunburstAlpha: 0.35,
      fireflyCount: 40
    },
    sunset: {
      name: 'sunset',
      icon: '🌅',
      skyTop: '#4a154b',
      skyBottom: '#120324',
      starAlpha: 0.3,
      sunGlow: 'rgba(249, 115, 22, 0.35)',
      sunburstAlpha: 0.5,
      fireflyCount: 30
    },
    dawn: {
      name: 'dawn',
      icon: '☀️',
      skyTop: '#153e5c',
      skyBottom: '#06111e',
      starAlpha: 0.1,
      sunGlow: 'rgba(56, 189, 248, 0.3)',
      sunburstAlpha: 0.45,
      fireflyCount: 20
    }
  };

  // ---------------------------------------------------------
  // 2. STATE MANAGEMENT
  // ---------------------------------------------------------
  const state = {
    started: false,
    stage: 'idle', // 'idle' | 'sprouting' | 'blooming' | 'bursting' | 'garden'
    stageProgress: 0,
    time: 0,
    width: 0,
    height: 0,
    dpr: 1,
    currentTheme: 'night',
    recipient: HARDCODED_NAME,
    musicPlaying: false,
    mouse: { x: -1000, y: -1000, isDown: false, vx: 0, vy: 0 },
    wind: 0,
    windTarget: 0
  };

  // ---------------------------------------------------------
  // 3. DOM ELEMENTS
  // ---------------------------------------------------------
  const canvas = document.getElementById('gardenCanvas');
  const ctx = canvas.getContext('2d');

  const introScreen = document.getElementById('introScreen');
  const btnStart = document.getElementById('btnStart');
  const captionContainer = document.getElementById('captionContainer');
  const recipientNameEl = document.getElementById('recipientName');
  const interactiveHint = document.getElementById('interactiveHint');

  const btnMusic = document.getElementById('btnMusic');
  const btnTheme = document.getElementById('btnTheme');
  const themeIcon = document.getElementById('themeIcon');
  const btnReplay = document.getElementById('btnReplay');

  const btnViewPhoto = document.getElementById('btnViewPhoto');
  const photoModal = document.getElementById('photoModal');
  const btnClosePhoto = document.getElementById('btnClosePhoto');

  // ---------------------------------------------------------
  // 4. ENTITY ARRAYS
  // ---------------------------------------------------------
  let stars = [];
  let grassBlades = [];
  let heroSunflower = null;
  let gardenFlowers = [];
  let burstingSeeds = [];
  let fallingPetals = [];
  let butterflies = [];
  let fireflies = [];
  let sparkles = [];
  let shockwaves = [];
  let floatingBadges = [];

  // ---------------------------------------------------------
  // 5. WEB AUDIO SYNTHESIZER (Music Box / Piano)
  // ---------------------------------------------------------
  class SoundEngine {
    constructor() {
      this.ctx = null;
      this.isPlaying = false;
      this.timerId = null;
      this.noteIndex = 0;
    }

    init() {
      if (!this.ctx) {
        const AudioContext = window.AudioContext || window.webkitAudioContext;
        this.ctx = new AudioContext();
      }
      if (this.ctx.state === 'suspended') {
        this.ctx.resume();
      }
    }

    playNote(freq, time, duration = 1.6, type = 'sine', gainVal = 0.14) {
      if (!this.ctx) return;
      const osc = this.ctx.createOscillator();
      const oscHarmonic = this.ctx.createOscillator();
      const gainNode = this.ctx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, time);

      // Subtle harmonic for music-box chime warmth
      oscHarmonic.type = 'triangle';
      oscHarmonic.frequency.setValueAtTime(freq * 2, time);

      gainNode.gain.setValueAtTime(0, time);
      gainNode.gain.linearRampToValueAtTime(gainVal, time + 0.02);
      gainNode.gain.exponentialRampToValueAtTime(0.0001, time + duration);

      osc.connect(gainNode);
      oscHarmonic.connect(gainNode);
      gainNode.connect(this.ctx.destination);

      osc.start(time);
      oscHarmonic.start(time);
      osc.stop(time + duration);
      oscHarmonic.stop(time + duration);
    }

    // Gentle Sparkle Sound Effect on Click
    playChime() {
      if (!this.ctx) return;
      if (this.ctx.state === 'suspended') this.ctx.resume();
      const baseTime = this.ctx.currentTime;
      const pitches = [523.25, 659.25, 783.99, 1046.5, 1318.51]; // C5, E5, G5, C6, E6
      pitches.forEach((p, idx) => {
        this.playNote(p, baseTime + idx * 0.06, 0.8, 'sine', 0.08);
      });
    }

    // Romantic Arpeggio Music Box Loop (Canon in D / River Flows feeling)
    startMusic() {
      this.init();
      this.isPlaying = true;

      // Note frequency map
      const N = {
        C4: 261.63, D4: 293.66, E4: 329.63, F4: 349.23, G4: 392.00, A4: 440.00, B4: 493.88,
        C5: 523.25, D5: 587.33, E5: 659.25, F5: 698.46, G5: 783.99, A5: 880.00, B5: 987.77,
        C6: 1046.50
      };

      // Gentle melodic arpeggio sequence
      const melody = [
        // Measure 1: C major
        { f: N.C4, d: 2 }, { f: N.G4, d: 1 }, { f: N.C5, d: 1 }, { f: N.E5, d: 1.5 }, { f: N.G5, d: 1 },
        // Measure 2: G major
        { f: N.G4, d: 2 }, { f: N.D5, d: 1 }, { f: N.B4, d: 1 }, { f: N.D5, d: 1.5 }, { f: N.G5, d: 1 },
        // Measure 3: A minor
        { f: N.A4, d: 2 }, { f: N.E5, d: 1 }, { f: N.C5, d: 1 }, { f: N.E5, d: 1.5 }, { f: N.A5, d: 1 },
        // Measure 4: E minor
        { f: N.E4, d: 2 }, { f: N.B4, d: 1 }, { f: N.G4, d: 1 }, { f: N.B4, d: 1.5 }, { f: N.E5, d: 1 },
        // Measure 5: F major
        { f: N.F4, d: 2 }, { f: N.C5, d: 1 }, { f: N.A4, d: 1 }, { f: N.C5, d: 1.5 }, { f: N.F5, d: 1 },
        // Measure 6: C major
        { f: N.C4, d: 2 }, { f: N.G4, d: 1 }, { f: N.E5, d: 1 }, { f: N.G5, d: 1.5 }, { f: N.C6, d: 1 },
        // Measure 7: F major -> G
        { f: N.F4, d: 1.5 }, { f: N.A4, d: 1 }, { f: N.C5, d: 1 }, { f: N.G4, d: 1.5 }, { f: N.D5, d: 1 }, { f: N.B4, d: 1.5 },
        // Measure 8: Warm resolution
        { f: N.C5, d: 3 }, { f: N.E5, d: 2 }, { f: N.G5, d: 2.5 }
      ];

      this.noteIndex = 0;
      const step = () => {
        if (!this.isPlaying) return;
        const current = melody[this.noteIndex % melody.length];
        this.playNote(current.f, this.ctx.currentTime, current.d * 0.9, 'sine', 0.12);

        this.noteIndex++;
        // Tempo: ~340ms per note arpeggiation
        this.timerId = setTimeout(step, 360);
      };

      step();
    }

    stopMusic() {
      this.isPlaying = false;
      if (this.timerId) {
        clearTimeout(this.timerId);
        this.timerId = null;
      }
    }
  }

  const soundEngine = new SoundEngine();

  // ---------------------------------------------------------
  // 5b. YOUTUBE BACKGROUND MUSIC CONTROLLER
  // Track: "M-TP - Đừng Làm Trái Tim Anh Đau (Cinematic Version)"
  // URL: https://www.youtube.com/watch?v=PZ3Rvpzrva8
  // ---------------------------------------------------------
  const YOUTUBE_CONFIG = {
    videoId: 'PZ3Rvpzrva8',
    title: 'Đừng Làm Trái Tim Anh Đau (Cinematic Version)'
  };

  let ytPlayer = null;
  let ytReady = false;
  let userWantedPlay = false;

  function initYouTubePlayer() {
    if (window.YT && window.YT.Player) {
      createYTPlayer();
    } else {
      const oldReady = window.onYouTubeIframeAPIReady;
      window.onYouTubeIframeAPIReady = function () {
        if (typeof oldReady === 'function') oldReady();
        createYTPlayer();
      };
    }
  }

  function createYTPlayer() {
    if (ytPlayer) return;
    const container = document.getElementById('ytPlayer');
    if (!container || !window.YT || !window.YT.Player) return;

    const playerVars = {
      autoplay: 0,
      controls: 0,
      disablekb: 1,
      fs: 0,
      iv_load_policy: 3,
      modestbranding: 1,
      rel: 0,
      loop: 1,
      playlist: YOUTUBE_CONFIG.videoId,
      playsinline: 1,
      enablejsapi: 1
    };

    if (window.location && window.location.origin && window.location.protocol.startsWith('http')) {
      playerVars.origin = window.location.origin;
    }

    try {
      ytPlayer = new YT.Player('ytPlayer', {
        height: '150',
        width: '150',
        videoId: YOUTUBE_CONFIG.videoId,
        playerVars: playerVars,
        events: {
          onReady: function (event) {
            ytReady = true;
            try {
              event.target.setVolume(85);
              event.target.unMute();
            } catch (e) {}

            if (userWantedPlay) {
              try {
                event.target.playVideo();
                state.musicPlaying = true;
                btnMusic.classList.add('playing');
              } catch (e) {}
            }
          },
          onStateChange: function (event) {
            // YT.PlayerState.PLAYING = 1, PAUSED = 2, ENDED = 0
            if (event.data === 1) {
              state.musicPlaying = true;
              btnMusic.classList.add('playing');
            } else if (event.data === 2) {
              state.musicPlaying = false;
              btnMusic.classList.remove('playing');
            } else if (event.data === 0) {
              try {
                event.target.seekTo(0);
                event.target.playVideo();
              } catch (e) {}
            }
          },
          onError: function (e) {
            console.warn('YouTube Player error, falling back to Web Audio:', e);
            if (userWantedPlay && !state.musicPlaying) {
              soundEngine.startMusic();
              state.musicPlaying = true;
              btnMusic.classList.add('playing');
            }
          }
        }
      });
    } catch (err) {
      console.warn('Could not initialize YT.Player:', err);
    }
  }

  function playMusic() {
    userWantedPlay = true;
    state.musicPlaying = true;
    btnMusic.classList.add('playing');

    if (ytPlayer && ytReady && typeof ytPlayer.playVideo === 'function') {
      try {
        ytPlayer.unMute();
        ytPlayer.setVolume(85);
        ytPlayer.playVideo();
      } catch (e) {
        console.warn('ytPlayer.playVideo failed:', e);
      }
    } else {
      // Fallback: If YouTube API is blocked or takes too long, start Web Audio synth
      setTimeout(() => {
        if (userWantedPlay && !state.musicPlaying && (!ytPlayer || !ytReady)) {
          soundEngine.startMusic();
          state.musicPlaying = true;
          btnMusic.classList.add('playing');
        }
      }, 2500);
    }
  }

  function stopMusic() {
    userWantedPlay = false;
    state.musicPlaying = false;
    btnMusic.classList.remove('playing');

    if (ytPlayer && ytReady && typeof ytPlayer.pauseVideo === 'function') {
      try {
        ytPlayer.pauseVideo();
      } catch (e) {}
    }
    soundEngine.stopMusic();
  }

  function toggleMusic() {
    if (state.musicPlaying) {
      stopMusic();
    } else {
      playMusic();
    }
  }

  // ---------------------------------------------------------
  // 6. PROCEDURAL FLOWER CLASSES
  // ---------------------------------------------------------

  /**
   * Hero Sunflower: The magnificent centerpiece that blooms first!
   */
  class HeroSunflower {
    constructor(x, targetY) {
      this.baseX = x;
      this.currentX = x;
      this.baseY = state.height + 60;
      this.targetY = targetY;
      this.currentY = this.baseY;

      // Sizing
      const isMobile = state.width < 640;
      this.maxHeadRadius = isMobile ? Math.min(state.width * 0.22, 90) : Math.min(state.width * 0.16, 130);
      this.headRadius = 0;
      this.stemHeight = 0;
      this.targetStemHeight = this.baseY - this.targetY;

      // Stem curve control points
      this.stemCurve = (Math.random() - 0.5) * 60;
      this.swayAngle = 0;

      // Blooming animation progress (0 -> 1)
      this.stemProgress = 0;
      this.bloomProgress = 0;
      this.leaves = [
        { progress: 0, side: -1, yRatio: 0.35, size: this.maxHeadRadius * 0.75, angle: -0.6 },
        { progress: 0, side: 1, yRatio: 0.55, size: this.maxHeadRadius * 0.85, angle: 0.55 },
        { progress: 0, side: -1, yRatio: 0.72, size: this.maxHeadRadius * 0.9, angle: -0.5 },
        { progress: 0, side: 1, yRatio: 0.88, size: this.maxHeadRadius * 0.7, angle: 0.45 }
      ];

      // Petal specifications (Fibonacci layers)
      this.petalCountOuter = 34;
      this.petalCountInner = 28;
      this.seedCount = 320;

      // Sun halo rotation
      this.haloAngle = 0;
      this.haloPulse = 0;
      this.burstEmitted = false;
    }

    update(dt) {
      // 1. Stem Growth Phase
      if (this.stemProgress < 1) {
        this.stemProgress = Math.min(1, this.stemProgress + dt * 0.45);
        this.currentY = this.baseY - this.targetStemHeight * easeOutCubic(this.stemProgress);

        // Leaves grow as stem advances
        this.leaves.forEach(leaf => {
          if (this.stemProgress > leaf.yRatio) {
            const leafGrowth = (this.stemProgress - leaf.yRatio) / (1 - leaf.yRatio);
            leaf.progress = Math.min(1, easeOutCubic(leafGrowth * 2));
          }
        });
      } else if (this.bloomProgress < 1) {
        // 2. Flower Head Unfolding Phase
        this.bloomProgress = Math.min(1, this.bloomProgress + dt * 0.4);
        this.headRadius = this.maxHeadRadius * easeOutBack(this.bloomProgress);

        // When almost fully bloomed, trigger stage transition to bursting garden
        if (this.bloomProgress >= 0.92 && !this.burstEmitted) {
          this.burstEmitted = true;
          triggerGardenBurst(this.currentX, this.currentY);
        }
      }

      // Smoothly lower to frame the centered card once garden blooms
      if (this.burstEmitted) {
        const finalY = (state.width < 640) ? state.height * 0.72 : state.height * 0.68;
        this.currentY += (finalY - this.currentY) * dt * 0.9;
      }

      // Dynamic wind sway
      this.swayAngle = Math.sin(state.time * 1.5) * 0.04 + state.wind * 0.08;
      this.haloAngle += dt * 0.2;
      this.haloPulse = Math.sin(state.time * 2.5) * 0.08 + 1;
    }

    draw(ctx) {
      ctx.save();

      // Draw Stem
      if (this.stemProgress > 0) {
        ctx.beginPath();
        const startX = this.baseX;
        const startY = this.baseY;
        const headX = this.currentX + Math.sin(this.swayAngle) * (this.baseY - this.currentY);
        const headY = this.currentY;

        const cpX = (startX + headX) / 2 + this.stemCurve + state.wind * 40;
        const cpY = (startY + headY) / 2;

        ctx.moveTo(startX, startY);
        ctx.quadraticCurveTo(cpX, cpY, headX, headY);

        const stemGrad = ctx.createLinearGradient(startX, startY, headX, headY);
        stemGrad.addColorStop(0, '#155e2d');
        stemGrad.addColorStop(0.5, '#22c55e');
        stemGrad.addColorStop(1, '#86efac');

        ctx.strokeStyle = stemGrad;
        ctx.lineWidth = Math.max(7, this.maxHeadRadius * 0.14);
        ctx.lineCap = 'round';
        ctx.stroke();

        // Draw Leaves
        this.leaves.forEach(leaf => {
          if (leaf.progress > 0) {
            const t = leaf.yRatio * this.stemProgress;
            // Approximate point on quadratic bezier
            const lx = Math.pow(1 - t, 2) * startX + 2 * (1 - t) * t * cpX + Math.pow(t, 2) * headX;
            const ly = Math.pow(1 - t, 2) * startY + 2 * (1 - t) * t * cpY + Math.pow(t, 2) * headY;

            this.drawLeaf(ctx, lx, ly, leaf, headX, headY);
          }
        });
      }

      // Draw Flower Head
      if (this.bloomProgress > 0) {
        const headX = this.currentX + Math.sin(this.swayAngle) * (this.baseY - this.currentY);
        const headY = this.currentY;

        ctx.save();
        ctx.translate(headX, headY);
        ctx.rotate(this.swayAngle * 0.6);

        // 1. Sunburst Halo behind petals
        this.drawSunHalo(ctx);

        // 2. Green Calyx Sepals
        this.drawSepals(ctx);

        // 3. Golden Sunflower Petals (Outer & Inner Layers)
        this.drawPetals(ctx);

        // 4. Fibonacci Spiral Seed Disc (Phyllotaxis)
        this.drawSeedDisc(ctx);

        // 5. Center Golden Shimmer Aura
        this.drawCenterShimmer(ctx);

        ctx.restore();
      }

      ctx.restore();
    }

    drawLeaf(ctx, x, y, leaf, headX, headY) {
      ctx.save();
      ctx.translate(x, y);

      const leafSway = leaf.angle + Math.sin(state.time * 2 + leaf.yRatio * 5) * 0.08;
      ctx.rotate(leafSway * leaf.side);
      ctx.scale(leaf.progress, leaf.progress);

      const w = leaf.size;
      const h = leaf.size * 0.55;

      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.bezierCurveTo(w * 0.3 * leaf.side, -h * 0.6, w * 0.8 * leaf.side, -h * 0.4, w * leaf.side, 0);
      ctx.bezierCurveTo(w * 0.8 * leaf.side, h * 0.6, w * 0.3 * leaf.side, h * 0.4, 0, 0);

      const leafGrad = ctx.createLinearGradient(0, 0, w * leaf.side, 0);
      leafGrad.addColorStop(0, '#15803d');
      leafGrad.addColorStop(0.6, '#22c55e');
      leafGrad.addColorStop(1, '#4ade80');
      ctx.fillStyle = leafGrad;
      ctx.fill();

      // Leaf main vein
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.quadraticCurveTo(w * 0.5 * leaf.side, -h * 0.05, w * leaf.side, 0);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 2;
      ctx.stroke();

      ctx.restore();
    }

    drawSunHalo(ctx) {
      const themeConfig = THEMES[state.currentTheme];
      const haloRadius = this.maxHeadRadius * 1.8 * this.bloomProgress * this.haloPulse;

      const haloGrad = ctx.createRadialGradient(0, 0, this.maxHeadRadius * 0.3, 0, 0, haloRadius);
      haloGrad.addColorStop(0, 'rgba(251, 191, 36, 0.4)');
      haloGrad.addColorStop(0.5, 'rgba(245, 158, 11, 0.2)');
      haloGrad.addColorStop(1, 'rgba(251, 191, 36, 0)');

      ctx.save();
      ctx.fillStyle = haloGrad;
      ctx.beginPath();
      ctx.arc(0, 0, haloRadius, 0, Math.PI * 2);
      ctx.fill();

      // Radiant sun rays
      const rayCount = 16;
      ctx.rotate(this.haloAngle);
      for (let i = 0; i < rayCount; i++) {
        const ang = (i * 2 * Math.PI) / rayCount;
        ctx.save();
        ctx.rotate(ang);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.lineTo(-haloRadius * 0.1, haloRadius);
        ctx.lineTo(haloRadius * 0.1, haloRadius);
        ctx.closePath();
        ctx.fillStyle = `rgba(253, 224, 71, ${0.08 * themeConfig.sunburstAlpha * this.bloomProgress})`;
        ctx.fill();
        ctx.restore();
      }
      ctx.restore();
    }

    drawSepals(ctx) {
      const sepalCount = 20;
      const sepalLen = this.headRadius * 0.65;
      ctx.save();
      for (let i = 0; i < sepalCount; i++) {
        const angle = (i * 2 * Math.PI) / sepalCount;
        ctx.save();
        ctx.rotate(angle);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.bezierCurveTo(-sepalLen * 0.2, sepalLen * 0.5, -sepalLen * 0.2, sepalLen * 0.8, 0, sepalLen);
        ctx.bezierCurveTo(sepalLen * 0.2, sepalLen * 0.8, sepalLen * 0.2, sepalLen * 0.5, 0, 0);
        ctx.fillStyle = '#166534';
        ctx.fill();
        ctx.restore();
      }
      ctx.restore();
    }

    drawPetals(ctx) {
      const maxLen = this.headRadius;
      const petalWidth = maxLen * 0.28;

      // 1. Outer Layer Petals (Bright Sunny Golden Yellow)
      for (let i = 0; i < this.petalCountOuter; i++) {
        const angle = (i * 2 * Math.PI) / this.petalCountOuter;
        this.renderSinglePetal(ctx, angle, maxLen, petalWidth, '#d97706', '#fbbf24', '#fef08a');
      }

      // 2. Inner Layer Petals (Staggered angle for rich lush depth)
      const innerOffset = Math.PI / this.petalCountInner;
      for (let i = 0; i < this.petalCountInner; i++) {
        const angle = (i * 2 * Math.PI) / this.petalCountInner + innerOffset;
        this.renderSinglePetal(ctx, angle, maxLen * 0.85, petalWidth * 0.9, '#b45309', '#f59e0b', '#fde047');
      }
    }

    renderSinglePetal(ctx, angle, len, width, cBase, cMid, cTip) {
      ctx.save();
      ctx.rotate(angle);

      ctx.beginPath();
      ctx.moveTo(0, 0);
      // Beautiful sculpted petal curve
      ctx.bezierCurveTo(-width * 0.65, len * 0.35, -width * 0.8, len * 0.75, 0, len);
      ctx.bezierCurveTo(width * 0.8, len * 0.75, width * 0.65, len * 0.35, 0, 0);

      const grad = ctx.createLinearGradient(0, 0, 0, len);
      grad.addColorStop(0, cBase);
      grad.addColorStop(0.45, cMid);
      grad.addColorStop(1, cTip);

      ctx.fillStyle = grad;
      ctx.fill();

      // Delicate central petal vein highlight
      ctx.beginPath();
      ctx.moveTo(0, len * 0.1);
      ctx.lineTo(0, len * 0.85);
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.lineWidth = 1.2;
      ctx.stroke();

      ctx.restore();
    }

    drawSeedDisc(ctx) {
      const discMaxRadius = this.headRadius * 0.44;
      const currentDiscRadius = discMaxRadius * this.bloomProgress;
      const c = currentDiscRadius / Math.sqrt(this.seedCount);

      // Dark brown disc base
      ctx.beginPath();
      ctx.arc(0, 0, currentDiscRadius + 3, 0, Math.PI * 2);
      ctx.fillStyle = '#261208';
      ctx.fill();

      // Phyllotaxis Fibonacci Seeds
      const activeSeeds = Math.floor(this.seedCount * this.bloomProgress);
      for (let n = 0; n < activeSeeds; n++) {
        const theta = n * GOLDEN_ANGLE;
        const r = c * Math.sqrt(n);

        const x = r * Math.cos(theta);
        const y = r * Math.sin(theta);

        const seedR = Math.max(1, (r / discMaxRadius) * 2.8 + 1.2);

        ctx.beginPath();
        ctx.arc(x, y, seedR, 0, Math.PI * 2);

        // Gradient from center deep dark espresso to golden pollen around perimeter
        const ratio = r / discMaxRadius;
        if (ratio > 0.75) {
          ctx.fillStyle = n % 2 === 0 ? '#fbbf24' : '#d97706'; // Golden pollen rim
        } else if (ratio > 0.4) {
          ctx.fillStyle = n % 2 === 0 ? '#542611' : '#3b1d0c';
        } else {
          ctx.fillStyle = n % 2 === 0 ? '#1f0d05' : '#2b1307';
        }
        ctx.fill();
      }
    }

    drawCenterShimmer(ctx) {
      if (this.bloomProgress < 0.8) return;
      const shimmerR = this.headRadius * 0.4 * (this.bloomProgress - 0.7) * 3.3;
      const grad = ctx.createRadialGradient(0, 0, 0, 0, 0, shimmerR);
      grad.addColorStop(0, 'rgba(254, 240, 138, 0.4)');
      grad.addColorStop(0.7, 'rgba(251, 191, 36, 0.2)');
      grad.addColorStop(1, 'rgba(251, 191, 36, 0)');

      ctx.beginPath();
      ctx.arc(0, 0, shimmerR, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();
    }
  }

  /**
   * Garden Flower Class: Multi-colored flowers that sprout from MULTIPLE ANGLES & CORNERS!
   * Variety: Sunflower, Rose, Tulip, Cosmos/Daisy, Hydrangea/Forget-me-not, Lotus
   * Origins: 'bottom', 'bottom-left', 'bottom-right', 'side-left', 'side-right', 'top-left', 'top-right'
   */
  class GardenFlower {
    constructor(targetX, targetY, type = 'random', options = {}) {
      this.targetX = targetX;
      this.targetY = targetY;
      this.targetXRatio = targetX / (state.width || 1);
      this.targetYRatio = targetY / (state.height || 1);

      // Species Selection
      const types = ['sunflower', 'rose', 'tulip', 'cosmos', 'forgetmenot', 'lotus'];
      this.type = type === 'random' ? types[Math.floor(Math.random() * types.length)] : type;

      // Color themes for variety
      this.color = options.color || this.pickColorForType(this.type);

      // Sizing
      const sizeScale = options.scale || (0.45 + Math.random() * 0.5);
      this.maxRadius = (35 + Math.random() * 28) * sizeScale;
      this.currentRadius = 0;

      // Origin point setup (growing from multiple corners and sides!)
      this.origin = options.origin || this.detectOrigin(targetX, targetY);
      this.initOrigin();

      this.swayOffset = Math.random() * Math.PI * 2;
      this.swaySpeed = 1.2 + Math.random() * 1.5;

      // Growth progress
      this.progress = 0;
      this.growthSpeed = 0.55 + Math.random() * 0.5;
      this.delay = options.delay || 0;
      this.bloomed = false;
    }

    detectOrigin(x, y) {
      const w = state.width || 800;
      const h = state.height || 600;

      if (x < w * 0.25 && y > h * 0.65) return 'bottom-left';
      if (x > w * 0.75 && y > h * 0.65) return 'bottom-right';
      if (x < w * 0.18 && y < h * 0.3) return 'top-left';
      if (x > w * 0.82 && y < h * 0.3) return 'top-right';
      if (x < w * 0.22) return 'side-left';
      if (x > w * 0.78) return 'side-right';
      return 'bottom';
    }

    initOrigin() {
      const w = state.width;
      const h = state.height;

      switch (this.origin) {
        case 'bottom-left':
          this.startX = -15 + Math.random() * 40;
          this.startY = h + 25;
          this.curveX = 40 + Math.random() * 40;
          this.curveY = -25 - Math.random() * 30;
          break;
        case 'bottom-right':
          this.startX = w + 15 - Math.random() * 40;
          this.startY = h + 25;
          this.curveX = -40 - Math.random() * 40;
          this.curveY = -25 - Math.random() * 30;
          break;
        case 'side-left':
          this.startX = -25;
          this.startY = Math.min(h - 20, this.targetY + 40 + Math.random() * 60);
          this.curveX = 45 + Math.random() * 35;
          this.curveY = -15;
          break;
        case 'side-right':
          this.startX = w + 25;
          this.startY = Math.min(h - 20, this.targetY + 40 + Math.random() * 60);
          this.curveX = -45 - Math.random() * 35;
          this.curveY = -15;
          break;
        case 'top-left':
          this.startX = -20;
          this.startY = -20;
          this.curveX = 30 + Math.random() * 30;
          this.curveY = 30 + Math.random() * 20;
          break;
        case 'top-right':
          this.startX = w + 20;
          this.startY = -20;
          this.curveX = -30 - Math.random() * 30;
          this.curveY = 30 + Math.random() * 20;
          break;
        case 'bottom':
        default:
          this.startX = this.targetX + (Math.random() - 0.5) * 35;
          this.startY = h + 35;
          this.curveX = (Math.random() - 0.5) * 40;
          this.curveY = -10;
          break;
      }
      this.currentX = this.startX;
      this.currentY = this.startY;
    }

    onResize() {
      this.targetX = this.targetXRatio * state.width;
      this.targetY = this.targetYRatio * state.height;
      this.initOrigin();
      if (this.progress >= 1) {
        this.currentX = this.targetX;
        this.currentY = this.targetY;
      }
    }

    pickColorForType(t) {
      const palette = {
        rose: [
          { base: '#991b1b', mid: '#e11d48', tip: '#fda4af' }, // Velvet red
          { base: '#be185d', mid: '#f43f5e', tip: '#fbcfe8' }, // Romantic pink
          { base: '#c2410c', mid: '#fb923c', tip: '#ffedd5' }  // Coral blush
        ],
        tulip: [
          { base: '#581c87', mid: '#a855f7', tip: '#f3e8ff' }, // Royal purple
          { base: '#9d174d', mid: '#ec4899', tip: '#fce7f3' }, // Radiant magenta
          { base: '#b45309', mid: '#eab308', tip: '#fef9c3' }  // Sunbeam yellow
        ],
        cosmos: [
          { base: '#be185d', mid: '#f472b6', tip: '#ffffff' }, // Pink daisy
          { base: '#4338ca', mid: '#818cf8', tip: '#e0e7ff' }, // Lavender cosmos
          { base: '#e2e8f0', mid: '#ffffff', tip: '#fef08a' }  // Pure white daisy
        ],
        forgetmenot: [
          { base: '#0369a1', mid: '#38bdf8', tip: '#e0f2fe' }, // Sky cyan
          { base: '#1d4ed8', mid: '#60a5fa', tip: '#dbeafe' }  // Deep ocean blue
        ],
        lotus: [
          { base: '#831843', mid: '#ec4899', tip: '#ffffff' }, // Sacred pink lotus
          { base: '#9f1239', mid: '#fb7185', tip: '#fff1f2' }  // Blush blossom
        ],
        sunflower: [
          { base: '#b45309', mid: '#fbbf24', tip: '#fef08a' }  // Golden mini
        ]
      };
      const choices = palette[t] || palette.rose;
      return choices[Math.floor(Math.random() * choices.length)];
    }

    update(dt) {
      if (this.delay > 0) {
        this.delay -= dt;
        return;
      }

      if (this.progress < 1) {
        this.progress = Math.min(1, this.progress + dt * this.growthSpeed);
        const p = easeOutCubic(this.progress);
        this.currentX = this.startX + (this.targetX - this.startX) * p;
        this.currentY = this.startY + (this.targetY - this.startY) * p;
        this.currentRadius = this.maxRadius * easeOutBack(Math.max(0, (this.progress - 0.4) / 0.6));
      }
    }

    draw(ctx) {
      if (this.progress <= 0) return;

      const sway = Math.sin(state.time * this.swaySpeed + this.swayOffset) * 0.05 + state.wind * 0.12;

      ctx.save();
      // Draw Stem
      const startX = this.startX;
      const startY = this.startY;
      const headX = this.currentX + Math.sin(sway) * 12;
      const headY = this.currentY + Math.cos(sway) * 4;

      const cpX = (startX + headX) / 2 + this.curveX + state.wind * 20;
      const cpY = (startY + headY) / 2 + this.curveY;

      ctx.beginPath();
      ctx.moveTo(startX, startY);
      ctx.quadraticCurveTo(cpX, cpY, headX, headY);
      ctx.strokeStyle = '#15803d';
      ctx.lineWidth = Math.max(3, this.maxRadius * 0.12);
      ctx.lineCap = 'round';
      ctx.stroke();

      // Draw Bloom when stem reached high enough
      if (this.progress > 0.35 && this.currentRadius > 1) {
        ctx.save();
        ctx.translate(headX, headY);

        // Align flower direction with curved stem growth tangent
        const stemAngle = Math.atan2(headY - cpY, headX - cpX);
        ctx.rotate(stemAngle - Math.PI / 2 + sway * 0.5);

        switch (this.type) {
          case 'rose':
            this.drawRose(ctx);
            break;
          case 'tulip':
            this.drawTulip(ctx);
            break;
          case 'cosmos':
            this.drawCosmos(ctx);
            break;
          case 'forgetmenot':
            this.drawForgetMeNot(ctx);
            break;
          case 'lotus':
            this.drawLotus(ctx);
            break;
          case 'sunflower':
          default:
            this.drawMiniSunflower(ctx);
            break;
        }

        ctx.restore();
      }

      ctx.restore();
    }

    drawMiniSunflower(ctx) {
      const r = this.currentRadius;
      const count = 18;
      for (let i = 0; i < count; i++) {
        const ang = (i * 2 * Math.PI) / count;
        ctx.save();
        ctx.rotate(ang);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.bezierCurveTo(-r * 0.2, r * 0.4, -r * 0.3, r * 0.8, 0, r);
        ctx.bezierCurveTo(r * 0.3, r * 0.8, r * 0.2, r * 0.4, 0, 0);
        ctx.fillStyle = this.color.mid;
        ctx.fill();
        ctx.restore();
      }

      // Disc
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.4, 0, Math.PI * 2);
      ctx.fillStyle = '#3b1d0c';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(0, 0, r * 0.25, 0, Math.PI * 2);
      ctx.fillStyle = '#1c0a03';
      ctx.fill();
    }

    drawRose(ctx) {
      const r = this.currentRadius;
      const layers = [
        { count: 7, len: r, w: r * 0.55, col: this.color.base },
        { count: 6, len: r * 0.8, w: r * 0.45, col: this.color.mid },
        { count: 5, len: r * 0.6, w: r * 0.35, col: this.color.tip }
      ];

      layers.forEach((layer, lIdx) => {
        const offset = (lIdx * Math.PI) / layer.count;
        for (let i = 0; i < layer.count; i++) {
          const ang = (i * 2 * Math.PI) / layer.count + offset;
          ctx.save();
          ctx.rotate(ang);
          ctx.beginPath();
          ctx.moveTo(0, 0);
          ctx.bezierCurveTo(-layer.w, layer.len * 0.5, layer.w, layer.len * 0.9, 0, layer.len);
          ctx.bezierCurveTo(-layer.w * 0.5, layer.len * 0.7, layer.w * 0.5, layer.len * 0.2, 0, 0);
          ctx.fillStyle = layer.col;
          ctx.fill();
          ctx.restore();
        }
      });

      // Rose core spiral bud
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.22, 0, Math.PI * 2);
      ctx.fillStyle = this.color.base;
      ctx.fill();
    }

    drawTulip(ctx) {
      const r = this.currentRadius * 1.1;
      const w = r * 0.45;

      // Back petals
      [-0.4, 0.4].forEach(rot => {
        ctx.save();
        ctx.rotate(rot);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.bezierCurveTo(-w * 0.8, -r * 0.6, -w * 0.4, -r, 0, -r);
        ctx.bezierCurveTo(w * 0.4, -r, w * 0.8, -r * 0.6, 0, 0);
        ctx.fillStyle = this.color.base;
        ctx.fill();
        ctx.restore();
      });

      // Front center petals
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.bezierCurveTo(-w, -r * 0.5, -w * 0.5, -r * 0.95, 0, -r * 0.95);
      ctx.bezierCurveTo(w * 0.5, -r * 0.95, w, -r * 0.5, 0, 0);
      const grad = ctx.createLinearGradient(0, 0, 0, -r);
      grad.addColorStop(0, this.color.mid);
      grad.addColorStop(1, this.color.tip);
      ctx.fillStyle = grad;
      ctx.fill();
    }

    drawCosmos(ctx) {
      const r = this.currentRadius;
      const count = 8;
      const w = r * 0.28;

      for (let i = 0; i < count; i++) {
        const ang = (i * 2 * Math.PI) / count;
        ctx.save();
        ctx.rotate(ang);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.bezierCurveTo(-w, r * 0.5, -w * 0.8, r * 0.9, 0, r);
        ctx.bezierCurveTo(w * 0.8, r * 0.9, w, r * 0.5, 0, 0);

        const grad = ctx.createLinearGradient(0, 0, 0, r);
        grad.addColorStop(0, this.color.base);
        grad.addColorStop(0.7, this.color.mid);
        grad.addColorStop(1, this.color.tip);
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.restore();
      }

      // Yellow Center Disc
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.25, 0, Math.PI * 2);
      ctx.fillStyle = '#eab308';
      ctx.fill();

      // Pollen dots
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.15, 0, Math.PI * 2);
      ctx.fillStyle = '#a16207';
      ctx.fill();
    }

    drawForgetMeNot(ctx) {
      const r = this.currentRadius * 0.9;
      const count = 5;
      const w = r * 0.42;

      for (let i = 0; i < count; i++) {
        const ang = (i * 2 * Math.PI) / count;
        ctx.save();
        ctx.rotate(ang);
        ctx.beginPath();
        ctx.arc(0, r * 0.5, w * 0.7, 0, Math.PI * 2);
        ctx.fillStyle = this.color.mid;
        ctx.fill();
        ctx.restore();
      }

      // Center white ring & yellow eye
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.2, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();

      ctx.beginPath();
      ctx.arc(0, 0, r * 0.1, 0, Math.PI * 2);
      ctx.fillStyle = '#fbbf24';
      ctx.fill();
    }

    drawLotus(ctx) {
      const r = this.currentRadius * 1.1;
      const w = r * 0.35;
      const count = 12;

      for (let i = 0; i < count; i++) {
        const ang = (i * 2 * Math.PI) / count;
        ctx.save();
        ctx.rotate(ang);
        ctx.beginPath();
        ctx.moveTo(0, 0);
        ctx.bezierCurveTo(-w, r * 0.4, -w * 0.5, r * 0.85, 0, r);
        ctx.bezierCurveTo(w * 0.5, r * 0.85, w, r * 0.4, 0, 0);

        const grad = ctx.createLinearGradient(0, 0, 0, r);
        grad.addColorStop(0, this.color.base);
        grad.addColorStop(0.5, this.color.mid);
        grad.addColorStop(1, '#ffffff');
        ctx.fillStyle = grad;
        ctx.fill();
        ctx.restore();
      }

      // Golden center
      ctx.beginPath();
      ctx.arc(0, 0, r * 0.28, 0, Math.PI * 2);
      ctx.fillStyle = '#f59e0b';
      ctx.fill();
    }
  }

  // ---------------------------------------------------------
  // 7. PARTICLES: SEED BURST, PETALS, BUTTERFLIES, FIREFLIES
  // ---------------------------------------------------------

  /**
   * Pollen/Seed Burst: Erupts from blooming sunflower and plants garden!
   */
  class BurstingSeed {
    constructor(x, y) {
      this.x = x;
      this.y = y;
      const angle = Math.random() * Math.PI * 2;
      const speed = Math.random() * 9 + 4;
      this.vx = Math.cos(angle) * speed;
      this.vy = Math.sin(angle) * speed - 4; // Upward bias
      this.gravity = 0.16;
      this.alpha = 1;
      this.size = Math.random() * 3 + 2;
      this.color = Math.random() > 0.4 ? '#fde047' : '#fb923c';
      this.targetY = state.height - (Math.random() * (state.height * 0.35) + 20);
      this.hasPlanted = false;
    }

    update(dt) {
      this.x += this.vx;
      this.y += this.vy;
      this.vy += this.gravity;
      this.vx *= 0.98;

      // Seed touches ground -> sprout flower!
      if (this.y >= this.targetY && !this.hasPlanted) {
        this.hasPlanted = true;
        this.alpha = 0;
        plantFlower(this.x, this.targetY);
      }
    }

    draw(ctx) {
      if (this.alpha <= 0) return;
      ctx.save();
      ctx.globalAlpha = this.alpha;
      ctx.fillStyle = this.color;
      ctx.shadowColor = '#fbbf24';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  /**
   * Falling Petals drifting in breeze
   */
  class FallingPetal {
    constructor() {
      this.reset(true);
    }

    reset(initial = false) {
      this.x = Math.random() * state.width;
      this.y = initial ? Math.random() * state.height : -30;
      this.size = Math.random() * 12 + 8;
      this.speedY = Math.random() * 1.2 + 0.8;
      this.speedX = Math.random() * 1.5 - 0.5;
      this.angle = Math.random() * Math.PI * 2;
      this.rotSpeed = (Math.random() - 0.5) * 0.04;
      this.color = Math.random() > 0.5
        ? { base: '#fbbf24', tip: '#fef08a' } // Sunflower petal
        : { base: '#f43f5e', tip: '#fbcfe8' }; // Rose petal
    }

    update() {
      this.y += this.speedY;
      this.x += this.speedX + state.wind * 2.5;
      this.angle += this.rotSpeed;

      if (this.y > state.height + 40 || this.x < -40 || this.x > state.width + 40) {
        this.reset();
      }
    }

    draw(ctx) {
      ctx.save();
      ctx.translate(this.x, this.y);
      ctx.rotate(this.angle);

      ctx.beginPath();
      ctx.moveTo(0, -this.size * 0.5);
      ctx.bezierCurveTo(-this.size * 0.4, 0, -this.size * 0.3, this.size * 0.5, 0, this.size * 0.5);
      ctx.bezierCurveTo(this.size * 0.3, this.size * 0.5, this.size * 0.4, 0, 0, -this.size * 0.5);

      const grad = ctx.createLinearGradient(0, -this.size * 0.5, 0, this.size * 0.5);
      grad.addColorStop(0, this.color.tip);
      grad.addColorStop(1, this.color.base);

      ctx.fillStyle = grad;
      ctx.globalAlpha = 0.85;
      ctx.fill();
      ctx.restore();
    }
  }

  /**
   * 3D Fluttering Butterfly with wandering physics
   */
  class Butterfly {
    constructor() {
      this.x = Math.random() * state.width;
      this.y = Math.random() * (state.height * 0.6) + state.height * 0.2;
      this.targetX = this.x;
      this.targetY = this.y;
      this.vx = 0;
      this.vy = 0;
      this.wingAngle = 0;
      this.wingSpeed = 16 + Math.random() * 6;
      this.size = Math.random() * 8 + 14;

      // Color species
      const types = [
        { main: '#38bdf8', border: '#0369a1', spot: '#ffffff' }, // Blue Morpho
        { main: '#fb923c', border: '#7c2d12', spot: '#ffffff' }, // Monarch Orange
        { main: '#f472b6', border: '#831843', spot: '#fef08a' }  // Pink Swallowtail
      ];
      this.color = types[Math.floor(Math.random() * types.length)];
      this.changeTargetTimer = 0;
    }

    update(dt) {
      this.changeTargetTimer -= dt;
      if (this.changeTargetTimer <= 0) {
        this.changeTargetTimer = Math.random() * 3 + 2;
        // Wander toward a flower or open area
        if (gardenFlowers.length > 0 && Math.random() > 0.3) {
          const flower = gardenFlowers[Math.floor(Math.random() * gardenFlowers.length)];
          this.targetX = flower.x + (Math.random() - 0.5) * 60;
          this.targetY = flower.targetY - Math.random() * 50 - 20;
        } else {
          this.targetX = Math.random() * (state.width - 100) + 50;
          this.targetY = Math.random() * (state.height * 0.6) + state.height * 0.2;
        }
      }

      // Smooth steering toward target
      const dx = this.targetX - this.x;
      const dy = this.targetY - this.y;
      this.vx += dx * 0.015;
      this.vy += dy * 0.015;

      // Add gentle sine wave flutter & wind
      this.vy += Math.sin(state.time * 8) * 0.4;
      this.vx += state.wind * 1.5;

      this.vx *= 0.92;
      this.vy *= 0.92;

      this.x += this.vx;
      this.y += this.vy;

      // Flapping wings
      this.wingAngle += this.wingSpeed * dt;
    }

    draw(ctx) {
      ctx.save();
      ctx.translate(this.x, this.y);

      // Rotate based on velocity direction
      const angle = Math.atan2(this.vy, this.vx);
      ctx.rotate(angle * 0.25);

      // 3D wing fold scale
      const wingScale = Math.cos(this.wingAngle);

      // Draw Left Wing
      ctx.save();
      ctx.scale(wingScale, 1);
      this.drawWings(ctx, -1);
      ctx.restore();

      // Draw Right Wing
      ctx.save();
      ctx.scale(-wingScale, 1);
      this.drawWings(ctx, 1);
      ctx.restore();

      // Body & Antennae
      ctx.beginPath();
      ctx.ellipse(0, 0, 1.8, this.size * 0.35, 0, 0, Math.PI * 2);
      ctx.fillStyle = '#1e1b4b';
      ctx.fill();

      ctx.restore();
    }

    drawWings(ctx, dir) {
      const s = this.size;

      // Forewing (Upper)
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.bezierCurveTo(s * 0.5, -s * 0.8, s * 1.2, -s * 0.6, s, 0);
      ctx.bezierCurveTo(s * 0.8, s * 0.2, s * 0.3, s * 0.1, 0, 0);
      ctx.fillStyle = this.color.main;
      ctx.fill();
      ctx.strokeStyle = this.color.border;
      ctx.lineWidth = 1.2;
      ctx.stroke();

      // Wing spots
      ctx.beginPath();
      ctx.arc(s * 0.65, -s * 0.2, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = this.color.spot;
      ctx.fill();

      // Hindwing (Lower)
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.bezierCurveTo(s * 0.4, s * 0.3, s * 0.7, s * 0.7, s * 0.4, s * 0.85);
      ctx.bezierCurveTo(s * 0.1, s * 0.7, 0, s * 0.4, 0, 0);
      ctx.fillStyle = this.color.main;
      ctx.fill();
      ctx.stroke();
    }
  }

  /**
   * Ambient Fireflies / Fairy Dust
   */
  class Firefly {
    constructor() {
      this.reset();
    }

    reset() {
      this.x = Math.random() * state.width;
      this.y = Math.random() * (state.height * 0.7) + state.height * 0.25;
      this.vx = (Math.random() - 0.5) * 0.6;
      this.vy = (Math.random() - 0.5) * 0.6;
      this.size = Math.random() * 2.2 + 1;
      this.alpha = Math.random();
      this.alphaSpeed = Math.random() * 2 + 1;
    }

    update(dt) {
      this.x += this.vx + state.wind * 0.5;
      this.y += this.vy;
      this.alpha = Math.sin(state.time * this.alphaSpeed) * 0.5 + 0.5;

      if (this.x < -20 || this.x > state.width + 20 || this.y < 0 || this.y > state.height) {
        this.reset();
      }
    }

    draw(ctx) {
      if (this.alpha <= 0.05) return;
      ctx.save();
      ctx.globalAlpha = this.alpha;
      ctx.shadowColor = '#fde047';
      ctx.shadowBlur = 10;
      ctx.fillStyle = '#fef08a';
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  /**
   * Sparkle particle for interactive mouse trail and clicks
   */
  class Sparkle {
    constructor(x, y, color = '#fde047') {
      this.x = x;
      this.y = y;
      const angle = Math.random() * Math.PI * 2;
      const spd = Math.random() * 3 + 1;
      this.vx = Math.cos(angle) * spd;
      this.vy = Math.sin(angle) * spd;
      this.alpha = 1;
      this.size = Math.random() * 3.5 + 1.5;
      this.color = color;
    }

    update(dt) {
      this.x += this.vx;
      this.y += this.vy;
      this.alpha -= dt * 1.6;
      this.size = Math.max(0, this.size - dt * 2);
    }

    draw(ctx) {
      if (this.alpha <= 0) return;
      ctx.save();
      ctx.globalAlpha = this.alpha;
      ctx.fillStyle = this.color;
      ctx.shadowColor = this.color;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }
  }

  /**
   * Radiant Golden Shockwave when hero flower blooms
   */
  class Shockwave {
    constructor(x, y) {
      this.x = x;
      this.y = y;
      this.radius = 10;
      this.maxRadius = Math.max(state.width, state.height) * 0.85;
      this.alpha = 1;
      this.speed = Math.max(state.width, state.height) * 0.65;
    }

    update(dt) {
      this.radius += this.speed * dt;
      this.alpha = Math.max(0, 1 - (this.radius / this.maxRadius));
    }

    draw(ctx) {
      if (this.alpha <= 0) return;
      ctx.save();
      ctx.beginPath();
      ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(251, 191, 36, ${this.alpha * 0.75})`;
      ctx.lineWidth = Math.max(2, 8 * this.alpha);
      ctx.shadowColor = '#f59e0b';
      ctx.shadowBlur = 24;
      ctx.stroke();
      ctx.restore();
    }
  }

  /**
   * Floating love badges on click (+1 Đóa Hoa, 💖, etc.)
   */
  class FloatingBadge {
    constructor(x, y, text) {
      this.x = x;
      this.y = y;
      this.text = text;
      this.alpha = 1;
      this.vy = -1.8;
    }

    update(dt) {
      this.y += this.vy;
      this.alpha -= dt * 0.85;
    }

    draw(ctx) {
      if (this.alpha <= 0) return;
      ctx.save();
      ctx.globalAlpha = this.alpha;
      ctx.font = '700 15px "Montserrat", sans-serif';
      ctx.fillStyle = '#fef08a';
      ctx.shadowColor = '#f59e0b';
      ctx.shadowBlur = 12;
      ctx.textAlign = 'center';
      ctx.fillText(this.text, this.x, this.y);
      ctx.restore();
    }
  }

  // ---------------------------------------------------------
  // 8. ENVIRONMENT & BACKGROUND
  // ---------------------------------------------------------

  function initStars() {
    stars = [];
    const count = Math.floor((state.width * state.height) / 8000);
    for (let i = 0; i < count; i++) {
      stars.push({
        x: Math.random() * state.width,
        y: Math.random() * state.height * 0.7,
        size: Math.random() * 1.8 + 0.4,
        alpha: Math.random(),
        twinkleSpeed: Math.random() * 2 + 1
      });
    }
  }

  function drawBackground(ctx) {
    const theme = THEMES[state.currentTheme];

    // Sky Gradient
    const sky = ctx.createLinearGradient(0, 0, 0, state.height);
    sky.addColorStop(0, theme.skyTop);
    sky.addColorStop(0.7, theme.skyBottom);
    sky.addColorStop(1, '#020617');
    ctx.fillStyle = sky;
    ctx.fillRect(0, 0, state.width, state.height);

    // Stars
    if (theme.starAlpha > 0) {
      ctx.save();
      stars.forEach(star => {
        const twinkle = Math.sin(state.time * star.twinkleSpeed) * 0.5 + 0.5;
        ctx.globalAlpha = star.alpha * twinkle * theme.starAlpha;
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(star.x, star.y, star.size, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
    }

    // Lush Rolling Meadow / Terrain Base
    drawMeadowTerrain(ctx);
  }

  function initGrass() {
    grassBlades = [];
    const count = Math.floor(state.width / 4);
    for (let i = 0; i < count; i++) {
      grassBlades.push({
        x: (i / count) * state.width + (Math.random() - 0.5) * 6,
        height: Math.random() * 55 + 35,
        tilt: (Math.random() - 0.5) * 15,
        speed: Math.random() * 2 + 1,
        color: Math.random() > 0.4 ? '#166534' : '#15803d'
      });
    }
  }

  function drawMeadowTerrain(ctx) {
    const groundY = state.height - 40;

    // Rolling hill silhouettes
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(0, state.height);
    ctx.lineTo(0, groundY + 15);
    ctx.bezierCurveTo(
      state.width * 0.3, groundY - 25,
      state.width * 0.7, groundY + 30,
      state.width, groundY
    );
    ctx.lineTo(state.width, state.height);
    ctx.closePath();

    const hillGrad = ctx.createLinearGradient(0, groundY - 30, 0, state.height);
    hillGrad.addColorStop(0, '#064e3b');
    hillGrad.addColorStop(0.5, '#022c22');
    hillGrad.addColorStop(1, '#01150f');
    ctx.fillStyle = hillGrad;
    ctx.fill();

    // Swaying Grass Blades
    grassBlades.forEach(g => {
      const sway = Math.sin(state.time * g.speed + g.x * 0.02) * 8 + state.wind * 20;
      ctx.beginPath();
      ctx.moveTo(g.x, state.height);
      ctx.quadraticCurveTo(g.x, state.height - g.height * 0.5, g.x + g.tilt + sway, state.height - g.height);
      ctx.strokeStyle = g.color;
      ctx.lineWidth = 2.5;
      ctx.lineCap = 'round';
      ctx.stroke();
    });

    ctx.restore();
  }

  // ---------------------------------------------------------
  // 9. CINEMATIC BLOOMING SEQUENCE
  // ---------------------------------------------------------

  function startExperience() {
    if (state.started) return;
    state.started = true;

    // Fade out intro screen
    introScreen.classList.add('fade-out');

    // Start audio (YouTube background song)
    if (!state.musicPlaying) {
      playMusic();
    }

    // Spawn Hero Sunflower in center
    initHeroSunflower();
    state.stage = 'sprouting';
  }

  function initHeroSunflower() {
    const isMobile = state.width < 640;
    const isLandscape = state.height < 520;
    const targetY = isLandscape ? state.height * 0.75 : (isMobile ? state.height * 0.72 : state.height * 0.68);
    heroSunflower = new HeroSunflower(state.width / 2, targetY);
  }

  /**
   * Supernova explosion of golden pollen seeds from the blooming sunflower!
   * This triggers the colorful garden sprouting all around.
   */
  function triggerGardenBurst(originX, originY) {
    state.stage = 'bursting';
    soundEngine.playChime();

    // Radiant shockwaves of golden light
    shockwaves.push(new Shockwave(originX, originY));
    setTimeout(() => {
      shockwaves.push(new Shockwave(originX, originY));
    }, 280);

    // Spawn hundreds of flying glowing seeds
    burstingSeeds = [];
    const seedBurstCount = state.width < 640 ? 70 : 120;
    for (let i = 0; i < seedBurstCount; i++) {
      burstingSeeds.push(new BurstingSeed(originX, originY));
    }

    // Create garden flowers blooming from multiple corners and edges!
    setTimeout(() => {
      createInitialGarden();
      state.stage = 'garden';

      // Reveal Caption smoothly in center
      captionContainer.classList.remove('hidden');
      interactiveHint.classList.remove('hidden');

      // Spawn Butterflies
      const butterflyCount = state.width < 640 ? 3 : 5;
      butterflies = [];
      for (let i = 0; i < butterflyCount; i++) {
        butterflies.push(new Butterfly());
      }
    }, 600);
  }

  function createInitialGarden() {
    gardenFlowers = [];
    const isMobile = state.width < 640;
    const isLandscape = state.height < 520;
    const types = ['rose', 'tulip', 'cosmos', 'forgetmenot', 'lotus', 'sunflower'];

    // 1. GÓC DƯỚI TRÁI (Bottom-Left Corner Cluster)
    const blCount = isMobile ? 8 : 14;
    for (let i = 0; i < blCount; i++) {
      const posX = state.width * (0.02 + Math.random() * 0.22);
      const posY = state.height * (0.62 + Math.random() * 0.28);
      const scale = 0.5 + Math.random() * 0.45;
      const type = types[Math.floor(Math.random() * types.length)];
      gardenFlowers.push(new GardenFlower(posX, posY, type, {
        origin: 'bottom-left',
        scale: scale,
        delay: Math.random() * 1.5
      }));
    }

    // 2. GÓC DƯỚI PHẢI (Bottom-Right Corner Cluster)
    const brCount = isMobile ? 8 : 14;
    for (let i = 0; i < brCount; i++) {
      const posX = state.width * (0.76 + Math.random() * 0.22);
      const posY = state.height * (0.62 + Math.random() * 0.28);
      const scale = 0.5 + Math.random() * 0.45;
      const type = types[Math.floor(Math.random() * types.length)];
      gardenFlowers.push(new GardenFlower(posX, posY, type, {
        origin: 'bottom-right',
        scale: scale,
        delay: Math.random() * 1.5
      }));
    }

    // 3. VÒM HOA LEO HAI BÊN (Side Climbers Left & Right)
    const sideCount = isMobile ? 4 : 8;
    for (let i = 0; i < sideCount; i++) {
      const isLeft = i % 2 === 0;
      const posX = isLeft ? state.width * (0.03 + Math.random() * 0.16) : state.width * (0.81 + Math.random() * 0.16);
      const posY = state.height * (0.42 + Math.random() * 0.25);
      const scale = 0.45 + Math.random() * 0.4;
      const type = types[Math.floor(Math.random() * types.length)];
      gardenFlowers.push(new GardenFlower(posX, posY, type, {
        origin: isLeft ? 'side-left' : 'side-right',
        scale: scale,
        delay: 0.3 + Math.random() * 1.6
      }));
    }

    // 4. GÓC TRÊN TRÁI & PHẢI (Top-Corner Floral Garlands)
    if (!isLandscape) {
      const topCount = isMobile ? 3 : 6;
      for (let i = 0; i < topCount; i++) {
        const isLeft = i % 2 === 0;
        const posX = isLeft ? state.width * (0.02 + Math.random() * 0.13) : state.width * (0.85 + Math.random() * 0.13);
        const posY = state.height * (0.05 + Math.random() * 0.18);
        const scale = 0.35 + Math.random() * 0.35;
        const type = Math.random() > 0.5 ? 'cosmos' : 'forgetmenot';
        gardenFlowers.push(new GardenFlower(posX, posY, type, {
          origin: isLeft ? 'top-left' : 'top-right',
          scale: scale,
          delay: 0.6 + Math.random() * 1.8
        }));
      }
    }

    // 5. THẢO NGUYÊN HOA DƯỚI ĐÁY (Bottom Meadow Carpet)
    const bottomCount = isMobile ? 18 : 30;
    const padding = state.width * 0.08;
    const availableWidth = state.width - padding * 2;
    for (let i = 0; i < bottomCount; i++) {
      const posX = padding + (i / bottomCount) * availableWidth + (Math.random() - 0.5) * 30;
      const depthRatio = Math.random();
      const posY = state.height - (depthRatio * (state.height * 0.28) + 35);
      const scale = 0.45 + depthRatio * 0.55;
      const type = types[Math.floor(Math.random() * types.length)];
      gardenFlowers.push(new GardenFlower(posX, posY, type, {
        origin: 'bottom',
        scale: scale,
        delay: Math.random() * 1.8
      }));
    }

    // Sort by Y so flowers in front overlap flowers behind
    gardenFlowers.sort((a, b) => a.targetY - b.targetY);
  }

  function plantFlower(x, y) {
    // Avoid planting inside caption card zone to avoid "hình đè chữ"
    const cardBottom = state.height < 520 ? 110 : (state.width < 640 ? 260 : 280);
    const cardLeft = state.width * 0.5 - (state.width < 640 ? 190 : 330);
    const cardRight = state.width * 0.5 + (state.width < 640 ? 190 : 330);

    let spawnX = x;
    let spawnY = y;

    if (x >= cardLeft && x <= cardRight && y <= cardBottom) {
      spawnY = cardBottom + 35 + Math.random() * 45;
    }

    const scale = 0.55 + Math.random() * 0.45;
    const newFlower = new GardenFlower(spawnX, spawnY, 'random', { scale: scale, delay: 0 });
    gardenFlowers.push(newFlower);

    // Re-sort depth
    gardenFlowers.sort((a, b) => a.targetY - b.targetY);

    // Floating cheer badge
    const cheers = ['🌻 +1 Đóa hoa', '💖 Xinh đẹp', '✨ Rực rỡ', '🌸 Hạnh phúc', '🌷 Tươi thắm', '💐 An yên'];
    const randomCheer = cheers[Math.floor(Math.random() * cheers.length)];
    floatingBadges.push(new FloatingBadge(x, y - 25, randomCheer));

    // Burst sparkles at planting spot
    for (let i = 0; i < 14; i++) {
      sparkles.push(new Sparkle(x, y));
    }
  }

  function replayExperience() {
    // Hide caption & reset entities
    captionContainer.classList.add('hidden');
    interactiveHint.classList.add('hidden');

    gardenFlowers = [];
    burstingSeeds = [];
    butterflies = [];
    sparkles = [];
    shockwaves = [];
    floatingBadges = [];

    initHeroSunflower();
    state.stage = 'sprouting';
    soundEngine.playChime();

    // Rewind background track to sync with rebirth
    if (ytPlayer && ytReady && typeof ytPlayer.seekTo === 'function') {
      try {
        ytPlayer.seekTo(0);
        ytPlayer.playVideo();
        state.musicPlaying = true;
        btnMusic.classList.add('playing');
      } catch (e) {}
    }
  }

  // ---------------------------------------------------------
  // 10. ANIMATION TICK & RENDER LOOP
  // ---------------------------------------------------------
  let lastTime = performance.now();

  function loop(now) {
    const dt = Math.min((now - lastTime) / 1000, 0.1);
    lastTime = now;
    state.time += dt;

    // Wind dynamics
    state.wind += (state.windTarget - state.wind) * dt * 2;
    state.windTarget = Math.sin(state.time * 0.8) * 0.4;

    // 1. Draw Sky & Backdrop
    drawBackground(ctx);

    // 2. Draw Garden Flowers
    gardenFlowers.forEach(flower => {
      flower.update(dt);
      flower.draw(ctx);
    });

    // 3. Draw Hero Sunflower
    if (heroSunflower) {
      heroSunflower.update(dt);
      heroSunflower.draw(ctx);
    }

    // 4. Draw Shockwaves
    for (let i = shockwaves.length - 1; i >= 0; i--) {
      const sw = shockwaves[i];
      sw.update(dt);
      sw.draw(ctx);
      if (sw.alpha <= 0) shockwaves.splice(i, 1);
    }

    // 5. Draw Bursting Seeds
    for (let i = burstingSeeds.length - 1; i >= 0; i--) {
      const seed = burstingSeeds[i];
      seed.update(dt);
      seed.draw(ctx);
      if (seed.alpha <= 0) {
        burstingSeeds.splice(i, 1);
      }
    }

    // 6. Draw Butterflies
    butterflies.forEach(b => {
      b.update(dt);
      b.draw(ctx);
    });

    // 7. Draw Falling Petals
    fallingPetals.forEach(p => {
      p.update();
      p.draw(ctx);
    });

    // 8. Draw Fireflies
    fireflies.forEach(f => {
      f.update(dt);
      f.draw(ctx);
    });

    // 9. Draw Sparkles
    for (let i = sparkles.length - 1; i >= 0; i--) {
      const s = sparkles[i];
      s.update(dt);
      s.draw(ctx);
      if (s.alpha <= 0) sparkles.splice(i, 1);
    }

    // 10. Draw Floating Badges
    for (let i = floatingBadges.length - 1; i >= 0; i--) {
      const fb = floatingBadges[i];
      fb.update(dt);
      fb.draw(ctx);
      if (fb.alpha <= 0) floatingBadges.splice(i, 1);
    }

    requestAnimationFrame(loop);
  }

  // ---------------------------------------------------------
  // 11. INTERACTION & EVENT LISTENERS
  // ---------------------------------------------------------

  function resize() {
    state.width = window.innerWidth;
    state.height = window.innerHeight;
    state.dpr = Math.min(window.devicePixelRatio || 1, 2);

    canvas.width = state.width * state.dpr;
    canvas.height = state.height * state.dpr;
    canvas.style.width = `${state.width}px`;
    canvas.style.height = `${state.height}px`;

    ctx.scale(state.dpr, state.dpr);

    initStars();
    initGrass();

    if (heroSunflower) {
      const isMobile = state.width < 640;
      const isLandscape = state.height < 520;
      heroSunflower.baseX = state.width / 2;
      heroSunflower.currentX = state.width / 2;
      heroSunflower.targetY = isLandscape ? state.height * 0.75 : (isMobile ? state.height * 0.72 : state.height * 0.68);
      heroSunflower.maxHeadRadius = isMobile ? Math.min(state.width * 0.20, 85) : Math.min(state.width * 0.15, 120);
      if (heroSunflower.burstEmitted) {
        heroSunflower.currentY = heroSunflower.targetY;
      }
    }

    // Dynamically adjust existing flowers across any device orientation
    gardenFlowers.forEach(f => {
      if (f.onResize) f.onResize();
    });
  }

  function handleInteraction(x, y) {
    if (!state.started) {
      startExperience();
      return;
    }

    // Plant new flower on click if garden is active
    if (state.stage === 'garden') {
      plantFlower(x, y);
      soundEngine.playChime();
    }
  }

  // Pointer Events
  window.addEventListener('pointerdown', (e) => {
    // Ignore clicks on UI buttons/cards
    if (e.target.closest('button, input, .modal-card, .caption-card')) return;
    handleInteraction(e.clientX, e.clientY);
  });

  window.addEventListener('pointermove', (e) => {
    state.mouse.x = e.clientX;
    state.mouse.y = e.clientY;

    // Mouse movement adds breeze and fairy dust trail
    if (state.stage === 'garden' && Math.random() > 0.6) {
      sparkles.push(new Sparkle(e.clientX, e.clientY, '#fef08a'));
    }
  });

  // Start Button Click
  btnStart.addEventListener('click', () => {
    startExperience();
  });

  // Music Toggle (YouTube track + synth fallback)
  btnMusic.addEventListener('click', () => {
    toggleMusic();
    soundEngine.playChime();
  });

  // Theme Switcher (Night -> Sunset -> Dawn -> Night)
  btnTheme.addEventListener('click', () => {
    const themeKeys = ['night', 'sunset', 'dawn'];
    const nextIdx = (themeKeys.indexOf(state.currentTheme) + 1) % themeKeys.length;
    setTheme(themeKeys[nextIdx]);
    soundEngine.playChime();
  });

  function setTheme(t) {
    state.currentTheme = t;
    document.body.className = `theme-${t}`;
    themeIcon.textContent = THEMES[t].icon;
  }

  // Replay Bloom Button
  btnReplay.addEventListener('click', () => {
    replayExperience();
  });

  // Photo Lightbox Modal (Ultra-Smooth Spring Reveal & Particle Shimmer)
  if (btnViewPhoto && photoModal) {
    function openPhotoModal() {
      photoModal.classList.remove('hidden');
      requestAnimationFrame(() => {
        photoModal.classList.add('active');
      });

      soundEngine.playChime();

      // Burst celebratory golden & rose sparkles around the screen center
      const centerX = state.width / 2;
      const centerY = state.height / 2;
      for (let i = 0; i < 28; i++) {
        const col = Math.random() > 0.4 ? '#fde047' : '#f472b6';
        sparkles.push(new Sparkle(centerX + (Math.random() - 0.5) * 180, centerY + (Math.random() - 0.5) * 180, col));
      }
    }

    function closePhotoModal() {
      photoModal.classList.remove('active');
      setTimeout(() => {
        if (!photoModal.classList.contains('active')) {
          photoModal.classList.add('hidden');
        }
      }, 400);
    }

    btnViewPhoto.addEventListener('click', openPhotoModal);

    if (btnClosePhoto) {
      btnClosePhoto.addEventListener('click', closePhotoModal);
    }

    photoModal.addEventListener('click', (e) => {
      if (e.target === photoModal) {
        closePhotoModal();
      }
    });

    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && photoModal.classList.contains('active')) {
        closePhotoModal();
      }
    });
  }

  // ---------------------------------------------------------
  // 12. ÁP DỤNG TÊN NGƯỜI NHẬN (HARD-CODE)
  // ---------------------------------------------------------
  function applyRecipientName() {
    const trimmed = (typeof HARDCODED_NAME === 'string') ? HARDCODED_NAME.trim() : '';
    if (trimmed !== '') {
      recipientNameEl.textContent = trimmed;
    } else {
      recipientNameEl.textContent = '';
    }
  }

  // ---------------------------------------------------------
  // 13. INITIALIZATION
  // ---------------------------------------------------------
  function init() {
    applyRecipientName();
    initYouTubePlayer();
    resize();
    window.addEventListener('resize', resize);
    window.addEventListener('orientationchange', () => setTimeout(resize, 120));

    // Populate initial background particles
    fallingPetals = [];
    for (let i = 0; i < 24; i++) {
      fallingPetals.push(new FallingPetal());
    }

    fireflies = [];
    for (let i = 0; i < 35; i++) {
      fireflies.push(new Firefly());
    }

    // Start render loop
    requestAnimationFrame(loop);
  }

  // Math Easing Helpers
  function easeOutCubic(x) {
    return 1 - Math.pow(1 - x, 3);
  }

  function easeOutBack(x) {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    return 1 + c3 * Math.pow(x - 1, 3) + c1 * Math.pow(x - 1, 2);
  }

  // Boot on DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
