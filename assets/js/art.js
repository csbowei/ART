/* ═══════════════════════════════════════════════════════
   art.js  —  ART Project Page
═══════════════════════════════════════════════════════ */

/* ── Dark / Light theme toggle ──────────────────────── */
(function () {
  var KEY = 'art-theme-mode';
  var current = localStorage.getItem(KEY) || 'dark';

  function applyTheme(mode) {
    if (mode === 'light') {
      document.documentElement.setAttribute('data-theme', 'light');
    } else {
      document.documentElement.removeAttribute('data-theme');
    }
    current = mode;
    localStorage.setItem(KEY, mode);

    var sunBtn  = document.getElementById('themeSun');
    var moonBtn = document.getElementById('themeMoon');
    if (sunBtn && moonBtn) {
      sunBtn.classList.toggle('active', mode === 'light');
      moonBtn.classList.toggle('active', mode === 'dark');
    }
  }

  // Apply on load before render to avoid flash
  applyTheme(current);

  document.addEventListener('DOMContentLoaded', function () {
    var sunBtn  = document.getElementById('themeSun');
    var moonBtn = document.getElementById('themeMoon');
    if (sunBtn)  sunBtn.addEventListener('click',  function () { applyTheme('light'); });
    if (moonBtn) moonBtn.addEventListener('click', function () { applyTheme('dark'); });
    // Re-sync button active state after DOM ready
    applyTheme(current);
  });
})();


/* ── Nav scroll + section highlight ────────────────── */
(function () {
  document.addEventListener('DOMContentLoaded', function () {
    var nav   = document.getElementById('topNav');
    var links = document.querySelectorAll('.nav-link[data-section]');

    var sectionIds = [];
    links.forEach(function (l) { sectionIds.push(l.getAttribute('data-section')); });

    function getSection(id) { return document.getElementById(id); }

    window.addEventListener('scroll', function () {
      // Scrolled class for slightly more opaque nav
      if (window.scrollY > 40) nav.classList.add('scrolled');
      else nav.classList.remove('scrolled');

      // Active link tracking
      var current = '';
      sectionIds.forEach(function (id) {
        var sec = getSection(id);
        if (sec && window.scrollY >= sec.offsetTop - 130) current = id;
      });

      links.forEach(function (l) {
        l.classList.toggle('active', l.getAttribute('data-section') === current);
      });
    }, { passive: true });
  });
})();


/* ── Scroll-reveal ──────────────────────────────────── */
(function () {
  var obs = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.07 });
  document.querySelectorAll('.reveal').forEach(function (el) { obs.observe(el); });
})();


/* ── Lazy-load non-hero images ──────────────────────── */
document.addEventListener('DOMContentLoaded', function () {
  document.querySelectorAll('img').forEach(function (img) {
    if (!img.closest('.hero') && !img.closest('#lightbox') && !img.hasAttribute('loading')) {
      img.setAttribute('loading', 'lazy');
    }
  });
});


/* ── Lightbox ───────────────────────────────────────── */
function openLightbox(src) {
  var img = document.getElementById('lightboxImg');
  img.src = '';
  img.src = src;
  document.getElementById('lightbox').classList.add('open');
  document.body.style.overflow = 'hidden';
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('open');
  document.body.style.overflow = '';
}
document.addEventListener('keydown', function (e) {
  if (e.key === 'Escape') closeLightbox();
});


/* ── Toast ──────────────────────────────────────────── */
function showToast(msg, duration) {
  var container = document.getElementById('toastContainer');
  var toast = document.createElement('div');
  toast.className = 'toast';
  toast.textContent = msg;
  container.appendChild(toast);
  requestAnimationFrame(function () { toast.classList.add('show'); });
  setTimeout(function () {
    toast.classList.remove('show');
    toast.classList.add('hide');
    setTimeout(function () { container.removeChild(toast); }, 300);
  }, duration || 2000);
}


/* ── BibTeX copy ────────────────────────────────────── */
function copyBibtex() {
  var text = document.getElementById('bibtexContent').textContent;
  var btn  = document.getElementById('bibtexCopyBtn');
  var ok   = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg> Copied!';
  var orig = btn.innerHTML;
  if (navigator.clipboard) {
    navigator.clipboard.writeText(text).then(function () {
      btn.innerHTML = ok;
      setTimeout(function () { btn.innerHTML = orig; }, 2000);
    });
  } else {
    var ta = document.createElement('textarea');
    ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
    document.body.appendChild(ta); ta.select(); document.execCommand('copy');
    document.body.removeChild(ta);
    btn.innerHTML = ok;
    setTimeout(function () { btn.innerHTML = orig; }, 2000);
  }
}


/* ── 3D Carousel ─────────────────────────────────────── */
(function () {
  var stage   = document.getElementById('c3dStage');
  var btnPrev = document.getElementById('c3dPrev');
  var btnNext = document.getElementById('c3dNext');
  var dotsWrap= document.getElementById('c3dDots');
  if (!stage) return;

  var cards = Array.prototype.slice.call(stage.querySelectorAll('.c3d-card'));
  var total = cards.length;
  var current = 0;
  var dots = [];

  /* Card geometry */
  var SIDE_X       = 360;  /* how far left/right the side cards translate */
  var SIDE_ROT     = 42;   /* Y-rotation degrees for side cards */
  var SIDE_Z       = -120; /* Z translate for side cards (depth) */
  var SIDE_SCALE   = 0.82; /* scale of side cards */
  var FAR_SCALE    = 0.60; /* scale of far cards */
  var FAR_X        = 560;
  var FAR_Z        = -280;
  var ACTIVE_SCALE = 1;    /* keep active at native scale to avoid blur */

  /* Build dots */
  for (var i = 0; i < total; i++) {
    var d = document.createElement('button');
    d.className = 'c3d-dot' + (i === 0 ? ' active' : '');
    d.setAttribute('aria-label', 'Card ' + (i + 1));
    (function (idx) { d.addEventListener('click', function () { goTo(idx); }); })(i);
    dotsWrap.appendChild(d);
    dots.push(d);
  }

  function resetStateClasses(card) {
    card.classList.remove('is-active', 'is-side', 'is-far', 'is-hidden');
  }

  function normalizeOffset(offset) {
    var half = Math.floor(total / 2);
    if (offset > half) offset -= total;
    if (offset < -half) offset += total;
    return offset;
  }

  function applyPositions() {
    cards.forEach(function (card, i) {
      var offset = normalizeOffset(i - current);
      var tx, ry, tz, sc, op, blur, pe;

      resetStateClasses(card);

      if (offset === 0) {
        tx = 0; ry = 0; tz = 0; sc = ACTIVE_SCALE; op = 1; blur = 0; pe = 'auto';
        card.classList.add('is-active');
      } else if (offset === -1) {
        tx = -SIDE_X; ry = SIDE_ROT; tz = SIDE_Z; sc = SIDE_SCALE; op = 0.74; blur = 1.2; pe = 'auto';
        card.classList.add('is-side');
      } else if (offset === 1) {
        tx = SIDE_X; ry = -SIDE_ROT; tz = SIDE_Z; sc = SIDE_SCALE; op = 0.74; blur = 1.2; pe = 'auto';
        card.classList.add('is-side');
      } else if (offset === -2) {
        tx = -FAR_X; ry = SIDE_ROT * 1.3; tz = FAR_Z; sc = FAR_SCALE; op = 0.30; blur = 2.4; pe = 'none';
        card.classList.add('is-far');
      } else if (offset === 2) {
        tx = FAR_X; ry = -SIDE_ROT * 1.3; tz = FAR_Z; sc = FAR_SCALE; op = 0.30; blur = 2.4; pe = 'none';
        card.classList.add('is-far');
      } else {
        tx = offset > 0 ? FAR_X * 1.5 : -FAR_X * 1.5;
        ry = offset > 0 ? -60 : 60;
        tz = -500; sc = 0.4; op = 0; blur = 5; pe = 'none';
        card.classList.add('is-hidden');
      }

      card.style.transform =
        'translateX(' + tx + 'px) translateZ(' + tz + 'px) rotateY(' + ry + 'deg) scale(' + sc + ')';
      card.style.opacity = op;
      card.style.filter  = blur > 0 ? 'blur(' + blur + 'px)' : 'none';
      card.style.pointerEvents = pe;
      card.style.zIndex  = offset === 0 ? 10 : (Math.abs(offset) === 1 ? 5 : 2);
    });

    dots.forEach(function (d, i) { d.classList.toggle('active', i === current); });
  }

  function goTo(idx) {
    current = ((idx % total) + total) % total;
    applyPositions();
  }

  /* Click side cards to navigate */
  cards.forEach(function (card, i) {
    card.addEventListener('click', function (e) {
      if (i === current) return; /* let content handle events when active */
      e.preventDefault();
      goTo(i);
    });
  });

  btnPrev.addEventListener('click', function () { goTo(current - 1); });
  btnNext.addEventListener('click', function () { goTo(current + 1); });

  /* Keyboard */
  document.addEventListener('keydown', function (e) {
    if (e.key === 'ArrowLeft')  goTo(current - 1);
    if (e.key === 'ArrowRight') goTo(current + 1);
  });

  /* Touch swipe */
  var touchStartX = 0;
  stage.addEventListener('touchstart', function (e) { touchStartX = e.touches[0].clientX; }, { passive: true });
  stage.addEventListener('touchend',   function (e) {
    var dx = e.changedTouches[0].clientX - touchStartX;
    if (Math.abs(dx) > 50) goTo(dx < 0 ? current + 1 : current - 1);
  }, { passive: true });

  applyPositions();
})();


/* ── SOTA Comparison ────────────────────────────────── */
var sotaPairIdx = 0;

(function () {
  var active = document.querySelector('.sota-method-item.active');
  if (active) {
    var val = active.dataset['p' + sotaPairIdx];
    if (val) {
      document.getElementById('sotaSliderRight').src = val;
      document.getElementById('sotaRightLabel').textContent = active.dataset.label;
    }
  }
})();

function sotaSelectPair(el) {
  document.querySelectorAll('.sota-pair-item').forEach(function (p) { p.classList.remove('active'); });
  el.classList.add('active');
  sotaPairIdx = parseInt(el.dataset.index, 10);
  document.getElementById('sotaSliderLeft').src = el.dataset.source;

  document.querySelectorAll('.sota-method-item').forEach(function (m) {
    var val = m.dataset['p' + sotaPairIdx];
    if (val) m.querySelector('img').src = val;
  });

  var active = document.querySelector('.sota-method-item.active');
  if (active) {
    var val = active.dataset['p' + sotaPairIdx];
    if (val) document.getElementById('sotaSliderRight').src = val;
  }
}

function sotaSelectMethod(el) {
  document.querySelectorAll('.sota-method-item').forEach(function (m) { m.classList.remove('active'); });
  el.classList.add('active');
  document.getElementById('sotaRightLabel').textContent = el.dataset.label;
  var val = el.dataset['p' + sotaPairIdx];
  if (val) document.getElementById('sotaSliderRight').src = val;
}


/* ── More Results — Cartesian selector ──────────────── */
var resultsSrcIdx = 1;
var resultsRefIdx = 1;

function resultsGetPaths(s, r) {
  var base = 'assets/images/more_visual/';
  return {
    src:  base + 'source_' + s + '.jpg',
    ours: base + 'generate_' + s + '_' + r + '.jpg'
  };
}

function resultsUpdate() {
  var p        = resultsGetPaths(resultsSrcIdx, resultsRefIdx);
  var noResult = document.getElementById('resultsNoResult');
  var leftImg  = document.getElementById('resultsLeft');
  var rightImg = document.getElementById('resultsRight');

  leftImg.src = p.src;
  rightImg.src = p.ours;
  noResult.classList.remove('show');

  rightImg.onerror = function () {
    noResult.classList.add('show');
    rightImg.onerror = null;
  };
}

function resultsSelectSrc(el) {
  document.querySelectorAll('#resultsSrcCol .results-col-item').forEach(function (i) { i.classList.remove('active'); });
  el.classList.add('active');
  resultsSrcIdx = parseInt(el.dataset.srcIdx, 10);
  resultsUpdate();
}

function resultsSelectRef(el) {
  document.querySelectorAll('#resultsRefCol .results-col-item').forEach(function (i) { i.classList.remove('active'); });
  el.classList.add('active');
  resultsRefIdx = parseInt(el.dataset.refIdx, 10);
  resultsUpdate();
}

resultsUpdate();


/* ── Magnifier ──────────────────────────────────────── */
(function () {
  var ZOOM = 0.2, LENS = 82;
  document.querySelectorAll('.magnifier-wrap').forEach(function (wrap) {
    var img    = wrap.querySelector('img');
    var lens   = wrap.querySelector('.magnifier-lens');
    var canvas = lens.querySelector('canvas');
    var ctx    = canvas.getContext('2d');
    canvas.width = canvas.height = LENS;

    function draw(ex, ey) {
      if (!img.complete || !img.naturalWidth) return;
      var W = wrap.offsetWidth, H = wrap.offsetHeight;
      var iW = img.naturalWidth, iH = img.naturalHeight;
      var srcW = LENS / ZOOM, srcH = LENS / ZOOM;
      var sx = Math.max(0, Math.min((ex / W) * iW - srcW / 2, iW - srcW));
      var sy = Math.max(0, Math.min((ey / H) * iH - srcH / 2, iH - srcH));
      ctx.clearRect(0, 0, LENS, LENS);
      ctx.save();
      ctx.beginPath();
      ctx.arc(LENS / 2, LENS / 2, LENS / 2, 0, Math.PI * 2);
      ctx.clip();
      ctx.drawImage(img, sx, sy, srcW, srcH, 0, 0, LENS, LENS);
      ctx.restore();
    }

    wrap.addEventListener('mousemove', function (e) {
      var r = wrap.getBoundingClientRect();
      var ex = e.clientX - r.left, ey = e.clientY - r.top;
      lens.style.left = ex + 'px';
      lens.style.top  = ey + 'px';
      if (img.complete && img.naturalWidth) { draw(ex, ey); }
      else { img.addEventListener('load', function onL() { draw(ex, ey); img.removeEventListener('load', onL); }); }
    });
  });
})();