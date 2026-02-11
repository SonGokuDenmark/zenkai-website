/* ═══════════════════════════════════════════════════════════════════════════
   ZENKAI CORPORATION — Website Scripts v2.0
   © 2026 Zenkai Corporation
   ═══════════════════════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
    initNavbar();
    initMobileMenu();
    initScrollAnimations();
    initHeroCanvas();
    initBetaForm();
});

/* ─────────────────────────────────────────────────────────────────────────────
   Navbar — scroll effect
   ───────────────────────────────────────────────────────────────────────────── */
function initNavbar() {
    const navbar = document.getElementById('navbar');
    if (!navbar) return;

    let ticking = false;
    window.addEventListener('scroll', () => {
        if (!ticking) {
            window.requestAnimationFrame(() => {
                navbar.classList.toggle('scrolled', window.scrollY > 50);
                ticking = false;
            });
            ticking = true;
        }
    });
}

/* ─────────────────────────────────────────────────────────────────────────────
   Mobile Menu Toggle
   ───────────────────────────────────────────────────────────────────────────── */
function initMobileMenu() {
    const toggle = document.getElementById('nav-toggle');
    const menu = document.getElementById('nav-menu');
    if (!toggle || !menu) return;

    toggle.addEventListener('click', () => {
        toggle.classList.toggle('active');
        menu.classList.toggle('active');
    });

    // Close menu when clicking a link
    menu.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', () => {
            toggle.classList.remove('active');
            menu.classList.remove('active');
        });
    });
}

/* ─────────────────────────────────────────────────────────────────────────────
   Scroll Animations — Intersection Observer
   ───────────────────────────────────────────────────────────────────────────── */
function initScrollAnimations() {
    const elements = document.querySelectorAll('.scroll-animate');
    if (!elements.length) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                observer.unobserve(entry.target);
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    elements.forEach(el => observer.observe(el));
}

/* ─────────────────────────────────────────────────────────────────────────────
   Beta Form — AJAX submit with success state
   ───────────────────────────────────────────────────────────────────────────── */
function initBetaForm() {
    const form = document.getElementById('beta-form');
    const success = document.getElementById('beta-success');
    if (!form || !success) return;

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        const btn = form.querySelector('.btn-submit');
        const origText = btn.innerHTML;
        btn.innerHTML = '<span class="btn-icon">⏳</span> Sending...';
        btn.disabled = true;

        try {
            const res = await fetch(form.action, {
                method: 'POST',
                body: new FormData(form),
                headers: { 'Accept': 'application/json' }
            });
            if (res.ok) {
                form.style.display = 'none';
                success.style.display = 'block';
            } else {
                btn.innerHTML = '<span class="btn-icon">❌</span> Try again';
                btn.disabled = false;
                setTimeout(() => { btn.innerHTML = origText; }, 2000);
            }
        } catch {
            btn.innerHTML = '<span class="btn-icon">❌</span> Try again';
            btn.disabled = false;
            setTimeout(() => { btn.innerHTML = origText; }, 2000);
        }
    });
}

/* ─────────────────────────────────────────────────────────────────────────────
   Hero Canvas — Floating particles
   ───────────────────────────────────────────────────────────────────────────── */
function initHeroCanvas() {
    const canvas = document.getElementById('hero-canvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let particles = [];
    let animationId;
    let width, height;

    function resize() {
        width = canvas.width = canvas.offsetWidth;
        height = canvas.height = canvas.offsetHeight;
    }

    function createParticles() {
        const count = Math.min(Math.floor((width * height) / 15000), 80);
        particles = [];
        for (let i = 0; i < count; i++) {
            particles.push({
                x: Math.random() * width,
                y: Math.random() * height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 2 + 0.5,
                opacity: Math.random() * 0.5 + 0.1,
                color: ['#00ff88', '#00d2ff', '#b366ff'][Math.floor(Math.random() * 3)]
            });
        }
    }

    function drawParticles() {
        ctx.clearRect(0, 0, width, height);

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                if (dist < 150) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(0, 255, 136, ${0.08 * (1 - dist / 150)})`;
                    ctx.lineWidth = 0.5;
                    ctx.stroke();
                }
            }
        }

        // Draw particles
        particles.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
            ctx.fillStyle = p.color;
            ctx.globalAlpha = p.opacity;
            ctx.fill();
            ctx.globalAlpha = 1;

            // Move
            p.x += p.vx;
            p.y += p.vy;

            // Wrap around
            if (p.x < 0) p.x = width;
            if (p.x > width) p.x = 0;
            if (p.y < 0) p.y = height;
            if (p.y > height) p.y = 0;
        });

        animationId = requestAnimationFrame(drawParticles);
    }

    // Handle resize
    window.addEventListener('resize', () => {
        resize();
        createParticles();
    });

    // Reduce animation when not visible
    const heroObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                if (!animationId) drawParticles();
            } else {
                cancelAnimationFrame(animationId);
                animationId = null;
            }
        });
    });

    heroObserver.observe(canvas.parentElement || canvas);

    resize();
    createParticles();
    drawParticles();
}
