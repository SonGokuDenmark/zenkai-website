/**
 * ═══════════════════════════════════════════════════════════════════════════
 * ZENKAI CORPORATION — Website JavaScript v1.0
 * © 2026 Zenkai Corporation
 * ═══════════════════════════════════════════════════════════════════════════
 */

// ─────────────────────────────────────────────────────────────────────────────
// Utility Functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Throttle function to limit execution rate
 */
function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

/**
 * Debounce function to delay execution
 */
function debounce(func, wait) {
    let timeout;
    return function(...args) {
        clearTimeout(timeout);
        timeout = setTimeout(() => func.apply(this, args), wait);
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// Navigation
// ─────────────────────────────────────────────────────────────────────────────

const navbar = document.getElementById('navbar');
const navToggle = document.getElementById('nav-toggle');
const navMenu = document.getElementById('nav-menu');
const navLinks = document.querySelectorAll('.nav-link');

// Scroll effect for navbar
const handleNavScroll = throttle(() => {
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
}, 100);

window.addEventListener('scroll', handleNavScroll);

// Mobile menu toggle
navToggle.addEventListener('click', () => {
    navToggle.classList.toggle('active');
    navMenu.classList.toggle('active');
    document.body.style.overflow = navMenu.classList.contains('active') ? 'hidden' : '';
});

// Close mobile menu when clicking a link
navLinks.forEach(link => {
    link.addEventListener('click', () => {
        navToggle.classList.remove('active');
        navMenu.classList.remove('active');
        document.body.style.overflow = '';
    });
});

// Close mobile menu on escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && navMenu.classList.contains('active')) {
        navToggle.classList.remove('active');
        navMenu.classList.remove('active');
        document.body.style.overflow = '';
    }
});

// ─────────────────────────────────────────────────────────────────────────────
// Scroll Animations
// ─────────────────────────────────────────────────────────────────────────────

// Add scroll-animate class to elements
const animateElements = [
    '.project-card',
    '.timeline-item',
    '.blog-card',
    '.section-header'
];

animateElements.forEach(selector => {
    document.querySelectorAll(selector).forEach(el => {
        el.classList.add('scroll-animate');
    });
});

// Intersection Observer for scroll animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const animationObserver = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            // Optionally stop observing after animation
            // animationObserver.unobserve(entry.target);
        }
    });
}, observerOptions);

document.querySelectorAll('.scroll-animate').forEach(el => {
    animationObserver.observe(el);
});

// ─────────────────────────────────────────────────────────────────────────────
// Hero Canvas Particle Animation
// ─────────────────────────────────────────────────────────────────────────────

class ParticleNetwork {
    constructor(canvas) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d');
        this.particles = [];
        this.particleCount = 80;
        this.maxDistance = 150;
        this.mouse = { x: null, y: null, radius: 150 };

        this.init();
        this.setupEvents();
        this.animate();
    }

    init() {
        this.resize();
        this.createParticles();
    }

    resize() {
        this.canvas.width = window.innerWidth;
        this.canvas.height = window.innerHeight;
    }

    createParticles() {
        this.particles = [];
        // Adjust particle count based on screen size
        const count = window.innerWidth < 768 ? 40 : this.particleCount;

        for (let i = 0; i < count; i++) {
            this.particles.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                radius: Math.random() * 2 + 1,
                color: this.getRandomColor()
            });
        }
    }

    getRandomColor() {
        const colors = [
            'rgba(0, 255, 136, 0.8)',   // Primary green
            'rgba(0, 210, 255, 0.8)',   // Cyan
            'rgba(179, 102, 255, 0.6)', // Purple
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    setupEvents() {
        // Debounced resize handler
        window.addEventListener('resize', debounce(() => {
            this.resize();
            this.createParticles();
        }, 200));

        // Mouse movement
        this.canvas.addEventListener('mousemove', (e) => {
            this.mouse.x = e.clientX;
            this.mouse.y = e.clientY;
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.mouse.x = null;
            this.mouse.y = null;
        });
    }

    drawParticles() {
        this.particles.forEach(particle => {
            this.ctx.beginPath();
            this.ctx.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
            this.ctx.fillStyle = particle.color;
            this.ctx.fill();
        });
    }

    drawConnections() {
        for (let i = 0; i < this.particles.length; i++) {
            for (let j = i + 1; j < this.particles.length; j++) {
                const dx = this.particles[i].x - this.particles[j].x;
                const dy = this.particles[i].y - this.particles[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < this.maxDistance) {
                    const opacity = 1 - (distance / this.maxDistance);
                    this.ctx.beginPath();
                    this.ctx.strokeStyle = `rgba(0, 255, 136, ${opacity * 0.2})`;
                    this.ctx.lineWidth = 1;
                    this.ctx.moveTo(this.particles[i].x, this.particles[i].y);
                    this.ctx.lineTo(this.particles[j].x, this.particles[j].y);
                    this.ctx.stroke();
                }
            }
        }
    }

    updateParticles() {
        this.particles.forEach(particle => {
            // Mouse interaction
            if (this.mouse.x !== null && this.mouse.y !== null) {
                const dx = this.mouse.x - particle.x;
                const dy = this.mouse.y - particle.y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < this.mouse.radius) {
                    const force = (this.mouse.radius - distance) / this.mouse.radius;
                    particle.vx -= (dx / distance) * force * 0.02;
                    particle.vy -= (dy / distance) * force * 0.02;
                }
            }

            // Update position
            particle.x += particle.vx;
            particle.y += particle.vy;

            // Boundary check with bounce
            if (particle.x < 0 || particle.x > this.canvas.width) {
                particle.vx *= -1;
                particle.x = Math.max(0, Math.min(this.canvas.width, particle.x));
            }
            if (particle.y < 0 || particle.y > this.canvas.height) {
                particle.vy *= -1;
                particle.y = Math.max(0, Math.min(this.canvas.height, particle.y));
            }

            // Damping
            particle.vx *= 0.99;
            particle.vy *= 0.99;

            // Minimum velocity
            const minVelocity = 0.1;
            if (Math.abs(particle.vx) < minVelocity) {
                particle.vx = (Math.random() - 0.5) * 0.5;
            }
            if (Math.abs(particle.vy) < minVelocity) {
                particle.vy = (Math.random() - 0.5) * 0.5;
            }
        });
    }

    animate() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.drawConnections();
        this.drawParticles();
        this.updateParticles();
        requestAnimationFrame(() => this.animate());
    }
}

// Initialize particle network
const heroCanvas = document.getElementById('hero-canvas');
if (heroCanvas) {
    new ParticleNetwork(heroCanvas);
}

// ─────────────────────────────────────────────────────────────────────────────
// Smooth Scroll Enhancement
// ─────────────────────────────────────────────────────────────────────────────

// Handle anchor links with smooth scroll
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        const href = this.getAttribute('href');
        if (href === '#') return;

        const target = document.querySelector(href);
        if (target) {
            e.preventDefault();
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// ─────────────────────────────────────────────────────────────────────────────
// Active Nav Link Highlighting
// ─────────────────────────────────────────────────────────────────────────────

const sections = document.querySelectorAll('section[id]');

const highlightNavLink = throttle(() => {
    const scrollY = window.scrollY;

    sections.forEach(section => {
        const sectionHeight = section.offsetHeight;
        const sectionTop = section.offsetTop - 100;
        const sectionId = section.getAttribute('id');

        if (scrollY > sectionTop && scrollY <= sectionTop + sectionHeight) {
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === `#${sectionId}`) {
                    link.classList.add('active');
                }
            });
        }
    });
}, 100);

window.addEventListener('scroll', highlightNavLink);

// ─────────────────────────────────────────────────────────────────────────────
// Staggered Animation for Cards
// ─────────────────────────────────────────────────────────────────────────────

const staggerElements = (selector, containerSelector) => {
    const container = document.querySelector(containerSelector);
    if (!container) return;

    const elements = container.querySelectorAll(selector);

    const observer = new IntersectionObserver((entries) => {
        if (entries[0].isIntersecting) {
            elements.forEach((el, index) => {
                setTimeout(() => {
                    el.classList.add('visible');
                }, index * 100);
            });
            observer.unobserve(container);
        }
    }, { threshold: 0.1 });

    observer.observe(container);
};

// Apply staggered animations
staggerElements('.project-card', '.projects-grid');
staggerElements('.timeline-item', '.timeline');
staggerElements('.blog-card', '.blog-grid');

// ─────────────────────────────────────────────────────────────────────────────
// Console Easter Egg
// ─────────────────────────────────────────────────────────────────────────────

console.log(`
%c⚡ ZENKAI CORPORATION ⚡
%c━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Building the future of trading,
gaming & automation — powered by AI.

Evolve or Die.

%c© 2026 Zenkai Corporation
Founded by Son Goku 🇩🇰
`,
'color: #00ff88; font-size: 20px; font-weight: bold; text-shadow: 0 0 10px #00ff88;',
'color: #00d2ff; font-size: 12px;',
'color: #8892a4; font-size: 10px;'
);

// ─────────────────────────────────────────────────────────────────────────────
// Page Load Animation
// ─────────────────────────────────────────────────────────────────────────────

window.addEventListener('load', () => {
    document.body.classList.add('loaded');

    // Trigger initial scroll check
    handleNavScroll();
    highlightNavLink();
});

// ─────────────────────────────────────────────────────────────────────────────
// Reduce Motion Support
// ─────────────────────────────────────────────────────────────────────────────

const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

if (prefersReducedMotion.matches) {
    // Disable animations for users who prefer reduced motion
    document.documentElement.style.setProperty('--transition-fast', '0ms');
    document.documentElement.style.setProperty('--transition-base', '0ms');
    document.documentElement.style.setProperty('--transition-slow', '0ms');

    // Mark all animated elements as visible
    document.querySelectorAll('.scroll-animate').forEach(el => {
        el.classList.add('visible');
    });

    document.querySelectorAll('.animate-fade-up').forEach(el => {
        el.style.animation = 'none';
        el.style.opacity = '1';
        el.style.transform = 'none';
    });
}
