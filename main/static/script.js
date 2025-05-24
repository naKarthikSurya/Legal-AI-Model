// script.js - Shared JavaScript functionality for all pages
document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle functionality
    const mobileMenuToggle = document.createElement('div');
    mobileMenuToggle.className = 'mobile-menu-toggle';
    mobileMenuToggle.innerHTML = '<i class="fas fa-bars"></i>';
    
    const header = document.querySelector('header');
    const nav = document.querySelector('nav');
    
    if (header && nav) {
        header.querySelector('.container').insertBefore(mobileMenuToggle, nav);
        
        mobileMenuToggle.addEventListener('click', function() {
            nav.classList.toggle('active');
            this.classList.toggle('active');
            
            // Change icon based on state
            if (this.classList.contains('active')) {
                this.innerHTML = '<i class="fas fa-times"></i>';
            } else {
                this.innerHTML = '<i class="fas fa-bars"></i>';
            }
        });
    }
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add animation for elements when they come into view
    const animateOnScroll = function() {
        const elements = document.querySelectorAll('.animate-on-scroll');
        
        elements.forEach(element => {
            const elementPosition = element.getBoundingClientRect().top;
            const windowHeight = window.innerHeight;
            
            if (elementPosition < windowHeight - 50) {
                element.classList.add('animated');
            }
        });
    };
    
    // Add animate-on-scroll class to elements that should animate
    document.querySelectorAll('.feature-card, .step, .tech-card').forEach(
        element => element.classList.add('animate-on-scroll')
    );
    
    window.addEventListener('scroll', animateOnScroll);
    animateOnScroll(); // Run once on page load
    
    // Show/hide back-to-top button
    const backToTopBtn = document.createElement('button');
    backToTopBtn.id = 'back-to-top';
    backToTopBtn.innerHTML = '<i class="fas fa-arrow-up"></i>';
    document.body.appendChild(backToTopBtn);
    
    window.addEventListener('scroll', function() {
        if (window.pageYOffset > 300) {
            backToTopBtn.classList.add('visible');
        } else {
            backToTopBtn.classList.remove('visible');
        }
    });
    
    backToTopBtn.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
    
    // Add accessibility improvements
    const improveAccessibility = function() {
        // Add aria labels to buttons without text
        document.querySelectorAll('button').forEach(button => {
            if (!button.textContent.trim() && !button.getAttribute('aria-label')) {
                const icon = button.querySelector('i');
                if (icon && icon.className) {
                    let label = '';
                    if (icon.className.includes('paper-plane')) label = 'Send message';
                    else if (icon.className.includes('copy')) label = 'Copy to clipboard';
                    else if (icon.className.includes('trash')) label = 'Clear history';
                    else if (icon.className.includes('arrow-up')) label = 'Back to top';
                    
                    if (label) button.setAttribute('aria-label', label);
                }
            }
        });
        
        // Add appropriate roles
        const mainContent = document.querySelector('main, .chat-interface, .model-interface');
        if (mainContent && !mainContent.getAttribute('role')) {
            mainContent.setAttribute('role', 'main');
        }
        
        const nav = document.querySelector('nav');
        if (nav && !nav.getAttribute('role')) {
            nav.setAttribute('role', 'navigation');
        }
    };
    
    improveAccessibility();
    
    // Handle theme preferences (light/dark)
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
    
    const toggleTheme = function(isDark) {
        if (isDark) {
            document.body.classList.add('dark-theme');
        } else {
            document.body.classList.remove('dark-theme');
        }
        
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    };
    
    // Check for saved theme preference or use system preference
    const savedTheme = localStorage.getItem('theme');
    if (savedTheme === 'dark') {
        toggleTheme(true);
    } else if (savedTheme === 'light') {
        toggleTheme(false);
    } else {
        toggleTheme(prefersDarkScheme.matches);
    }
    
    // Create theme toggle button
    const themeToggleBtn = document.createElement('button');
    themeToggleBtn.id = 'theme-toggle';
    themeToggleBtn.setAttribute('aria-label', 'Toggle dark/light theme');
    
    // Update button icon based on current theme
    if (document.body.classList.contains('dark-theme')) {
        themeToggleBtn.innerHTML = '<i class="fas fa-sun"></i>';
    } else {
        themeToggleBtn.innerHTML = '<i class="fas fa-moon"></i>';
    }
    
    document.body.appendChild(themeToggleBtn);
    
    themeToggleBtn.addEventListener('click', function() {
        const isDarkNow = document.body.classList.contains('dark-theme');
        toggleTheme(!isDarkNow);
        
        // Update button icon
        if (!isDarkNow) {
            this.innerHTML = '<i class="fas fa-sun"></i>';
        } else {
            this.innerHTML = '<i class="fas fa-moon"></i>';
        }
    });
});