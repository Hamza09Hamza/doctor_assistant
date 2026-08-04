/** @type {import('tailwindcss').Config} */
module.exports = {
  presets: [require('../ui/tailwind.config.js')],
  content: [
    './pages/**/*.{ts,tsx}',
    './components/**/*.{ts,tsx}',
    './app/**/*.{ts,tsx}',
    './src/**/*.{ts,tsx}',
  ],
  prefix: '',
  theme: {
    // `theme.fontFamily` (not `extend.fontFamily`) fully replaces Tailwind's default
    // sans/serif/mono map — before this, no `font-serif` utility existed anywhere in
    // this app. `serif` here is Clinique Amina's brand type for headings/wordmark
    // (Playfair Display, loaded in platform/app/public/html-templates/index.html);
    // `sans`/`inter` both stay Inter so `font-sans`/`font-inter` are interchangeable.
    fontFamily: {
      inter: ['Inter', 'sans-serif'],
      sans: ['Inter', 'sans-serif'],
      serif: ['"Playfair Display"', 'serif'],
    },
    fontSize: {
      xxs: '0.625rem', // 10px
      xs: '0.6875rem', // 11px
      sm: '0.75rem', // 12px
      base: '0.8125rem', // 13px
      lg: '0.875rem', // 14px
      xl: '1rem', // 16px
      // 2xl and above will be updated in an upcoming version
      '2xl': '1.5rem',
      '3xl': '1.875rem',
      '4xl': '2.25rem',
      '5xl': '3rem',
      '6xl': '4rem',
      // '2xl': '1.125rem', // 18px
      // '3xl': '1.375rem', // 22px
      // '4xl': '1.5rem', // 24px
      // '5xl': '1.875rem', // 30px
    },
    fontWeight: {
      hairline: '100',
      thin: '200',
      light: '300',
      normal: '400',
      medium: '500',
      semibold: '600',
      bold: '700',
      extrabold: '800',
      black: '900',
    },
    extend: {
      colors: {
        highlight: 'hsl(var(--highlight))',
        neutral: 'hsl(var(--neutral))',
        'neutral-light': 'hsl(var(--neutral-light))',
        'neutral-dark': 'hsl(var(--neutral-dark))',
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        // Semantic badge/status tokens (already-defined --success-*/--info-*/etc. vars
        // in tailwind.css are full `hsl(...)` strings, unlike the vars above, so these
        // are wired with bare `var(--x)`, not `hsl(var(--x))`). Note: `extend.info`
        // below (a sibling of `extend.colors`, not nested in it) is a different,
        // unrelated key that happens to share the name — no actual collision.
        success: {
          bg: 'var(--success-bg)',
          border: 'var(--success-border)',
          text: 'var(--success-text)',
        },
        warning: {
          bg: 'var(--warning-bg)',
          border: 'var(--warning-border)',
          text: 'var(--warning-text)',
        },
        error: {
          bg: 'var(--error-bg)',
          border: 'var(--error-border)',
          text: 'var(--error-text)',
        },
        info: {
          bg: 'var(--info-bg)',
          border: 'var(--info-border)',
          text: 'var(--info-text)',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      keyframes: {
        'accordion-down': {
          from: { height: '0' },
          to: { height: 'var(--radix-accordion-content-height)' },
        },
        'accordion-up': {
          from: { height: 'var(--radix-accordion-content-height)' },
          to: { height: '0' },
        },
      },
      animation: {
        'accordion-down': 'accordion-down 0.2s ease-out',
        'accordion-up': 'accordion-up 0.2s ease-out',
      },
      bkg: {
        low: '#050615',
        med: '#090C29',
        full: '#041C4A',
      },
      info: {
        primary: '#FFFFFF',
        secondary: '#7BB2CE',
      },
      actions: {
        primary: '#348CFD',
        highlight: '#5ACCE6',
        hover: 'rgba(52, 140, 253, 0.2)',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
};
