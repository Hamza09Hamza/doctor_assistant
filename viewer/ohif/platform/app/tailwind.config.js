/** @type {import('tailwindcss').Config} */
module.exports = {
  // Note: in Tailwind 3.0, JIT will purge unused styles by default
  // but in development, it is often useful to disable this to see
  // and try out all the styles that are available.
  // ...(process.env.NODE_ENV === 'development' && {
  //   safelist: [{ pattern: /.*/ }],
  // }),
  presets: [require('../ui/tailwind.config.js'), require('../ui-next/tailwind.config.js')],
  content: [
    './src/**/*.{jsx,js,ts,tsx,css}',
    '../../extensions/*/src/**/*.{jsx,js,ts,tsx,css}',
    '../ui/src/**/*.{jsx,js,ts,tsx,css}',
    '../ui-next/src/**/*.{jsx,js,ts,tsx,css}',
    '../../modes/*/src/**/*.{jsx,js,ts,tsx,css}',
    './node_modules/@ohif/ui/src/**/*.{js,jsx,ts,tsx,css}',
    '../../node_modules/@ohif/ui/src/**/*.{js,jsx,ts,tsx,css}',
    '../../node_modules/@ohif/ui-next/src/**/*.{js,jsx,ts,tsx,css}',
    '../../node_modules/@ohif/extension-*/src/**/*.{js,jsx,ts,tsx,css}',
  ],
  theme: {
    fontFamily: {
      sans: [
        '"IBM Plex Sans"',
        'Inter',
        'system-ui',
        '-apple-system',
        'BlinkMacSystemFont',
        '"Segoe UI"',
        'Roboto',
        '"Helvetica Neue"',
        'Arial',
        '"Noto Sans"',
        'sans-serif',
        '"Apple Color Emoji"',
        '"Segoe UI Emoji"',
        '"Segoe UI Symbol"',
        '"Noto Color Emoji"',
      ],
      // A diagnostic workspace needs one neutral interface family, not a
      // decorative display face. Keep `serif` mapped to the clinical sans so
      // inherited stock classes cannot reintroduce boutique typography.
      serif: ['"IBM Plex Sans"', 'Inter', 'system-ui', 'sans-serif'],
      mono: ['"IBM Plex Mono"', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
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
    },
  },
};
