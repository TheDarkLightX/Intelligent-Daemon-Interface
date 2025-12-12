/** @type {import('tailwindcss').Config} */
export default {
  content: ['./src/**/*.{html,js,svelte,ts}', './index.html'],
  theme: {
    extend: {
      colors: {
        // Synth aesthetic colors
        synth: {
          bg: '#1a1a2e',
          panel: '#252540',
          surface: '#2d2d4a',
          border: '#3d3d5c',
          accent: '#f9a826',
          'accent-dim': '#c78520',
          success: '#4ade80',
          warning: '#fbbf24',
          danger: '#ef4444',
          text: '#e4e4e7',
          'text-dim': '#a1a1aa',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        glow: '0 0 20px rgba(249, 168, 38, 0.3)',
        'glow-sm': '0 0 10px rgba(249, 168, 38, 0.2)',
        panel: '0 4px 20px rgba(0, 0, 0, 0.4)',
      },
      animation: {
        pulse: 'pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        glow: 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(249, 168, 38, 0.2)' },
          '100%': { boxShadow: '0 0 20px rgba(249, 168, 38, 0.4)' },
        },
      },
    },
  },
  plugins: [],
};
