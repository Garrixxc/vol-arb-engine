/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#0d1117",
        panel: "#161b22",
        border: "#30363d",
        text: "#e6edf3",
        "text-muted": "#8b949e",
        primary: "#58a6ff",
        success: "#3fb950",
        danger: "#f85149",
        warning: "#d29922",
      }
    },
  },
  plugins: [],
}
