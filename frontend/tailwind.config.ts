import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Apple-minimalist + nature palette
        canvas: "#f5f3ee",
        ink: "#1a1a1a",
        muted: "#6e6a60",
        line: "#d8d3c6",
        accent: "#7a8060", // matches LAND material from backend
        accentRoute: "#c44545", // matches ROUTE material
      },
      fontFamily: {
        sans: ["ui-sans-serif", "system-ui", "-apple-system", "BlinkMacSystemFont", "sans-serif"],
      },
      letterSpacing: {
        tightish: "-0.01em",
      },
    },
  },
  plugins: [],
} satisfies Config;
