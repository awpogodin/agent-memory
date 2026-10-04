import convexPlugin from "@convex-dev/eslint-plugin";
import tseslint from "typescript-eslint";

export default [
  {
    ignores: ["src/component/_generated/**"],
  },
  {
    files: ["src/**/*.ts"],
    languageOptions: {
      parser: tseslint.parser,
    },
    plugins: {
      "@typescript-eslint": tseslint.plugin,
    },
  },
  {
    plugins: {
      "@convex-dev": convexPlugin,
    },
    files: ["src/component/**/*.ts"],
    rules: {
      "@convex-dev/no-old-registered-function-syntax": "error",
      "@convex-dev/require-args-validator": "error",
      "@convex-dev/no-collect-in-query": "error",
    },
  },
];
