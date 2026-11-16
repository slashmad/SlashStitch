import { build } from "esbuild";
const common = { bundle: true, platform: "node", target: "node22", format: "cjs", sourcemap: true, logLevel: "info", external: ["electron"] };
await Promise.all([
  build({ ...common, entryPoints: ["src/main/index.ts"], outfile: "dist/main/index.js" }),
  build({ ...common, entryPoints: ["src/main/demo-worker.ts"], outfile: "dist/main/demo-worker.js" }),
  build({ ...common, entryPoints: ["src/preload/index.ts"], outfile: "dist/preload/index.js" }),
]);
