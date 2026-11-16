// Synkar ikonuppsättning från build/<app>-icons till build/ och src/renderer/ (mönster från SlashCAD/SlashOBD).
import { copyFileSync, existsSync, mkdirSync, readdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));
const appSlug = process.env.SLASH_ICON_SLUG || "slashstitch";
const src = join(root, "build", `${appSlug}-icons`);
const hicolor = join(src, "hicolor");
if (!existsSync(hicolor)) throw new Error(`icon set not found: ${src}`);

const icons = join(root, "build", "icons");
rmSync(icons, { recursive: true, force: true });
mkdirSync(icons, { recursive: true });

for (const size of readdirSync(hicolor)) {
  const png = join(hicolor, size, "apps", `${appSlug}.png`);
  if (/^\d+x\d+$/.test(size) && existsSync(png)) {
    copyFileSync(png, join(icons, `${size}.png`));
  }
}

const master = join(src, `${appSlug}-1024.png`);
if (existsSync(master)) copyFileSync(master, join(root, "build", "icon.png"));
const ico = join(src, `${appSlug}.ico`);
if (existsSync(ico)) copyFileSync(ico, join(root, "build", "icon.ico"));

const pub = join(root, "src", "renderer");
mkdirSync(pub, { recursive: true });
for (const n of [64, 256]) {
  const png = join(hicolor, `${n}x${n}`, "apps", `${appSlug}.png`);
  if (existsSync(png)) { copyFileSync(png, join(pub, `icon-${n}.png`)); mkdirSync(join(pub, "public"), { recursive: true }); copyFileSync(png, join(pub, "public", `icon-${n}.png`)); }
}
console.log("icons synced from", src);
