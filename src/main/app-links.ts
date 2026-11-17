// External destinations the renderer may open. The renderer only names a key; the URL is
// fixed here, so a compromised page cannot open arbitrary addresses.

export const APP_LINKS = {
  sponsor: "https://github.com/sponsors/slashmad",
  source: "https://github.com/slashmad/SlashStitch",
  guide: "https://github.com/slashmad/SlashStitch/blob/main/docs/USER_GUIDE.md",
  releases: "https://github.com/slashmad/SlashStitch/releases",
} as const;

const NEW_ISSUE = "https://github.com/slashmad/SlashStitch/issues/new";

export interface BugReportContext {
  version: string;
  platform: string;
  arch: string;
  osRelease: string;
  electron: string;
  /** Short engine description from the UI (backend and threads), sanitized here. */
  engine?: unknown;
}

/** A pre-filled GitHub issue: environment only, no paths, file names or personal data. */
export function bugReportUrl(ctx: BugReportContext): string {
  const engine = typeof ctx.engine === "string" ? ctx.engine.replace(/[^\w .·()/+-]/g, "").slice(0, 80) : "";
  const body = [
    "**What happened?**", "", "", "**Steps to reproduce**", "1. ", "", "**Expected result**", "", "",
    "**Environment**",
    `- SlashStitch ${ctx.version}`,
    `- ${ctx.platform} ${ctx.osRelease} (${ctx.arch})`,
    `- Electron ${ctx.electron}`,
    ...(engine ? [`- Engine: ${engine}`] : []),
    "",
  ].join("\n");
  return `${NEW_ISSUE}?labels=bug&body=${encodeURIComponent(body)}`;
}

/** URL for a renderer link key, or null for anything unknown. */
export function linkFor(key: unknown, ctx: BugReportContext): string | null {
  if (key === "bug") return bugReportUrl(ctx);
  return typeof key === "string" && Object.hasOwn(APP_LINKS, key) ? APP_LINKS[key as keyof typeof APP_LINKS] : null;
}
