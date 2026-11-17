import { useEffect, useRef, useState } from "react";

/** Keys the main process maps to fixed URLs (see src/main/app-links.ts). */
type LinkKey = "sponsor" | "bug" | "guide" | "source" | "releases";

function openLink(key: LinkKey, engine?: string) {
  void window.Host.call("open-link", { key, engine });
}

const Heart = () => (
  <svg width="13" height="13" viewBox="0 0 16 16" aria-hidden="true">
    <path d="M8 14s-5.5-3.4-5.5-7.4A3 3 0 0 1 8 4.4a3 3 0 0 1 5.5 2.2C13.5 10.6 8 14 8 14z" fill="currentColor" />
  </svg>
);
const Bug = () => (
  <svg width="13" height="13" viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
    <rect x="4.5" y="5" width="7" height="8.5" rx="3.5" />
    <path d="M6 5V4a2 2 0 0 1 4 0v1M8 8v5M2.5 8.5h2M11.5 8.5h2M3 4.5l1.8 1.3M13 4.5l-1.8 1.3M3 13l1.8-1.3M13 13l-1.8-1.3" />
  </svg>
);
const Info = () => (
  <svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round">
    <circle cx="8" cy="8" r="6.2" />
    <path d="M8 7.2v4M8 4.8v.1" />
  </svg>
);

/**
 * Application title bar: identity and project on the left, support/feedback/version on the
 * right. The bar is a window drag region; native window controls are overlaid by the OS at
 * the right (Windows/Linux) or left (macOS traffic lights).
 */
export function TitleBar({
  platform,
  version,
  projectName,
  projectPath,
  dirty,
  engine,
  showSponsor,
  onShowSponsor,
}: {
  platform: string;
  version: string;
  projectName: string;
  projectPath: string | null;
  dirty: boolean;
  engine: string;
  showSponsor: boolean;
  onShowSponsor: (show: boolean) => void;
}) {
  const [aboutOpen, setAboutOpen] = useState(false);
  const about = useRef<HTMLDivElement>(null);
  const aboutButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!aboutOpen) return;
    const close = (e: MouseEvent) => {
      if (about.current && !about.current.contains(e.target as Node)) setAboutOpen(false);
    };
    const key = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setAboutOpen(false);
        aboutButton.current?.focus();
      }
    };
    window.addEventListener("mousedown", close);
    window.addEventListener("keydown", key);
    return () => {
      window.removeEventListener("mousedown", close);
      window.removeEventListener("keydown", key);
    };
  }, [aboutOpen]);

  return (
    <header className={`titlebar titlebar-${platform === "darwin" ? "mac" : "overlay"}`}>
      <img className="titlebar-icon" src="./icon-64.png" alt="" width={18} height={18} />
      <span className="titlebar-name">SlashStitch</span>
      <span className="titlebar-sep" aria-hidden="true" />
      <span className="titlebar-project" title={projectPath ?? "Unsaved project"}>
        {projectName}
        {dirty && <span className="titlebar-dirty" aria-label="unsaved changes" title="Unsaved changes" />}
      </span>
      <span className="titlebar-drag" />
      {showSponsor && (
        <span className="titlebar-sponsor">
          <button className="tb-pill tb-sponsor" onClick={() => openLink("sponsor")} title="Support SlashStitch on GitHub Sponsors (opens your browser)">
            <Heart />
            Sponsor
          </button>
          <button className="tb-dismiss" onClick={() => onShowSponsor(false)} aria-label="Hide the Sponsor button" title="Hide the Sponsor button (show it again under About)">
            ×
          </button>
        </span>
      )}
      <button className="tb-pill" onClick={() => openLink("bug", engine)} title="Report a bug on GitHub (opens your browser; only version and system details are filled in)">
        <Bug />
        Report a bug
      </button>
      <div className="titlebar-about" ref={about}>
        <button
          ref={aboutButton}
          className="tb-icon"
          aria-label="About SlashStitch"
          aria-haspopup="dialog"
          aria-expanded={aboutOpen}
          title="About SlashStitch"
          onClick={() => setAboutOpen((o) => !o)}
        >
          <Info />
        </button>
        {aboutOpen && (
          <div className="about-popover" role="dialog" aria-label="About SlashStitch">
            <div className="about-head">
              <img src="./icon-64.png" alt="" width={32} height={32} />
              <div>
                <strong>SlashStitch</strong>
                <div className="muted">Version {version} · MIT License</div>
              </div>
            </div>
            <div className="muted about-engine">{engine || "Engine not connected"}</div>
            <div className="about-links">
              <button className="link-button" onClick={() => openLink("guide")}>User guide</button>
              <button className="link-button" onClick={() => openLink("releases")}>Releases</button>
              <button className="link-button" onClick={() => openLink("source")}>Source code</button>
            </div>
            <label className="about-check">
              <input type="checkbox" checked={showSponsor} onChange={(e) => onShowSponsor(e.target.checked)} />
              Show the Sponsor button
            </label>
          </div>
        )}
      </div>
      <span className="titlebar-version" title={`SlashStitch ${version}`}>v{version}</span>
    </header>
  );
}
