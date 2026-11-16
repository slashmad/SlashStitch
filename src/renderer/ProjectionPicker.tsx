import { useEffect, useRef, useState, type KeyboardEvent, type ReactNode } from "react";

/** Panorama projections offered in the viewport, with a minimal shape icon each. */
export const PROJECTION_CHOICES: Array<{ value: string; label: string }> = [
  { value: "equirectangular", label: "Equirectangular (360×180)" },
  { value: "cylindrical", label: "Cylindrical" },
  { value: "mercator", label: "Mercator" },
  { value: "rectilinear", label: "Rectilinear (Flat)" },
  { value: "vedutismo", label: "Vedutismo (Panini)" },
  { value: "stereographic", label: "Stereographic" },
  { value: "stereographic_down", label: "Stereographic Down (Little Planet)" },
  { value: "circular_fisheye", label: "Circular Fisheye" },
  { value: "fullframe_fisheye", label: "Fullframe Fisheye" },
  { value: "transverse_equirectangular", label: "Transverse Equirectangular" },
  { value: "transverse_cylindrical", label: "Transverse Cylindrical" },
  { value: "transverse_mercator", label: "Transverse Mercator" },
  { value: "transverse_vedutismo", label: "Transverse Vedutismo" },
];

/** Base shapes drawn in a 24×24 box (content in the middle band so a 90° turn still fits). */
function baseShape(kind: string): ReactNode {
  switch (kind) {
    case "equirectangular":
      return (
        <>
          <rect x="2" y="6.5" width="20" height="11" rx="0.8" />
          <path d="M2 12h20M7 6.5v11M12 6.5v11M17 6.5v11" opacity="0.55" />
        </>
      );
    case "cylindrical":
      return (
        <>
          <ellipse cx="12" cy="7.5" rx="7" ry="2" />
          <path d="M5 7.5v9M19 7.5v9M5 16.5a7 2 0 0 0 14 0" />
        </>
      );
    case "mercator":
      return (
        <>
          <rect x="2" y="5" width="20" height="14" rx="0.8" />
          <path d="M2 12h20M2 9.6h20M2 14.4h20M2 6.6h20M2 17.4h20" opacity="0.55" />
        </>
      );
    case "rectilinear":
      return (
        <>
          <rect x="3" y="6" width="18" height="12" rx="0.8" />
          <path d="M3 6l6.5 4.5h5L21 6M3 18l6.5-4.5h5L21 18M9.5 10.5v3M14.5 10.5v3" opacity="0.55" />
        </>
      );
    case "vedutismo":
      return (
        <>
          <rect x="2" y="6" width="20" height="12" rx="0.8" />
          <path d="M2 9.5Q12 6.5 22 9.5M2 14.5Q12 17.5 22 14.5M8 6v12M16 6v12" opacity="0.55" />
        </>
      );
    case "stereographic":
      return (
        <>
          <rect x="2" y="6" width="20" height="12" rx="0.8" />
          <path d="M2 13.5Q12 7 22 13.5" />
          <path d="M6 18Q12 14 18 18" opacity="0.55" />
        </>
      );
    case "stereographic_down":
      return (
        <>
          <circle cx="12" cy="12" r="8.5" opacity="0.55" />
          <circle cx="12" cy="12" r="3.5" />
          <path d="M12 8.5V5.5M15.5 12h3M12 15.5v3M8.5 12h-3" />
        </>
      );
    case "circular_fisheye":
      return (
        <>
          <circle cx="12" cy="12" r="8.5" />
          <path d="M3.5 12h17M12 3.5v17" opacity="0.55" />
          <ellipse cx="12" cy="12" rx="4" ry="8.5" opacity="0.55" />
        </>
      );
    case "fullframe_fisheye":
      return (
        <>
          <rect x="3" y="5.5" width="18" height="13" rx="3.5" />
          <path d="M3.5 12h17M12 6v12" opacity="0.55" />
          <path d="M5 8.5Q12 5.5 19 8.5M5 15.5Q12 18.5 19 15.5" opacity="0.55" />
        </>
      );
    default:
      return <rect x="3" y="6" width="18" height="12" rx="1" />;
  }
}

export function ProjectionIcon({ value, size = 22 }: { value: string; size?: number }) {
  const transverse = value.startsWith("transverse_");
  const kind = transverse ? value.slice("transverse_".length) : value;
  return (
    <svg
      className="projection-icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <g transform={transverse ? "rotate(90 12 12)" : undefined}>{baseShape(kind)}</g>
    </svg>
  );
}

/** Accessible dropdown (listbox) for the panorama projection with shape icons. */
export function ProjectionPicker({
  value,
  disabled,
  onChange,
}: {
  value: string;
  disabled?: boolean;
  onChange: (value: string) => void;
}) {
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const root = useRef<HTMLDivElement>(null);
  const button = useRef<HTMLButtonElement>(null);
  const list = useRef<HTMLUListElement>(null);
  const current = PROJECTION_CHOICES.find((c) => c.value === value) ?? PROJECTION_CHOICES[0];

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (root.current && !root.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("mousedown", close);
    return () => window.removeEventListener("mousedown", close);
  }, [open]);

  useEffect(() => {
    if (open) list.current?.querySelectorAll<HTMLElement>("[role=option]")[active]?.focus();
  }, [open, active]);

  const openList = () => {
    if (disabled) return;
    setActive(Math.max(0, PROJECTION_CHOICES.findIndex((c) => c.value === value)));
    setOpen(true);
  };
  // Closing always returns focus to the button, so keyboard users keep their place.
  const close = () => {
    setOpen(false);
    button.current?.focus();
  };
  const choose = (v: string) => {
    close();
    if (v !== value) onChange(v);
  };
  const onListKey = (e: KeyboardEvent) => {
    if (e.key === "ArrowDown") { e.preventDefault(); setActive((a) => Math.min(PROJECTION_CHOICES.length - 1, a + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setActive((a) => Math.max(0, a - 1)); }
    else if (e.key === "Home") { e.preventDefault(); setActive(0); }
    else if (e.key === "End") { e.preventDefault(); setActive(PROJECTION_CHOICES.length - 1); }
    else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); choose(PROJECTION_CHOICES[active].value); }
    else if (e.key === "Escape") { e.preventDefault(); close(); }
    // Tab: focus the button first, then the default action moves on in natural order.
    else if (e.key === "Tab") { close(); }
  };

  return (
    <div className="projection-picker" ref={root}>
      <button
        ref={button}
        type="button"
        className={disabled ? "projection-button is-disabled" : "projection-button"}
        // aria-disabled (not disabled) keeps keyboard focus on the button while a short
        // job runs right after a choice; interaction is ignored meanwhile.
        aria-disabled={disabled || undefined}
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-label={`Projection: ${current.label}`}
        data-value={current.value}
        onClick={() => (open ? setOpen(false) : openList())}
        onKeyDown={(e) => {
          if (e.key === "ArrowDown" || e.key === "ArrowUp") { e.preventDefault(); openList(); }
        }}
      >
        <ProjectionIcon value={current.value} />
        <span className="projection-label">{current.label}</span>
        <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden="true" className="projection-caret">
          <path d="M2 3.5l3 3 3-3" fill="none" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
        </svg>
      </button>
      {open && (
        <ul className="projection-list" role="listbox" aria-label="Projection" ref={list} onKeyDown={onListKey}>
          {PROJECTION_CHOICES.map((c, i) => (
            <li
              key={c.value}
              role="option"
              aria-selected={c.value === value}
              tabIndex={i === active ? 0 : -1}
              className={`projection-option${c.value === value ? " selected" : ""}${i === active ? " active" : ""}`}
              data-value={c.value}
              onMouseEnter={() => setActive(i)}
              onClick={() => choose(c.value)}
            >
              <ProjectionIcon value={c.value} />
              <span>{c.label}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
