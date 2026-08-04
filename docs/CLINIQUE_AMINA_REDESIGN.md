# Clinique Amina: how the OHIF viewer was rebranded, and how to keep changing it

This documents everything done to turn the vendored OHIF Viewer at `viewer/ohif/` into
the Clinique Amina-branded review UI — what OHIF actually is versus what we built on top
of it, every file touched and why, the two real bugs hit along the way (and the general
lessons they teach about this codebase), and a practical guide for making further design
changes without re-discovering all of this the hard way.

## 1. What OHIF is, and what "our UI" actually means

`viewer/ohif/` is a **vendored copy** of the open-source [OHIF Viewers](https://ohif.org/)
monorepo (history stripped, tracked as regular files in this repo — see the original
Phase 3 plan in `RADIOLOGY_WORKFLOW_COPILOT_VISION.md` for why vendoring was chosen over a
separate checkout). It is a real, general-purpose DICOM viewer built by the Open Health
Imaging Foundation, used by hospitals and research institutions worldwide. We did not
write the image rendering, the DICOM protocol handling, the pan/zoom/window-level tools,
or the underlying React/service architecture — that is all stock OHIF (`platform/core`,
`extensions/cornerstone`, `extensions/default`, etc.), and it should stay stock: it is
maintained upstream, security-reviewed, and re-implementing it would be enormous wasted
effort for zero benefit.

**"Our UI" is everything layered on top via OHIF's own customization surface:**
- `extensions/extension-doctor-assistant/` — a new extension package we wrote (the
  findings panel, its API client, WorkList column/toolbar customizations, i18n overrides).
- `modes/doctor-assistant/` — a new mode package we wrote (which panels/tools/routes are
  active, extending the stock `@ohif/mode-basic` mode via object-spread rather than
  forking it).
- `platform/app/public/config/doctor_assistant.js` — our app config (data sources, the
  Clinique Amina logo, the investigational-use dialog).
- A small number of **shared design tokens** (`platform/ui-next/src/tailwind.css`,
  `platform/ui-next/tailwind.config.js`, `platform/app/tailwind.config.js`) — these are
  global CSS variables and Tailwind config, not React components; changing them re-skins
  every component that reads them (ours and OHIF's stock ones) without forking anything.
- Two flagged, single-line **core edits** — the only places we touched OHIF's own
  component source directly, each justified individually below (§4).

Everything else under `platform/`, `extensions/` (besides `extension-doctor-assistant`),
and `modes/` (besides `doctor-assistant`) is unmodified stock OHIF.

## 1a. Pass 3 — elevation, a real brand mark, and why pass 2 wasn't enough

Pass 2 (the palette/typography rewrite above) was a **token swap**: new CSS variable
values, still consumed by the exact same flat, bordered, shadowless component styling
OHIF ships by default, with a plain-text logo. That's not a redesign, it's a re-skin —
correctly called out. Pass 3 changes the actual visual language, still entirely inside
files we own (no new core edits):

- **Elevation system** (`platform/ui-next/src/tailwind.css`): three brand-tinted shadow
  tokens (`--shadow-brand-sm/-/-lg`, tinted toward the primary teal hue instead of neutral
  black) and matching `.shadow-brand*` utility classes, plus `--primary-hover`/
  `--accent-hover` tokens and `.bg-primary-hover`/`.bg-accent-hover` utilities — the stock
  `Button` component only darkens via opacity (`bg-primary/85 → /100`), which is barely
  visible on an already-saturated color; these give a real, visible hover state.
  Radius also went from `0.625rem` to `0.875rem` — rounder cards read as softer/more
  boutique than OHIF's sharper stock corners.
- **A real monogram mark**, not text-plus-a-dot: a gold-ringed circular "A" in teal,
  used identically in two places — `DoctorAssistantPanel.tsx`'s `PanelBrandHeader` and
  the WorkList logo slot in `config/doctor_assistant.js` — so the findings panel and the
  landing page read as the same product, not two different re-skins.
- **Findings/recommendation cards** (`DoctorAssistantPanel.tsx`) moved from flat
  `border` boxes to elevated (`shadow-brand-sm`) cards with a colored **left accent bar**:
  teal for high-confidence findings (probability ≥ 0.7), and the matching
  success/warning/error hue for each recommendation's urgency — so scanning the list for
  "what matters" doesn't require reading every badge individually.
- **Run Analysis** became its own elevated card with a serif heading instead of being
  wrapped in the same collapsible `PanelSection` accordion used for content lists — a
  single always-visible CTA doesn't need to be collapsible.

**What's still stock, and why (documented honestly, not silently skipped):** the icon
*glyph* set (`Icons.*` from `platform/ui-next`) is still OHIF's own SVG library. Icons
already inherit `currentColor`, so they pick up the brand palette correctly wherever
they're used (see `NotificationInfo.tsx` — the circle is `fill="currentColor"`), but the
actual shape/style of each glyph is unchanged. Replacing the glyph set itself is a
different order of work — 100+ SVGs across the entire vendored app, most of them in
stock OHIF components we don't own — and wasn't pursued in this pass. If a fully custom
icon language becomes a priority, that's a scoped follow-up, not a quiet gap.
Similarly, the core viewport chrome (the actual image canvas, its toolbar shell
structure) stays OHIF's own layout — trimmed down in pass 1 (§3) but not visually rebuilt,
since that's exactly the "don't fork core rendering" line from `AGENTS.md` §1.

## 2. Design direction: why these choices, not others

The brand target is **Clinique Amina**, a real clinic, presented to patients — not
radiologists. Two design passes got us here:

**Pass 1 (light-first redesign).** The original theme was OHIF's own dark "reading room"
convention (every stock preset in `platform/ui-next/src/themes/themes.css` is dark). That
was rejected outright: dark, dense, jargon-heavy UI reads as intimidating and
"robotic" to a non-radiologist patient, which is the opposite of the goal. Pass 1 moved
to a light, warm palette (medical teal + health green, ivory background) and trimmed the
toolbar down to only what a lay user needs (see §3 for what was cut and why).

**Pass 2 (Clinique Amina brand, this document's main subject).** Pass 1's palette was
functionally right (light, warm, non-clinical) but still generic — it read as "a nice
light SaaS theme," not "a specific clinic's product." This pass:
- **Deepened the teal** from a flatter 192°/82%/31% to a richer 174°/62%/24%
  emerald-teal — reads as boutique/considered rather than default-SaaS-blue.
- **Moved the accent color off green (142°) onto warm gold (38°/68%/42%).** This is not
  cosmetic — green was previously used for two unrelated things at once: "brand accent"
  (`--accent`) and "positive/success result" (the `--success-*` badge tokens, also green).
  Gold gives the brand its own distinct, premium-reading color and leaves green free to
  mean exactly one thing: a good clinical result.
- **Warmed the background** slightly further (34°/42%/96.5%), and added a genuine serif
  display typeface — **Playfair Display** for headings/wordmark, paired with the existing
  **Inter** for body text (the "Classic Elegant" pairing: high contrast between an
  elegant serif and a clean, legible sans, standard for premium/boutique brands without
  sacrificing the density a viewer UI needs for its actual content).
- **Added an explicit Clinique Amina identity**: a brand strip at the top of the findings
  panel, a wordmark replacing OHIF's own logo in the WorkList toolbar, and the
  investigational-use dialog now says "Clinique Amina is an experimental research tool"
  instead of "OHIF Viewer is...".

These recommendations came from three separate `ui-ux-pro-max` design-system queries
(general redesign, boutique-clinic style search, healthcare typography search), which
independently converged on the teal/green medical hue family and flagged AI
purple/pink gradients as an anti-pattern for health products — both honored here. Where
the tool's raw suggestion was overridden (e.g. its default cyan-tinted light background
in pass 1, kept warm/ivory instead; its default green accent in pass 2, moved to gold),
that override is recorded inline as a code comment at the point of the decision, not just
here — see `platform/ui-next/src/tailwind.css`'s `:root` comment block.

## 3. Every file touched, and which mechanism it uses

| File | Mechanism | What it does |
|---|---|---|
| `platform/ui-next/src/tailwind.css` | Design tokens (CSS vars) | The full color palette — `:root`/`.dark` HSL variables consumed by every `bg-*`/`text-*`/`border-*` Tailwind utility app-wide. |
| `platform/app/tailwind.config.js` | Design tokens (Tailwind config) | **The actual final font-family config** (see §5, gotcha #2) — `serif` now resolves to Playfair Display. |
| `platform/ui-next/tailwind.config.js` | Design tokens (Tailwind config) | `success`/`warning`/`error`/`info` badge color wiring; also declares `serif`/`sans` (harmless but shadowed by the app-level config above — kept for consistency). |
| `platform/app/public/html-templates/index.html` | Static template | Page `<title>`, `theme-color`, `application-name` meta, and the Google Fonts `<link>` for Inter + Playfair Display. |
| `platform/app/public/config/doctor_assistant.js` | App config | Data sources, `investigationalUseDialog` config, and `whiteLabeling.createLogoComponentFn` (the Clinique Amina wordmark in the WorkList toolbar). |
| `modes/doctor-assistant/src/index.ts` | Mode | Which panels are active (`rightPanels`), which toolbar sections/buttons show (`toolbarSections`), route registration. |
| `extensions/extension-doctor-assistant/src/getCustomizationModule.tsx` | Extension, Default-scope `CustomizationService` | WorkList column trim, toolbar button relabeling, viewport overlay trim — all via `$apply` commands merged against OHIF's own registered defaults. |
| `extensions/extension-doctor-assistant/src/index.tsx` | Extension `preRegistration` | i18n string overrides (StudyList empty state, investigational-use dialog copy — including the brand name swap). |
| `extensions/extension-doctor-assistant/src/DoctorAssistantPanel.tsx` | Extension panel component | The findings panel itself: brand header, trust banner, findings/recommendation cards, urgency badges, serif section headers. |
| `platform/ui-next/src/components/InvestigationalUseDialog/InvestigationalUseDialog.tsx` | **Core edit (flagged)** | Wrapped the hardcoded `"OHIF Viewer is"` string in `t()` so it becomes overridable — see §4. |
| `platform/app/src/routes/WorkList/WorkList.tsx` | **Core edit (flagged)** | `bg-black` → `bg-background` on the page's outer shell — see §4. |

## 4. The two core edits, and why they were unavoidable

Per `viewer/ohif/AGENTS.md`: *"Do not modify the core... only as a last resort if all
other fail or there's an architectural constraint."* Both edits below were made only
after confirming no extension/mode/customization surface reaches the thing being changed.

1. **`WorkList.tsx:128`, `bg-black` → `bg-background`.** This was the *only* hardcoded
   surface color in the entire WorkList route (`className` is a JSX literal, not
   customization-driven) — everything else already correctly used theme tokens. Left
   as-is, the very first screen a patient sees renders as a stark black frame around an
   otherwise warm, light app. One line, one class, no other behavior touched.

2. **`InvestigationalUseDialog.tsx:73`, `OHIF Viewer is` → `{t('OHIF Viewer is')}`.**
   The string was plain JSX text with **no i18n key at all** — `t()` was already used two
   lines below it for `'for investigational use only'`, but the "OHIF Viewer is" prefix
   itself wasn't wrapped, so no extension-level `i18n.addResourceBundle()` override could
   reach it. Wrapping it in `t()` doesn't change what renders by default (i18next falls
   back to the literal key text when no translation is registered) — it only makes the
   string reachable for override, which `extension-doctor-assistant`'s `preRegistration`
   then uses to render "Clinique Amina is..." instead. The adjacent "Learn more about
   OHIF Viewer" link (and its `https://ohif.org/` target) was deliberately **left
   pointing at the real OHIF project** — rewriting it to a `cliniqueamina.com` URL would
   mean fabricating a page that doesn't exist.

No other OHIF component source was touched. If a future design change seems to need a
third core edit, exhaust the customization surface first (see §6) — in both cases above,
the fix ended up being one line specifically because everything else was already
reachable through configuration.

## 5. Two real bugs hit during this work, and the general lessons

These aren't just "bugs we fixed" — they're structural gotchas in how this specific
vendored setup resolves configuration, worth knowing before making further changes.

**Gotcha #1 — `modeCustomizations` can't hold `$apply` commands for *other* customization
IDs.** An earlier attempt nested `'cornerstone.toolbarButtons': { $apply: ... }` inside
the mode's own named customization block (`doctorAssistantModeCustomizations`,
referenced via `modeCustomizations: 'doctorAssistantModeCustomizations'`). This crashed
at app init with `TypeError: Cannot read properties of undefined (reading
'cornerstone.toolbarButtons')`. Root cause: `platform/app/src/appInit.js` eagerly
registers **every** mode's `customizations` export at Default scope, keyed by its own
top-level name — so `doctorAssistantModeCustomizations` itself becomes a *fresh* Default
customization with no prior value. When that fresh value contains a nested `$apply`,
immutability-helper tries to walk into `undefined['cornerstone.toolbarButtons']` and
throws. **The fix:** `$apply`/`$set` commands only work when applied directly against an
*already-registered* top-level customization ID (like `'cornerstone.toolbarButtons'`
itself, which the `cornerstone` extension registers first) — so those two overrides moved
into `extension-doctor-assistant`'s `getCustomizationModule.tsx` (Default scope,
registered after cornerstone) instead of the mode's customization wrapper. **Lesson:** a
mode's own `customizations` block (see `modes/basic/src/index.tsx`'s
`basicModeCustomizations` for the working pattern) is for *plain values*
(`'panelSegmentation.disableEditing': true`), not for re-issuing commands against other
extensions' customization keys — those belong in an extension's `getCustomizationModule`.

**Gotcha #2 — `platform/app/tailwind.config.js` silently shadows
`platform/ui-next/tailwind.config.js`'s `fontFamily`.** The app's final Tailwind config
uses `presets: [ui-tailwind-config, ui-next-tailwind-config]`, but *also* defines its own
top-level (non-`extend`) `theme.fontFamily`. Tailwind's preset-merge rule: a **non-extend**
`theme.*` key in the final config completely replaces the same key from every preset —
it doesn't merge. So editing `serif` in `ui-next/tailwind.config.js` alone compiled
cleanly, showed no error, and simply had zero effect — `font-serif` kept resolving to
Tailwind's built-in default (Georgia/Cambria/Times) because `platform/app/tailwind.config.js`
is the config that actually wins. The real fix had to go in the app-level file. **Lesson:**
when changing any *non-`extend`* `theme.*` key (fontFamily, fontSize, fontWeight are all
non-extend in this codebase — see the same file), check `platform/app/tailwind.config.js`
first, since it's the final word regardless of what any preset says. `extend.colors` (used
for the actual palette) does *not* have this problem — `extend` always merges.

## 6. How to make further design changes

**To re-theme colors:** edit the HSL values in `platform/ui-next/src/tailwind.css`'s
`:root` block (keep `.dark` identical — there is no dark-mode toggle, it's a
compatibility shim only). Every `bg-primary`, `text-foreground`, `border-border`, etc.
utility across the entire app — stock OHIF components included — picks this up
automatically; no component edits needed.

**To change typography:** edit `platform/app/tailwind.config.js`'s `theme.fontFamily`
(not the `ui-next` one — see Gotcha #2). Update the Google Fonts `<link>` in
`platform/app/public/html-templates/index.html` to match if you change the actual
typeface, not just which slot (`sans`/`serif`) uses which font.

**To change which toolbar buttons/panels/columns show:** this is `modes/doctor-assistant/`
territory for structural layout (`rightPanels`, `toolbarSections` primary/MoreTools
arrays) and `extensions/extension-doctor-assistant/src/getCustomizationModule.tsx` for
`$apply`-style filtering/relabeling of *existing* customization IDs (`workList.columns`,
`cornerstone.toolbarButtons`, `viewportOverlay.*`). If you're relabeling or filtering
something OHIF's own extensions already registered, it goes in
`getCustomizationModule.tsx` (Default scope) — not the mode's `customizations` block
(see Gotcha #1).

**To change the findings panel's copy or layout:**
`extensions/extension-doctor-assistant/src/DoctorAssistantPanel.tsx` directly — it's our
own component, not OHIF's. Don't touch `useActiveSeriesInstanceUid()`, the `PanelState`
union, or the three `useEffect`s — that's the pub/sub data-fetching logic, orthogonal to
presentation (see the file's own header comment).

**To change brand strings elsewhere in OHIF's own components** (dialogs, empty states,
etc.): first check whether the string is already wrapped in `t('...')` — if so, override
it via `i18n.addResourceBundle()` in `extension-doctor-assistant/src/index.tsx`'s
`preRegistration()`, same pattern as the existing three overrides there. If it's *not*
wrapped (hardcoded JSX text with no i18n key), that's the one legitimate reason for a
further core edit — but exhaust every other option first, and keep the edit to "wrap this
one string in `t()`" the way `InvestigationalUseDialog.tsx` was, not a broader rewrite.

**Before shipping any change:** this sandbox has no working browser (Playwright/Chromium
install fails here — unsupported OS, no sudo), so verification during this work was
compile-status (`0 rspack/TS errors`) plus served-bundle content checks (`curl | grep`
for the expected token/string in the correct compiled chunk), not a real visual render.
Always do a real hard-refresh check (`Ctrl+Shift+R` — the PWA service worker caches
aggressively) in an actual browser before considering a design change complete.
