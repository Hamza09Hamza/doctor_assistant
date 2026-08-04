import { id } from './id';
import {
  isValidMode,
  layoutTemplate,
  modeFactory,
  modeInstance as basicModeInstance,
  basicLayout,
  extensionDependencies as baseExtensionDependencies,
} from '@ohif/mode-basic';

/**
 * Extends @ohif/mode-basic the way modes/basic-test-mode/src/index.ts does
 * (object-spread over the real basicModeInstance), plus UI/UX simplification for
 * this product's actual audience (general/non-specialist, not radiologists — see
 * the redesign plan): a trimmed toolbar, relabeled controls, a cleaner viewport
 * overlay, and the findings panel as the sole right-side surface.
 */
const doctorAssistantPanel = {
  findings: '@doctor-assistant/extension-doctor-assistant.panelModule.findingsPanel',
};

export const extensionDependencies = {
  ...baseExtensionDependencies,
  '@doctor-assistant/extension-doctor-assistant': '^0.0.1',
};

export const doctorAssistantLayout = {
  ...basicLayout,
  props: {
    ...basicLayout.props,
    // Replaces (not appends to) basic's rightPanels: with MeasurementTools and
    // segmentation dropped from the toolbar below, those panel tabs would be
    // permanent dead ends — the findings panel is this mode's only right-side
    // surface.
    rightPanels: [doctorAssistantPanel.findings],
    rightPanelClosed: false,
  },
};

export const doctorAssistantRoute = {
  path: 'doctor-assistant',
  layoutTemplate,
  layoutInstance: doctorAssistantLayout,
};

export const modeInstance = {
  ...basicModeInstance,
  id,
  routeName: 'doctor-assistant',
  displayName: 'Doctor Assistant',
  hide: false,
  isValidMode,
  routes: [doctorAssistantRoute],
  extensions: extensionDependencies,
  // Composes a second pack on top of basic's `[{ $reference: 'cornerstone.toolbarSections' }]`
  // (packs merge by Object.assign in order, later keys win) rather than patching
  // `toolbarSections.primary` via a modeCustomizations $set — that specific path is a
  // documented no-op (see the redesign plan: Mode.tsx seeds this as an array from the
  // raw value above, and registerModeToolbar's `Object.assign({}, ...toArray(...))`
  // only unpacks indexed elements, silently dropping a non-index `.primary` property
  // attached via $set). Keeping the $reference pack preserves viewportActionMenu /
  // AdvancedRenderingControls, which are unrelated to this trim.
  toolbarSections: [
    { $reference: 'cornerstone.toolbarSections' },
    {
      // MeasurementTools dropped: caliper/angle/ROI tools need clinical training a lay
      // user doesn't have, and keeping them next to "AI Findings" creates two competing
      // "what do I do here" surfaces — the single highest-leverage cut for this audience.
      // TrackballRotate/Crosshairs dropped: volumetric/MPR-only, meaningless for 2D
      // chest X-ray review. Capture dropped for v1 minimalism (trivial to re-add).
      primary: ['WindowLevel', 'Zoom', 'Pan', 'Layout', 'MoreTools'],
      // Trimmed to basic image manipulation only — drops TagBrowser (raw DICOM tag
      // browser), Probe, Cine, angle/calibration tools, StackScroll, etc.
      MoreTools: ['Reset', 'rotate-right', 'flipHorizontal', 'invert', 'Magnify'],
    },
  ],
  // 'cornerstone.toolbarButtons' relabel and 'viewportOverlay.bottomLeft' strip moved
  // to extensions/extension-doctor-assistant/src/getCustomizationModule.tsx (Default
  // scope, applied after cornerstone's own registration) — a mode-scope wrapper here
  // referenced via modeCustomizations can't hold $apply commands keyed by *other*
  // customization ids; appInit.js registers the whole `customizations` export as a
  // fresh Default entry under its own name first, so immutability-helper has nothing
  // to walk into yet and throws. See that file's comment for the full trace.
};

const mode = {
  id,
  modeFactory,
  modeInstance,
  extensionDependencies,
};

export default mode;
