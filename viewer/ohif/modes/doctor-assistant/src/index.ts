import { id } from './id';
import {
  isValidMode,
  layoutTemplate,
  modeFactory,
  modeInstance as basicModeInstance,
  basicLayout,
  cornerstone,
  extensionDependencies as baseExtensionDependencies,
} from '@ohif/mode-basic';

/**
 * Extends @ohif/mode-basic the way modes/basic-test-mode/src/index.ts does
 * (object-spread over the real basicModeInstance), plus UI/UX simplification for
 * this product's actual audience: clinicians reviewing imaging evidence. The
 * viewport remains dominant while a persistent Detect -> Inspect -> Compare rail
 * carries the AI workflow.
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
    // Keep the product-specific findings surface first, then expose OHIF's
    // read-only segmentation panel for DICOM SEG objects produced by
    // TotalSegmentator.  Editing remains disabled by the inherited basic-mode
    // customization: these are model results for visual QC, not user-authored
    // clinical contours.
    rightPanels: [doctorAssistantPanel.findings, cornerstone.segmentation],
    rightPanelClosed: false,
    rightPanelResizable: true,
    rightPanelInitialExpandedWidth: 376,
    rightPanelMinimumExpandedWidth: 336,
    leftPanelClosed: true,
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
  displayName: 'Chest CT AI Review',
  hide: false,
  isValidMode,
  routes: [doctorAssistantRoute],
  extensions: extensionDependencies,
  // Keep the review rail visible when DICOM SEG objects hydrate. The doctor can
  // deliberately open the comparison panel from the final workflow step without
  // losing candidate/result context mid-review.
  activatePanelTriggers: [],
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
      // Clinical essentials stay visible and use their conventional names. The
      // specialized AI action remains one deliberate control rather than another
      // generic measurement dropdown item.
      primary: [
        'WindowLevel',
        'Zoom',
        'Pan',
        'StackScroll',
        'MeasurementTools',
        'MedSAMBox',
        'Layout',
        'MoreTools',
      ],
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
