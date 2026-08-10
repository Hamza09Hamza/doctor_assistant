import i18n from '@ohif/i18n';
import { id } from './id';
import getPanelModule from './getPanelModule';
import getCustomizationModule from './getCustomizationModule';
import { findSeriesByDicomUid } from './apiClient';
import registerMedSAMBoxTool from './tools/registerMedSAMBoxTool';

let unregisterMedSAMBoxTool: (() => void) | undefined;

/**
 * Candidate-findings panel for doctor_assistant. Talks to the API (see
 * ../../../../../api/) over plain fetch() — see apiClient.ts. No custom viewport,
 * commands, or hanging-protocol behavior needed for this slice; just the panel and
 * a couple of WorkList customizations (getCustomizationModule.tsx) for a
 * general/non-specialist audience.
 */
const doctorAssistantExtension = {
  id,
  // Overrides the StudyList empty-state string (platform/ui-next's Table.tsx hardcodes
  // t('No studies available'), namespace 'StudyList' — no ReactNode slot exists, only
  // the string is reachable, and only via an i18n resource override, not a component
  // edit). preRegistration runs at extension bootstrap, before any route is entered.
  preRegistration() {
    i18n.addResourceBundle(
      'en-US',
      'StudyList',
      { 'No studies available': 'No scans here yet — import a study to get started.' },
      true,
      true
    );
    // "OHIF Viewer is" used to be hardcoded JSX with no i18n key at all — made
    // translatable via a one-line, precisely-scoped core edit (t('OHIF Viewer is'),
    // same flagged-exception precedent as WorkList.tsx's bg-black fix) specifically
    // so the brand name could be overridden here rather than guessed at with a
    // second core edit. "Learn more about OHIF Viewer" and its https://ohif.org/
    // link are left pointing at the real OHIF project — overriding them to a
    // cliniqueamina.com URL would mean fabricating a page that doesn't exist.
    i18n.addResourceBundle(
      'en-US',
      'InvestigationalUseDialog',
      {
        'OHIF Viewer is': 'Clinique Amina is',
        'for investigational use only': 'an experimental research tool — not a medical diagnosis',
      },
      true,
      true
    );
  },
  getPanelModule,
  getCustomizationModule,
  // Registers the MedSAMBoxTool (draw-a-box interactive segmentation, see
  // tools/registerMedSAMBoxTool.ts) and its annotation-completed handler. Re-entering
  // the mode re-registers idempotently (addTool/toolGroup.addTool both guard against
  // duplicates internally); the previous listener is torn down first regardless.
  onModeEnter: ({ servicesManager }: withAppTypes): void => {
    unregisterMedSAMBoxTool?.();
    const { unsubscribe } = registerMedSAMBoxTool({
      servicesManager,
      seriesIdForSeriesUid: async (seriesInstanceUid: string) => {
        const series = await findSeriesByDicomUid(seriesInstanceUid);
        return series.id;
      },
    });
    unregisterMedSAMBoxTool = unsubscribe;
  },
  onModeExit: (): void => {
    unregisterMedSAMBoxTool?.();
    unregisterMedSAMBoxTool = undefined;
  },
};

export default doctorAssistantExtension;
