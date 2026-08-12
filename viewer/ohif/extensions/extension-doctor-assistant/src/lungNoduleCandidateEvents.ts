import { eventTarget } from '@cornerstonejs/core';

import type { LungNoduleCandidate } from './apiClient';

export const LUNG_NODULE_CANDIDATE_SELECTED =
  'DOCTOR_ASSISTANT_LUNG_NODULE_CANDIDATE_SELECTED';

export interface LungNoduleCandidateSelectedDetail {
  seriesId: string;
  candidateIndex: number;
  candidate: LungNoduleCandidate;
}

/**
 * Keep the panel and Cornerstone tool wiring decoupled through Cornerstone's existing
 * event target. The panel owns candidate selection; the viewport integration owns
 * slice navigation, remote refinement, and labelmap painting.
 */
export function selectLungNoduleCandidate(
  detail: LungNoduleCandidateSelectedDetail
): void {
  eventTarget.dispatchEvent(
    new CustomEvent<LungNoduleCandidateSelectedDetail>(
      LUNG_NODULE_CANDIDATE_SELECTED,
      { detail }
    )
  );
}
