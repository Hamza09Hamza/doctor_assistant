import { eventTarget } from '@cornerstonejs/core';

import type { LungNoduleCandidate, ReferenceComparison } from './apiClient';

export const LUNG_NODULE_CANDIDATE_SELECTED =
  'DOCTOR_ASSISTANT_LUNG_NODULE_CANDIDATE_SELECTED';
export const LUNG_NODULE_CANDIDATE_OUTLINE_REQUESTED =
  'DOCTOR_ASSISTANT_LUNG_NODULE_CANDIDATE_OUTLINE_REQUESTED';

export const INTERACTIVE_REVIEW_STARTED =
  'DOCTOR_ASSISTANT_INTERACTIVE_REVIEW_STARTED';
export const INTERACTIVE_REVIEW_COMPLETED =
  'DOCTOR_ASSISTANT_INTERACTIVE_REVIEW_COMPLETED';
export const INTERACTIVE_REVIEW_FAILED =
  'DOCTOR_ASSISTANT_INTERACTIVE_REVIEW_FAILED';

export interface LungNoduleCandidateSelectedDetail {
  seriesId: string;
  candidateIndex: number;
  candidate: LungNoduleCandidate;
}

export interface InteractiveReviewContext {
  seriesId: string;
  source: 'detector-candidate' | 'manual-box';
  candidateNumber?: number;
  candidateScore?: number;
}

export interface InteractiveReviewCompletedDetail extends InteractiveReviewContext {
  segmentationId: string;
  label: string;
  modelVersion: string;
  segmentedSliceCount: number;
  volumeMl: number | null;
  axialBoxDiagonalMm: number | null;
  craniocaudalExtentMm: number | null;
  dicomSegSopInstanceUid: string | null;
  orthancStatus: 'published' | 'disabled' | 'failed' | 'not-created';
  artifact: {
    downloadPath: string;
    sha256: string;
    byteLength: number;
    seriesInstanceUid: string;
    sopInstanceUid: string;
  } | null;
  referenceComparison: ReferenceComparison | null;
  warning: string | null;
}

export interface InteractiveReviewFailedDetail extends Partial<InteractiveReviewContext> {
  message: string;
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

export function outlineLungNoduleCandidate(
  detail: LungNoduleCandidateSelectedDetail
): void {
  eventTarget.dispatchEvent(
    new CustomEvent<LungNoduleCandidateSelectedDetail>(
      LUNG_NODULE_CANDIDATE_OUTLINE_REQUESTED,
      { detail }
    )
  );
}

export function publishInteractiveReviewStarted(detail: InteractiveReviewContext): void {
  eventTarget.dispatchEvent(
    new CustomEvent<InteractiveReviewContext>(INTERACTIVE_REVIEW_STARTED, { detail })
  );
}

export function publishInteractiveReviewCompleted(
  detail: InteractiveReviewCompletedDetail
): void {
  eventTarget.dispatchEvent(
    new CustomEvent<InteractiveReviewCompletedDetail>(INTERACTIVE_REVIEW_COMPLETED, { detail })
  );
}

export function publishInteractiveReviewFailed(detail: InteractiveReviewFailedDetail): void {
  eventTarget.dispatchEvent(
    new CustomEvent<InteractiveReviewFailedDetail>(INTERACTIVE_REVIEW_FAILED, { detail })
  );
}
