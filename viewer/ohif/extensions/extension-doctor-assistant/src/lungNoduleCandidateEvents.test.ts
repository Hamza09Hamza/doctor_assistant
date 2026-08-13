jest.mock('@cornerstonejs/core', () => ({
  eventTarget: {
    dispatchEvent: jest.fn(),
  },
}));

import { eventTarget } from '@cornerstonejs/core';

import type { LungNoduleCandidate } from './apiClient';
import {
  INTERACTIVE_REVIEW_COMPLETED,
  INTERACTIVE_REVIEW_FAILED,
  INTERACTIVE_REVIEW_STARTED,
  LUNG_NODULE_CANDIDATE_OUTLINE_REQUESTED,
  LUNG_NODULE_CANDIDATE_SELECTED,
  outlineLungNoduleCandidate,
  publishInteractiveReviewCompleted,
  publishInteractiveReviewFailed,
  publishInteractiveReviewStarted,
  selectLungNoduleCandidate,
} from './lungNoduleCandidateEvents';

const candidate: LungNoduleCandidate = {
  candidate_id: 'candidate-a',
  score: 0.996,
  center_lps_mm: [1, 2, 3],
  size_whd_mm: [10, 11, 12],
  seed_sop_instance_uid: 'sop-86',
  box_xyxy: [91, 250, 108, 265],
};

const dispatchEvent = eventTarget.dispatchEvent as jest.Mock;

function expectLastEvent(type: string, detail: unknown) {
  const event = dispatchEvent.mock.calls.at(-1)?.[0] as CustomEvent;
  expect(event).toBeInstanceOf(CustomEvent);
  expect(event.type).toBe(type);
  expect(event.detail).toBe(detail);
}

describe('interactive review event lifecycle', () => {
  beforeEach(() => dispatchEvent.mockClear());

  it('keeps inspect navigation separate from the expensive outline request', () => {
    const detail = { seriesId: 'series-a', candidateIndex: 0, candidate };

    selectLungNoduleCandidate(detail);
    expectLastEvent(LUNG_NODULE_CANDIDATE_SELECTED, detail);

    outlineLungNoduleCandidate(detail);
    expectLastEvent(LUNG_NODULE_CANDIDATE_OUTLINE_REQUESTED, detail);

    expect(LUNG_NODULE_CANDIDATE_SELECTED).not.toBe(LUNG_NODULE_CANDIDATE_OUTLINE_REQUESTED);
  });

  it('publishes started, completed, and failed states with the same series context', () => {
    const started = {
      seriesId: 'series-a',
      source: 'detector-candidate' as const,
      candidateNumber: 1,
      candidateScore: 0.996,
    };
    publishInteractiveReviewStarted(started);
    expectLastEvent(INTERACTIVE_REVIEW_STARTED, started);

    const completed = {
      ...started,
      segmentationId: 'segmentation-1',
      label: 'Candidate 1 outline',
      modelVersion: 'medsam2:test',
      segmentedSliceCount: 4,
      volumeMl: 0.59,
      axialBoxDiagonalMm: 13.5,
      craniocaudalExtentMm: 5,
      dicomSegSopInstanceUid: 'seg-sop-1',
      orthancStatus: 'disabled' as const,
      artifact: null,
      referenceComparison: null,
      warning: null,
    };
    publishInteractiveReviewCompleted(completed);
    expectLastEvent(INTERACTIVE_REVIEW_COMPLETED, completed);

    const failed = { ...started, message: 'remote worker stopped' };
    publishInteractiveReviewFailed(failed);
    expectLastEvent(INTERACTIVE_REVIEW_FAILED, failed);
  });
});
