jest.mock('@cornerstonejs/core', () => ({ eventTarget: new EventTarget() }));

import { publishInteractiveReviewCompleted } from './lungNoduleCandidateEvents';
import registerReviewWorkflowEvents from './registerReviewWorkflowEvents';
import {
  getSeriesReviewWorkflowState,
  resetReviewWorkflowStoreForTests,
} from './useReviewWorkflowStore';

describe('mode-lifetime review workflow bridge', () => {
  beforeEach(resetReviewWorkflowStoreForTests);

  it('captures a completion while the clinical panel is unmounted', () => {
    const registration = registerReviewWorkflowEvents();
    publishInteractiveReviewCompleted({
      seriesId: 'series-a',
      source: 'detector-candidate',
      candidateNumber: 1,
      candidateScore: 0.996,
      segmentationId: 'segmentation-a',
      label: 'Candidate 1 outline',
      modelVersion: 'medsam2:test',
      segmentedSliceCount: 4,
      volumeMl: 0.59,
      axialBoxDiagonalMm: 13.5,
      craniocaudalExtentMm: 12,
      dicomSegSopInstanceUid: 'seg-sop-a',
      orthancStatus: 'disabled',
      artifact: null,
      referenceComparison: null,
      warning: null,
    });

    const state = getSeriesReviewWorkflowState('series-a');
    expect(state.review.phase).toBe('complete');
    expect(state.candidateReviews[0]?.segmentationId).toBe('segmentation-a');
    registration.unsubscribe();
  });
});
