import { act } from 'react';
import { renderHook } from '@testing-library/react';

import type { LungNoduleDetectionResult } from './apiClient';
import {
  getSeriesReviewWorkflowState,
  resetReviewWorkflowStoreForTests,
  updateSeriesReviewWorkflow,
  useReviewWorkflowStore,
} from './useReviewWorkflowStore';

const detectionResult: LungNoduleDetectionResult = {
  series_id: 'series-a',
  model_version: 'detector:test',
  min_score: 0.3,
  source_slice_count: 238,
  detections: [
    {
      score: 0.996,
      center_lps_mm: [1, 2, 3],
      size_whd_mm: [10, 11, 12],
      seed_sop_instance_uid: 'sop-86',
      box_xyxy: [91, 250, 108, 265],
    },
  ],
};

describe('review workflow store', () => {
  beforeEach(() => resetReviewWorkflowStoreForTests());

  it('isolates expensive detector output and doctor disposition by source series', () => {
    updateSeriesReviewWorkflow('series-a', {
      candidateScan: { phase: 'complete', result: detectionResult, elapsedSeconds: 18.4 },
      selectedCandidateIndex: 0,
      candidateAssessments: { 0: 'supported' },
    });

    expect(getSeriesReviewWorkflowState('series-a')).toEqual(
      expect.objectContaining({
        selectedCandidateIndex: 0,
        candidateAssessments: { 0: 'supported' },
        candidateScan: expect.objectContaining({ phase: 'complete', result: detectionResult }),
      })
    );
    expect(getSeriesReviewWorkflowState('series-b')).toEqual(
      expect.objectContaining({
        candidateScan: { phase: 'idle' },
        selectedCandidateIndex: null,
        candidateAssessments: {},
      })
    );
  });

  it('merges functional updates without dropping existing review fields', () => {
    updateSeriesReviewWorkflow('series-a', {
      selectedCandidateIndex: 0,
      candidateAssessments: { 0: 'uncertain' },
    });
    updateSeriesReviewWorkflow('series-a', current => ({
      candidateAssessments: { ...current.candidateAssessments, 1: 'dismissed' },
      artifactSave: { phase: 'saved' },
    }));

    expect(getSeriesReviewWorkflowState('series-a')).toEqual(
      expect.objectContaining({
        selectedCandidateIndex: 0,
        candidateAssessments: { 0: 'uncertain', 1: 'dismissed' },
        artifactSave: { phase: 'saved' },
        review: { phase: 'idle' },
      })
    );
  });

  it('notifies a mounted panel and preserves its state across remounts', () => {
    const firstMount = renderHook(() => useReviewWorkflowStore('series-a'));

    act(() => {
      firstMount.result.current[1]({ selectedCandidateIndex: 0 });
    });
    expect(firstMount.result.current[0].selectedCandidateIndex).toBe(0);

    firstMount.unmount();
    const remount = renderHook(() => useReviewWorkflowStore('series-a'));
    expect(remount.result.current[0].selectedCandidateIndex).toBe(0);
  });
});
