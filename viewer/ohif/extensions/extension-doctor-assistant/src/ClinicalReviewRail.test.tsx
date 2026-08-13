import React from 'react';
import { fireEvent, render, screen } from '@testing-library/react';

import type { LungNoduleDetectionResult, SeriesInfo } from './apiClient';
import ClinicalReviewRail from './ClinicalReviewRail';
import type { SeriesReviewWorkflowState } from './useReviewWorkflowStore';

jest.mock('@ohif/ui-next', () => ({
  Button: ({ children, variant: _variant, ...props }) => <button {...props}>{children}</button>,
  Icons: {
    LoadingOHIFMark: props => (
      <span
        data-icon="loading"
        {...props}
      />
    ),
  },
}));

const series: SeriesInfo = {
  id: 'series-a',
  study_id: 'study-a',
  dicom_series_uid: '1.2.3.4',
  modality: 'ct',
  body_part: 'chest',
  analysis_eligible: true,
  ineligible_reason: null,
};

const detectionResult: LungNoduleDetectionResult = {
  series_id: series.id,
  model_version: 'detector:test',
  run_id: 'run-a',
  cache_status: 'miss',
  elapsed_ms: 18400,
  generated_at: '2026-08-13T12:00:00Z',
  source_fingerprint: 'sha256:source',
  model_fingerprint: 'sha256:model',
  cache_key: 'cache-a',
  min_score: 0.3,
  source_slice_count: 238,
  detections: [
    {
      candidate_id: 'candidate-a',
      score: 0.996,
      center_lps_mm: [1, 2, 3],
      size_whd_mm: [10, 11, 12],
      seed_sop_instance_uid: 'sop-86',
      box_xyxy: [91, 250, 108, 265],
    },
  ],
};

const idleWorkflow: SeriesReviewWorkflowState = {
  candidateScan: { phase: 'idle' },
  selectedCandidateIndex: null,
  review: { phase: 'idle' },
  candidateReviews: {},
  candidateAssessments: {},
  artifactSave: { phase: 'idle' },
};

const callbacks = () => ({
  onDetect: jest.fn(),
  onRerun: jest.fn(),
  onSelectCandidate: jest.fn(),
  onOutlineCandidate: jest.fn(),
  onOpenComparison: jest.fn(),
  onSaveOutline: jest.fn(),
  onSetAssessment: jest.fn(),
});

function renderRail(workflow: SeriesReviewWorkflowState, handlers = callbacks()) {
  const view = render(
    <ClinicalReviewRail
      series={series}
      detectorConfigured
      detectorVersion="detector:test"
      segmenterVersion="medsam2:test"
      referenceSegCount={4}
      workflow={workflow}
      runningSeconds={17}
      {...handlers}
    />
  );
  return { ...view, handlers };
}

describe('ClinicalReviewRail', () => {
  it('presents the doctor workflow as Detect, Inspect, Compare and starts detection', () => {
    const { handlers } = renderRail(idleWorkflow);

    expect(screen.getByRole('heading', { name: 'Find nodule candidates' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'Review in image context' })).toBeTruthy();
    expect(screen.getByRole('heading', { name: 'Review outline and references' })).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Find nodule candidates' }));
    expect(handlers.onDetect).toHaveBeenCalledTimes(1);
  });

  it('selects a candidate without inference, then exposes a separate outline action', () => {
    const handlers = callbacks();
    const completeScan: SeriesReviewWorkflowState = {
      ...idleWorkflow,
      candidateScan: { phase: 'complete', result: detectionResult, elapsedSeconds: 18.4 },
    };
    const { rerender } = renderRail(completeScan, handlers);

    fireEvent.click(screen.getByRole('button', { name: /Candidate 1/i }));
    expect(handlers.onSelectCandidate).toHaveBeenCalledWith(detectionResult.detections[0], 0);
    expect(handlers.onOutlineCandidate).not.toHaveBeenCalled();

    rerender(
      <ClinicalReviewRail
        series={series}
        detectorConfigured
        detectorVersion="detector:test"
        segmenterVersion="medsam2:test"
        referenceSegCount={4}
        workflow={{ ...completeScan, selectedCandidateIndex: 0 }}
        runningSeconds={17}
        {...handlers}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: 'Generate 3D outline' }));
    expect(handlers.onOutlineCandidate).toHaveBeenCalledWith(detectionResult.detections[0], 0);
  });

  it('shows measurements, reader agreement, persistence, and doctor disposition', () => {
    const handlers = callbacks();
    const workflow: SeriesReviewWorkflowState = {
      ...idleWorkflow,
      candidateScan: { phase: 'complete', result: detectionResult, elapsedSeconds: 18.4 },
      selectedCandidateIndex: 0,
      review: {
        phase: 'complete',
        seriesId: series.id,
        source: 'detector-candidate',
        candidateNumber: 1,
        candidateScore: 0.996,
        segmentationId: 'segmentation-1',
        label: 'Candidate 1 outline',
        modelVersion: 'medsam2:test',
        segmentedSliceCount: 4,
        volumeMl: 0.59,
        axialBoxDiagonalMm: 13.5,
        craniocaudalExtentMm: 5,
        dicomSegSopInstanceUid: 'seg-sop-1',
        orthancStatus: 'disabled',
        artifact: {
          downloadPath: '/v1/artifacts/seg-sop-1',
          sha256: 'abc123',
          byteLength: 1024,
          seriesInstanceUid: 'seg-series-1',
          sopInstanceUid: 'seg-sop-1',
        },
        referenceComparison: {
          reference_set: 'lidc-readers',
          matching_method: 'prompt-overlap',
          reader_count: 4,
          matched_reader_count: 4,
          consensus_rule: 'at-least-three-readers',
          consensus_reader_threshold: 3,
          consensus_available: true,
          consensus_dice: 0.812,
          consensus_voxel_count: 800,
          consensus_volume_ml: 0.56,
          consensus_segmented_slice_count: 4,
          readers: [],
        },
        warning: null,
      },
      candidateReviews: {},
      candidateAssessments: {},
      artifactSave: { phase: 'idle' },
    };

    renderRail(workflow, handlers);

    expect(screen.getByText('0.59')).toBeTruthy();
    expect(screen.getByText('Dice 0.812')).toBeTruthy();
    expect(screen.getByText(/Matched 4 of 4 readers/)).toBeTruthy();

    fireEvent.click(screen.getByRole('button', { name: 'Compare overlays' }));
    fireEvent.click(screen.getByRole('button', { name: 'Save DICOM SEG to case' }));
    fireEvent.click(screen.getByRole('button', { name: 'Supported' }));
    expect(handlers.onOpenComparison).toHaveBeenCalledTimes(1);
    expect(handlers.onSaveOutline).toHaveBeenCalledTimes(1);
    expect(handlers.onSetAssessment).toHaveBeenCalledWith('supported');
  });

  it('does not allow a detector re-run while an outline owns the inference slot', () => {
    const completeScan: SeriesReviewWorkflowState = {
      ...idleWorkflow,
      candidateScan: { phase: 'complete', result: detectionResult, elapsedSeconds: 18.4 },
      selectedCandidateIndex: 0,
      review: {
        phase: 'running',
        seriesId: series.id,
        source: 'detector-candidate',
        candidateNumber: 1,
        candidateScore: 0.996,
      },
    };

    renderRail(completeScan);

    expect(
      (screen.getByRole('button', { name: 'Re-run model' }) as HTMLButtonElement).disabled
    ).toBe(true);
  });
});
