import React from 'react';
import { Button, Icons } from '@ohif/ui-next';

import type { LungNoduleCandidate, SeriesInfo } from './apiClient';
import type { SeriesReviewWorkflowState } from './useReviewWorkflowStore';

type StepTone = 'waiting' | 'active' | 'complete' | 'error';

interface ClinicalReviewRailProps {
  series: SeriesInfo;
  detectorConfigured: boolean;
  detectorVersion: string | null;
  segmenterVersion: string | null;
  referenceSegCount: number;
  workflow: SeriesReviewWorkflowState;
  runningSeconds: number;
  onDetect: () => void;
  onRerun: () => void;
  onSelectCandidate: (candidate: LungNoduleCandidate, index: number) => void;
  onOutlineCandidate: (candidate: LungNoduleCandidate, index: number) => void;
  onOpenComparison: () => void;
  onSaveOutline: () => void;
  onSetAssessment: (assessment: 'supported' | 'dismissed' | 'uncertain') => void;
}

export default function ClinicalReviewRail({
  series,
  detectorConfigured,
  detectorVersion,
  segmenterVersion,
  referenceSegCount,
  workflow,
  runningSeconds,
  onDetect,
  onRerun,
  onSelectCandidate,
  onOutlineCandidate,
  onOpenComparison,
  onSaveOutline,
  onSetAssessment,
}: ClinicalReviewRailProps) {
  const candidateResult =
    workflow.candidateScan.phase === 'complete' ? workflow.candidateScan.result : null;
  const selectedCandidate =
    candidateResult && workflow.selectedCandidateIndex != null
      ? candidateResult.detections[workflow.selectedCandidateIndex]
      : null;

  const detectTone: StepTone =
    workflow.candidateScan.phase === 'running'
      ? 'active'
      : workflow.candidateScan.phase === 'complete'
        ? 'complete'
        : workflow.candidateScan.phase === 'error'
          ? 'error'
          : 'waiting';
  const inspectTone: StepTone = selectedCandidate
    ? 'complete'
    : candidateResult
      ? 'active'
      : 'waiting';
  const compareTone: StepTone =
    workflow.review.phase === 'running'
      ? 'active'
      : workflow.review.phase === 'complete'
        ? 'complete'
        : workflow.review.phase === 'error'
          ? 'error'
          : 'waiting';

  return (
    <div className="clinical-review-shell" data-cy="clinical-ai-review">
      <ClinicalHeader
        series={series}
        imageCount={candidateResult?.source_slice_count ?? null}
      />

      <div className="clinical-runtime-line" aria-label="AI runtime status">
        <span className="clinical-live-dot" aria-hidden="true" />
        AI review ready
        {referenceSegCount > 0 && (
          <span className="clinical-runtime-reference">
            {referenceSegCount} reader SEG{referenceSegCount === 1 ? '' : 's'} loaded
          </span>
        )}
      </div>

      <div className="clinical-workflow" aria-live="polite">
        <ReviewStep
          number="01"
          label="Detect"
          title="Find nodule candidates"
          tone={detectTone}
        >
          <p>
            Run the volumetric detector once, then continue with a ranked shortlist. A mark is
            a location to inspect—not a diagnosis.
          </p>

          {!detectorConfigured && (
            <InlineState tone="error" title="Detector unavailable">
              The connected inference runtime did not load the lung-nodule detector.
            </InlineState>
          )}

          {detectorConfigured && workflow.candidateScan.phase === 'idle' && (
            <Button
              variant="default"
              onClick={onDetect}
              data-cy="scan-lung-nodule-candidates"
              className="clinical-primary-action"
            >
              Find nodule candidates
            </Button>
          )}

          {workflow.candidateScan.phase === 'running' && (
            <InlineState tone="active" title="Scanning the complete CT…">
              <span className="clinical-mono">{formatDuration(runningSeconds)}</span> elapsed ·
              one inference runs at a time
            </InlineState>
          )}

          {workflow.candidateScan.phase === 'error' && (
            <>
              <InlineState tone="error" title="Candidate scan failed">
                {cleanError(workflow.candidateScan.message)}
              </InlineState>
              <Button
                variant="secondary"
                onClick={onDetect}
                className="clinical-secondary-action"
              >
                Retry detection
              </Button>
            </>
          )}

          {workflow.candidateScan.phase === 'complete' && (
            <div className="clinical-complete-line">
              <span>
                {workflow.candidateScan.result.detections.length} candidate
                {workflow.candidateScan.result.detections.length === 1 ? '' : 's'}
              </span>
              <span className="clinical-mono">
                {formatDuration(workflow.candidateScan.elapsedSeconds)}
              </span>
              {workflow.candidateScan.result.cache_status === 'hit' && (
                <span className="clinical-cache-note">saved run</span>
              )}
              <button
                type="button"
                onClick={onRerun}
                disabled={workflow.review.phase === 'running' || workflow.artifactSave.phase === 'saving'}
                className="clinical-text-action"
              >
                Re-run model
              </button>
            </div>
          )}
        </ReviewStep>

        <ReviewStep
          number="02"
          label="Inspect"
          title="Review in image context"
          tone={inspectTone}
        >
          {!candidateResult && (
            <p className="clinical-muted-copy">
              The shortlist will appear here. Selecting a row moves the axial viewport without
              spending another inference.
            </p>
          )}

          {candidateResult && candidateResult.detections.length === 0 && (
            <InlineState tone="waiting" title="No candidate crossed the threshold">
              This does not establish that the scan is clear. Continue the image review.
            </InlineState>
          )}

          {candidateResult && candidateResult.detections.length > 0 && (
            <div className="clinical-candidate-list" data-cy="lung-nodule-candidate-list">
              {candidateResult.detections.map((candidate, index) => {
                const isSelected = workflow.selectedCandidateIndex === index;
                return (
                  <CandidateRow
                    key={candidate.candidate_id || `${candidate.seed_sop_instance_uid}-${index}`}
                    candidate={candidate}
                    index={index}
                    selected={isSelected}
                    disabled={workflow.review.phase === 'running'}
                    outlined={Boolean(workflow.candidateReviews[index])}
                    assessment={workflow.candidateAssessments[index]}
                    onSelect={() => onSelectCandidate(candidate, index)}
                  />
                );
              })}
            </div>
          )}

          {selectedCandidate && (
            <div className="clinical-selection-action" data-cy="selected-nodule-candidate">
              <div>
                <span className="clinical-selection-kicker">
                  Candidate {workflow.selectedCandidateIndex! + 1} selected
                </span>
                <strong>
                  {workflow.candidateReviews[workflow.selectedCandidateIndex!]
                    ? 'A 3D outline is available; inspect before regenerating it.'
                    : 'Inspect the location before generating an outline.'}
                </strong>
              </div>
              <Button
                variant="default"
                onClick={() =>
                  onOutlineCandidate(selectedCandidate, workflow.selectedCandidateIndex!)
                }
                disabled={workflow.review.phase === 'running'}
                data-cy="outline-selected-nodule-candidate"
                className="clinical-primary-action"
              >
                {workflow.review.phase === 'running'
                  ? 'Generating 3D outline…'
                  : workflow.candidateReviews[workflow.selectedCandidateIndex!]
                    ? 'Regenerate 3D outline'
                    : 'Generate 3D outline'}
              </Button>
            </div>
          )}
        </ReviewStep>

        <ReviewStep
          number="03"
          label="Compare"
          title="Review outline and references"
          tone={compareTone}
          last
        >
          {workflow.review.phase === 'idle' && (
            <p className="clinical-muted-copy">
              Measurements and the generated labelmap remain here after the outline completes.
            </p>
          )}

          {workflow.review.phase === 'running' && (
            <InlineState tone="active" title="Generating the 3D outline…">
              The selected source slice and box are locked until the result returns.
            </InlineState>
          )}

          {workflow.review.phase === 'error' && (
            <InlineState tone="error" title="Outline failed">
              {cleanError(workflow.review.message)}
            </InlineState>
          )}

          {workflow.review.phase === 'complete' && (
            <div className="clinical-result" data-cy="interactive-review-result">
              <div className="clinical-result-title">
                <span className="clinical-result-check" aria-hidden="true">✓</span>
                <div>
                  <span>Outline ready</span>
                  <strong>{workflow.review.label}</strong>
                </div>
              </div>

              <div className="clinical-metric-grid clinical-metric-grid-four">
                <Metric value={String(workflow.review.segmentedSliceCount)} label="slices" />
                <Metric
                  value={workflow.review.volumeMl == null ? '—' : workflow.review.volumeMl.toFixed(2)}
                  label="mL"
                />
                <Metric
                  value={
                    workflow.review.axialBoxDiagonalMm == null
                      ? '—'
                      : workflow.review.axialBoxDiagonalMm.toFixed(1)
                  }
                  label="mm axial box"
                />
                <Metric
                  value={
                    workflow.review.craniocaudalExtentMm == null
                      ? '—'
                      : workflow.review.craniocaudalExtentMm.toFixed(1)
                  }
                  label="mm C–C"
                />
              </div>

              {workflow.review.candidateScore != null && (
              <div className="clinical-score-line">
                  Detector rank score
                  <span className="clinical-mono">
                    {workflow.review.candidateScore.toFixed(3)}
                  </span>
                  <span className="clinical-score-help" title="Ranking score, not a calibrated probability">
                    not probability
                  </span>
                </div>
              )}

              {workflow.review.referenceComparison?.consensus_available &&
                workflow.review.referenceComparison.consensus_dice != null && (
                  <div className="clinical-agreement" data-cy="reference-comparison-result">
                    <div>
                      <span>Reader consensus agreement</span>
                      <strong className="clinical-mono">
                        Dice {workflow.review.referenceComparison.consensus_dice.toFixed(3)}
                      </strong>
                    </div>
                    <div className="clinical-agreement-bar" aria-hidden="true">
                      <span
                        style={{
                          width: `${workflow.review.referenceComparison.consensus_dice * 100}%`,
                        }}
                      />
                    </div>
                    <p>
                      Matched {workflow.review.referenceComparison.matched_reader_count} of{' '}
                      {workflow.review.referenceComparison.reader_count} readers · ≥
                      {workflow.review.referenceComparison.consensus_reader_threshold} reader
                      voxel consensus ·{' '}
                      {workflow.review.referenceComparison.consensus_volume_ml?.toFixed(2) ?? '—'} mL
                    </p>
                  </div>
                )}

              {workflow.review.referenceComparison &&
                !workflow.review.referenceComparison.consensus_available && (
                  <p className="clinical-safety-note">
                    Reader annotations were present, but no unambiguous prompt-matched consensus
                    object was available for an overlap score.
                  </p>
                )}

              <div className="clinical-result-actions">
                <Button
                  variant="secondary"
                  onClick={onOpenComparison}
                  data-cy="open-segmentation-comparison"
                  className="clinical-secondary-action"
                >
                  Compare overlays
                </Button>
                {workflow.review.artifact && workflow.artifactSave.phase !== 'saved' && (
                  <Button
                    variant="secondary"
                    onClick={onSaveOutline}
                    disabled={workflow.artifactSave.phase === 'saving'}
                    data-cy="save-dicom-seg-to-case"
                    className="clinical-secondary-action"
                  >
                    {workflow.artifactSave.phase === 'saving'
                      ? 'Saving to case…'
                      : workflow.artifactSave.phase === 'error'
                        ? 'Retry save to case'
                        : 'Save DICOM SEG to case'}
                  </Button>
                )}
              </div>

              {workflow.artifactSave.phase === 'saved' && (
                <div className="clinical-saved-line" data-cy="dicom-seg-saved-to-case">
                  ✓ DICOM SEG saved to local Orthanc. Reload the study to hydrate the durable copy.
                </div>
              )}
              {workflow.artifactSave.phase === 'error' && (
                <div className="clinical-save-error">
                  {cleanError(workflow.artifactSave.message || 'Could not save to local Orthanc.')}
                </div>
              )}

              {workflow.review.source === 'detector-candidate' &&
                workflow.selectedCandidateIndex != null && (
                  <div className="clinical-disposition">
                    <span>Review disposition</span>
                    <div role="group" aria-label="Candidate review disposition">
                      <DispositionButton
                        label="Supported"
                        value="supported"
                        current={workflow.candidateAssessments[workflow.selectedCandidateIndex]}
                        onSelect={onSetAssessment}
                      />
                      <DispositionButton
                        label="Dismiss mark"
                        value="dismissed"
                        current={workflow.candidateAssessments[workflow.selectedCandidateIndex]}
                        onSelect={onSetAssessment}
                      />
                      <DispositionButton
                        label="Uncertain"
                        value="uncertain"
                        current={workflow.candidateAssessments[workflow.selectedCandidateIndex]}
                        onSelect={onSetAssessment}
                      />
                    </div>
                    <p>Session review state only; this is not a diagnosis or signed report.</p>
                  </div>
                )}

              <p className="clinical-safety-note">
                The mask follows the selected box. It does not confirm a nodule, malignancy,
                or normality. {persistenceCopy(workflow.review.orthancStatus)}
              </p>
            </div>
          )}

          <div className="clinical-reference-line">
            <span className="clinical-reference-swatch" aria-hidden="true" />
            {referenceSegCount > 0
              ? `${referenceSegCount} individual reader reference SEG${referenceSegCount === 1 ? '' : 's'} available`
              : 'No reader reference SEG loaded for this study'}
          </div>
        </ReviewStep>
      </div>

      <details className="clinical-disclosure">
        <summary>Manual refinement</summary>
        <p>
          Use <strong>Refine with box</strong> in the toolbar to prompt any visible structure on
          an axial slice. A prompt mask is not anomaly evidence.
        </p>
      </details>

      <details className="clinical-disclosure">
        <summary>Model evidence and versions</summary>
        <div className="clinical-evidence-copy">
          <p>
            At the fixed 0.30 score threshold, this project detected 21 of 23 derived
            consensus nodules across 27 eligible LIDC CT series and produced 2.15 false
            candidates per scan. Eligible SeriesInstanceUIDs were absent from LUNA16's
            published 888-series corpus. This is a small research evaluation—not clinical
            validation or a cancer probability.
          </p>
          {detectorVersion && <code>{detectorVersion}</code>}
          {segmenterVersion && <code>{segmenterVersion}</code>}
        </div>
      </details>
    </div>
  );
}

function ClinicalHeader({
  series,
  imageCount,
}: {
  series: SeriesInfo;
  imageCount: number | null;
}) {
  return (
    <header className="clinical-panel-header">
      <span className="clinical-mark" aria-hidden="true">
        <svg viewBox="0 0 32 32" role="presentation">
          <path d="M7 12V7h5M20 7h5v5M25 20v5h-5M12 25H7v-5" />
          <path d="M8 16h5l2-4 3 8 2-4h4" />
        </svg>
      </span>
      <div className="clinical-panel-title">
        <strong>Clinique Amina</strong>
        <span>Imaging review workspace</span>
      </div>
      <div className="clinical-study-chip">
        <span>{series.modality}</span>
        <span>{series.body_part || 'study'}</span>
        {imageCount != null && <span>{imageCount} images</span>}
      </div>
    </header>
  );
}

function ReviewStep({
  number,
  label,
  title,
  tone,
  last = false,
  children,
}: {
  number: string;
  label: string;
  title: string;
  tone: StepTone;
  last?: boolean;
  children: React.ReactNode;
}) {
  return (
    <section className={`clinical-step clinical-step-${tone}${last ? ' clinical-step-last' : ''}`}>
      <div className="clinical-step-rail" aria-hidden="true">
        <span>{tone === 'complete' ? '✓' : number}</span>
      </div>
      <div className="clinical-step-body">
        <div className="clinical-step-heading">
          <span>{label}</span>
          <h3>{title}</h3>
        </div>
        <div className="clinical-step-content">{children}</div>
      </div>
    </section>
  );
}

function CandidateRow({
  candidate,
  index,
  selected,
  disabled,
  outlined,
  assessment,
  onSelect,
}: {
  candidate: LungNoduleCandidate;
  index: number;
  selected: boolean;
  disabled: boolean;
  outlined: boolean;
  assessment?: 'supported' | 'dismissed' | 'uncertain';
  onSelect: () => void;
}) {
  const [width, height, depth] = candidate.size_whd_mm;
  return (
    <button
      type="button"
      onClick={onSelect}
      disabled={disabled}
      aria-pressed={selected}
      data-cy={`lung-nodule-candidate-${index + 1}`}
      className={`clinical-candidate${selected ? ' clinical-candidate-selected' : ''}`}
    >
      <span className="clinical-candidate-rank">{String(index + 1).padStart(2, '0')}</span>
      <span className="clinical-candidate-copy">
        <strong>Candidate {index + 1}</strong>
        <span className="clinical-mono">
          {width.toFixed(1)} × {height.toFixed(1)} × {depth.toFixed(1)} mm
        </span>
      </span>
      <span className="clinical-candidate-score">
        <strong>{assessment ? assessmentLabel(assessment) : candidate.score.toFixed(3)}</strong>
        <span>{assessment ? (outlined ? 'outlined · reviewed' : 'reviewed') : outlined ? 'outlined' : 'rank score'}</span>
      </span>
      <span className="clinical-candidate-arrow" aria-hidden="true">→</span>
    </button>
  );
}

function DispositionButton({
  label,
  value,
  current,
  onSelect,
}: {
  label: string;
  value: 'supported' | 'dismissed' | 'uncertain';
  current?: 'supported' | 'dismissed' | 'uncertain';
  onSelect: (value: 'supported' | 'dismissed' | 'uncertain') => void;
}) {
  return (
    <button
      type="button"
      aria-pressed={current === value}
      className={current === value ? 'clinical-disposition-selected' : ''}
      onClick={() => onSelect(value)}
    >
      {label}
    </button>
  );
}

function assessmentLabel(value: 'supported' | 'dismissed' | 'uncertain'): string {
  if (value === 'dismissed') {
    return 'dismissed';
  }
  return value;
}

function InlineState({
  tone,
  title,
  children,
}: {
  tone: StepTone;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className={`clinical-inline-state clinical-inline-${tone}`}>
      {tone === 'active' && <Icons.LoadingOHIFMark className="clinical-state-spinner" />}
      <div>
        <strong>{title}</strong>
        <span>{children}</span>
      </div>
    </div>
  );
}

function Metric({ value, label }: { value: string; label: string }) {
  return (
    <div className="clinical-metric">
      <strong className="clinical-mono">{value}</strong>
      <span>{label}</span>
    </div>
  );
}

function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.max(0, Math.round(seconds))}s`;
  }
  const minutes = Math.floor(seconds / 60);
  return `${minutes}m ${Math.round(seconds % 60)}s`;
}

function cleanError(message: string): string {
  const firstLine = message.split('\n')[0];
  return firstLine.replace(/^Error:\s*/i, '').slice(0, 240);
}

function persistenceCopy(status: 'published' | 'disabled' | 'failed' | 'not-created'): string {
  switch (status) {
    case 'published':
      return 'A DICOM SEG was saved to Orthanc.';
    case 'disabled':
      return 'The DICOM SEG currently lives only in the inference session.';
    case 'failed':
      return 'The DICOM SEG was created remotely, but storage to Orthanc failed.';
    default:
      return 'The fallback result was not written as a volumetric DICOM SEG.';
  }
}
