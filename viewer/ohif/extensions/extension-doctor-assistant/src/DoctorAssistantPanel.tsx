import React, { useEffect, useState, useCallback, useRef } from 'react';
import { useSystem } from '@ohif/core';
import { Button, Badge, PanelSection, Icons } from '@ohif/ui-next';
import {
  findSeriesByDicomUid,
  listSeriesAnalyses,
  submitSeriesAnalysis,
  getAnalysisResult,
  getRuntimeHealth,
  detectLungNodules,
  getLatestLungNoduleDetection,
  saveDicomSegArtifactToLocalOrthanc,
  NotFoundError,
  SeriesInfo,
  AnalysisResult,
  FindingInfo,
  RecommendationInfo,
  LungNoduleCandidate,
} from './apiClient';
import ClinicalReviewRail from './ClinicalReviewRail';
import {
  outlineLungNoduleCandidate,
  selectLungNoduleCandidate,
} from './lungNoduleCandidateEvents';
import { useReviewWorkflowStore } from './useReviewWorkflowStore';
import { resolveSourceSeriesInstanceUid } from './sourceSeries';

const POLL_INTERVAL_MS = 1500;

/**
 * Read the source image series from OHIF's active display sets. A study can put a
 * DICOM SEG first (as LIDC-IDRI-0117 does), so using `displaySets[0].SeriesInstanceUID`
 * directly would query the API for the reader SEG and incorrectly report that the CT
 * was not imported.
 */
function useActiveSeriesInstanceUid(): string | null {
  const { servicesManager } = useSystem();
  const { displaySetService, viewportGridService } = servicesManager.services;
  const [seriesInstanceUid, setSeriesInstanceUid] = useState<string | null>(null);

  const readActiveSeries = useCallback(() => {
    const viewportState = viewportGridService.getState?.();
    const activeViewport = viewportState?.viewports?.get(viewportState.activeViewportId);
    const viewportDisplaySets = (activeViewport?.displaySetInstanceUIDs || [])
      .map((uid: string) => displaySetService.getDisplaySetByUID(uid))
      .filter(Boolean);
    const displaySets = viewportDisplaySets.length
      ? viewportDisplaySets
      : displaySetService.getActiveDisplaySets();
    setSeriesInstanceUid(
      resolveSourceSeriesInstanceUid(displaySets, uid => displaySetService.getDisplaySetByUID(uid))
    );
  }, [displaySetService, viewportGridService]);

  useEffect(() => {
    readActiveSeries();
    const subscriptions = [
      displaySetService.subscribe(displaySetService.EVENTS.DISPLAY_SETS_ADDED, readActiveSeries),
      displaySetService.subscribe(displaySetService.EVENTS.DISPLAY_SETS_CHANGED, readActiveSeries),
      displaySetService.subscribe(displaySetService.EVENTS.DISPLAY_SETS_REMOVED, readActiveSeries),
      viewportGridService.subscribe(
        viewportGridService.EVENTS.ACTIVE_VIEWPORT_ID_CHANGED,
        readActiveSeries
      ),
      viewportGridService.subscribe(
        viewportGridService.EVENTS.GRID_STATE_CHANGED,
        readActiveSeries
      ),
    ];
    return () => subscriptions.forEach(subscription => subscription.unsubscribe());
  }, [displaySetService, viewportGridService, readActiveSeries]);

  return seriesInstanceUid;
}

function useReferenceSegCount(seriesInstanceUid: string | null): number {
  const { servicesManager } = useSystem();
  const { displaySetService } = servicesManager.services;
  const [count, setCount] = useState(0);

  const readCount = useCallback(() => {
    if (!seriesInstanceUid) {
      setCount(0);
      return;
    }
    const segmentations = displaySetService.getActiveDisplaySets().filter((displaySet: any) => {
      if (String(displaySet.Modality || '').toUpperCase() !== 'SEG' || displaySet.madeInClient) {
        return false;
      }
      if (displaySet.referencedSeriesInstanceUID) {
        return displaySet.referencedSeriesInstanceUID === seriesInstanceUid;
      }
      if (displaySet.referencedDisplaySetInstanceUID) {
        return (
          displaySetService.getDisplaySetByUID(displaySet.referencedDisplaySetInstanceUID)
            ?.SeriesInstanceUID === seriesInstanceUid
        );
      }
      return true;
    });
    setCount(segmentations.length);
  }, [displaySetService, seriesInstanceUid]);

  useEffect(() => {
    readCount();
    const subscriptions = [
      displaySetService.subscribe(displaySetService.EVENTS.DISPLAY_SETS_ADDED, readCount),
      displaySetService.subscribe(displaySetService.EVENTS.DISPLAY_SETS_CHANGED, readCount),
      displaySetService.subscribe(displaySetService.EVENTS.DISPLAY_SETS_REMOVED, readCount),
    ];
    return () => subscriptions.forEach(subscription => subscription.unsubscribe());
  }, [displaySetService, readCount]);

  return count;
}

type PanelState =
  | { phase: 'loading' }
  | { phase: 'not-imported' }
  | {
      phase: 'interactive-ready';
      series: SeriesInfo;
      modelVersion: string | null;
      detectorConfigured: boolean;
      detectorVersion: string | null;
    }
  | { phase: 'ready'; series: SeriesInfo; latest: AnalysisResult | null }
  | { phase: 'running'; series: SeriesInfo; analysisId: string }
  | { phase: 'error'; message: string };

export default function DoctorAssistantPanel() {
  const seriesInstanceUid = useActiveSeriesInstanceUid();
  const referenceSegCount = useReferenceSegCount(seriesInstanceUid);
  const { servicesManager } = useSystem();
  const [state, setState] = useState<PanelState>({ phase: 'loading' });
  const workflowSeriesId = state.phase === 'interactive-ready' ? state.series.id : null;
  const [workflow, updateWorkflow] = useReviewWorkflowStore(workflowSeriesId);
  const [clock, setClock] = useState(Date.now());
  const candidateScanGeneration = useRef(0);
  const recoveredDetectorSeriesIds = useRef(new Set<string>());

  useEffect(() => {
    let cancelled = false;
    candidateScanGeneration.current += 1;
    if (!seriesInstanceUid) {
      setState({ phase: 'loading' });
      return;
    }
    setState({ phase: 'loading' });
    (async () => {
      try {
        const series = await findSeriesByDicomUid(seriesInstanceUid);
        // `/health` is a Colab-inference capability probe, not a prerequisite for
        // normal API analysis. Older/full API deployments legitimately omit it; only
        // switch panel modes when the endpoint explicitly identifies Colab mode.
        let runtime = null;
        try {
          runtime = await getRuntimeHealth();
        } catch (error) {
          if (!(error instanceof NotFoundError)) {
            throw error;
          }
        }
        if (runtime?.mode === 'colab-inference-only') {
          if (!cancelled) {
            setState({
              phase: 'interactive-ready',
              series,
              modelVersion: runtime.model_version,
              detectorConfigured: Boolean(runtime.lung_nodule_detector_configured),
              detectorVersion: runtime.lung_nodule_detector_version ?? null,
            });
          }
          return;
        }
        const analyses = await listSeriesAnalyses(series.id);
        if (cancelled) {
          return;
        }
        if (!analyses.length) {
          setState({ phase: 'ready', series, latest: null });
          return;
        }
        const latest = await getAnalysisResult(analyses[0].id);
        if (!cancelled) {
          setState({ phase: 'ready', series, latest });
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        if (error instanceof NotFoundError) {
          setState({ phase: 'not-imported' });
        } else {
          setState({ phase: 'error', message: String(error) });
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [seriesInstanceUid]);

  useEffect(() => {
    if (state.phase !== 'running') {
      return;
    }
    const { series, analysisId } = state;
    const interval = setInterval(async () => {
      try {
        const result = await getAnalysisResult(analysisId);
        if (result.status === 'complete' || result.status === 'failed') {
          clearInterval(interval);
          setState({ phase: 'ready', series, latest: result });
        }
      } catch (error) {
        clearInterval(interval);
        setState({ phase: 'error', message: String(error) });
      }
    }, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [state]);

  useEffect(() => {
    if (workflow.candidateScan.phase !== 'running') {
      return;
    }
    setClock(Date.now());
    const interval = window.setInterval(() => setClock(Date.now()), 1000);
    return () => window.clearInterval(interval);
  }, [workflow.candidateScan.phase]);

  useEffect(() => {
    if (
      state.phase !== 'interactive-ready' ||
      workflow.candidateScan.phase !== 'idle' ||
      recoveredDetectorSeriesIds.current.has(state.series.id)
    ) {
      return;
    }
    const seriesId = state.series.id;
    const requestGeneration = candidateScanGeneration.current + 1;
    candidateScanGeneration.current = requestGeneration;
    recoveredDetectorSeriesIds.current.add(seriesId);
    let cancelled = false;
    getLatestLungNoduleDetection(seriesId)
      .then(result => {
        if (
          !cancelled &&
          candidateScanGeneration.current === requestGeneration &&
          result.series_id === seriesId
        ) {
          updateWorkflow({
            candidateScan: {
              phase: 'complete',
              result,
              elapsedSeconds: Math.max(0, result.elapsed_ms / 1000),
            },
          });
        }
      })
      .catch(error => {
        if (!(error instanceof NotFoundError)) {
          console.warn('Could not recover the latest detector run', error);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [state, workflow.candidateScan.phase, updateWorkflow]);

  const runAnalysis = useCallback(async () => {
    if (state.phase !== 'ready') {
      return;
    }
    const { series } = state;
    try {
      const submitted = await submitSeriesAnalysis(series.id);
      setState({ phase: 'running', series, analysisId: submitted.analysis_id });
    } catch (error) {
      setState({ phase: 'error', message: String(error) });
    }
  }, [state]);

  const scanForNoduleCandidates = useCallback(async (force = false) => {
    if (
      state.phase !== 'interactive-ready' ||
      !state.detectorConfigured ||
      workflow.review.phase === 'running' ||
      workflow.artifactSave.phase === 'saving'
    ) {
      return;
    }
    const requestedSeriesId = state.series.id;
    const requestGeneration = candidateScanGeneration.current + 1;
    candidateScanGeneration.current = requestGeneration;
    recoveredDetectorSeriesIds.current.add(requestedSeriesId);
    const startedAt = Date.now();
    // Candidate ranks and IDs belong to one detector run. A forced refresh must not
    // leave an outline or disposition from the previous run attached by array index.
    updateWorkflow({
      candidateScan: { phase: 'running', startedAt },
      selectedCandidateIndex: null,
      review: { phase: 'idle' },
      candidateReviews: {},
      candidateAssessments: {},
      artifactSave: { phase: 'idle' },
    });
    try {
      const result = await detectLungNodules(state.series.id, { force });
      if (
        candidateScanGeneration.current === requestGeneration &&
        result.series_id === requestedSeriesId
      ) {
        updateWorkflow({
          candidateScan: {
            phase: 'complete',
            result,
            elapsedSeconds:
              result.cache_status === 'hit'
                ? Math.max(0, result.elapsed_ms / 1000)
                : Math.max(0, (Date.now() - startedAt) / 1000),
          },
        });
      }
    } catch (error) {
      if (candidateScanGeneration.current === requestGeneration) {
        updateWorkflow({ candidateScan: { phase: 'error', message: String(error) } });
      }
    }
  }, [state, workflow.review.phase, workflow.artifactSave.phase, updateWorkflow]);

  const inspectCandidate = useCallback(
    (candidate: LungNoduleCandidate, candidateIndex: number) => {
      if (state.phase !== 'interactive-ready') {
        return;
      }
      updateWorkflow(current => ({
        selectedCandidateIndex: candidateIndex,
        review: current.candidateReviews[candidateIndex]
          ? { phase: 'complete', ...current.candidateReviews[candidateIndex] }
          : current.selectedCandidateIndex === candidateIndex
            ? current.review
            : { phase: 'idle' },
        artifactSave:
          current.selectedCandidateIndex === candidateIndex
            ? current.artifactSave
            : { phase: 'idle' },
      }));
      selectLungNoduleCandidate({
        seriesId: state.series.id,
        candidateIndex,
        candidate,
      });
    },
    [state, updateWorkflow]
  );

  const outlineCandidate = useCallback(
    (candidate: LungNoduleCandidate, candidateIndex: number) => {
      if (state.phase !== 'interactive-ready') {
        return;
      }
      updateWorkflow({ selectedCandidateIndex: candidateIndex });
      outlineLungNoduleCandidate({
        seriesId: state.series.id,
        candidateIndex,
        candidate,
      });
    },
    [state, updateWorkflow]
  );

  const openComparison = useCallback(() => {
    servicesManager.services.panelService.activatePanel(
      '@ohif/extension-cornerstone.panelModule.panelSegmentation',
      true
    );
  }, [servicesManager]);

  const saveOutlineToCase = useCallback(async () => {
    if (workflow.review.phase !== 'complete' || !workflow.review.artifact) {
      return;
    }
    const artifact = workflow.review.artifact;
    updateWorkflow({ artifactSave: { phase: 'saving' } });
    try {
      await saveDicomSegArtifactToLocalOrthanc(artifact.downloadPath);
      updateWorkflow(current =>
        current.review.phase === 'complete' &&
        current.review.artifact?.sopInstanceUid === artifact.sopInstanceUid
          ? { artifactSave: { phase: 'saved' } }
          : {}
      );
    } catch (error) {
      updateWorkflow(current =>
        current.review.phase === 'complete' &&
        current.review.artifact?.sopInstanceUid === artifact.sopInstanceUid
          ? {
              artifactSave: {
                phase: 'error',
                message: error instanceof Error ? error.message : String(error),
              },
            }
          : {}
      );
    }
  }, [workflow.review, updateWorkflow]);

  const setCandidateAssessment = useCallback(
    (assessment: 'supported' | 'dismissed' | 'uncertain') => {
      if (workflow.selectedCandidateIndex == null) {
        return;
      }
      updateWorkflow(current => ({
        candidateAssessments: {
          ...current.candidateAssessments,
          [workflow.selectedCandidateIndex!]: assessment,
        },
      }));
    },
    [workflow.selectedCandidateIndex, updateWorkflow]
  );

  if (state.phase === 'loading') {
    return (
      <div className="clinical-review-shell flex flex-col gap-3">
        <PanelBrandHeader />
        <div className="text-muted-foreground flex items-center gap-2 p-1 text-sm">
          <Icons.LoadingOHIFMark className="h-5 w-5" />
          Loading scan details…
        </div>
      </div>
    );
  }

  if (state.phase === 'not-imported') {
    return (
      <div className="clinical-review-shell flex flex-col gap-3">
        <PanelBrandHeader />
        <InfoCard
          icon={<Icons.NotificationInfo className="h-5 w-5" />}
          headline="This scan hasn't been set up for AI review yet."
        />
      </div>
    );
  }

  if (state.phase === 'error') {
    return (
      <div className="clinical-review-shell flex flex-col gap-3">
        <PanelBrandHeader />
        <div className="bg-error-bg border-error-border text-error-text rounded-md border p-3">
          <div className="text-sm font-semibold">Something went wrong while checking this scan</div>
          <div className="text-error-text/80 mt-1 font-mono text-xs">{state.message}</div>
        </div>
      </div>
    );
  }

  if (state.phase === 'interactive-ready') {
    const runningSeconds =
      workflow.candidateScan.phase === 'running'
        ? Math.max(0, (clock - workflow.candidateScan.startedAt) / 1000)
        : 0;
    return (
      <ClinicalReviewRail
        series={state.series}
        detectorConfigured={state.detectorConfigured}
        detectorVersion={state.detectorVersion}
        segmenterVersion={state.modelVersion}
        referenceSegCount={referenceSegCount}
        workflow={workflow}
        runningSeconds={runningSeconds}
        onDetect={() => scanForNoduleCandidates(false)}
        onRerun={() => scanForNoduleCandidates(true)}
        onSelectCandidate={inspectCandidate}
        onOutlineCandidate={outlineCandidate}
        onOpenComparison={openComparison}
        onSaveOutline={saveOutlineToCase}
        onSetAssessment={setCandidateAssessment}
      />
    );
  }

  const series = state.series;

  return (
    <div className="clinical-review-shell flex flex-col gap-3">
      <PanelBrandHeader />
      <div className="text-muted-foreground text-xs">
        {series.modality} / {series.body_part}
      </div>

      {!series.analysis_eligible && (
        <div className="bg-warning-bg border-warning-border text-warning-text rounded-md border p-3">
          <div className="text-sm font-semibold">AI review isn't available for this scan</div>
          {series.ineligible_reason && (
            <div className="mt-1 text-xs opacity-80">{series.ineligible_reason}</div>
          )}
        </div>
      )}

      {state.phase === 'running' && (
        <div className="bg-info-bg border-info-border text-info-text flex items-center gap-2 rounded-md border p-3">
          <Icons.LoadingOHIFMark className="h-5 w-5" />
          <div>
            <div className="text-sm font-semibold">Analyzing this scan…</div>
            <div className="text-xs opacity-80">Usually takes under a minute.</div>
          </div>
        </div>
      )}

      {state.phase === 'ready' && series.analysis_eligible && !state.latest && (
        <div className="border-border bg-card shadow-brand rounded-xl border p-4">
          <div className="flex flex-col gap-3">
            <h3 className="text-foreground text-[15px] font-semibold">Run AI Analysis</h3>
            <p className="text-muted-foreground text-sm">
              Our AI will scan this image for the findings it's trained to detect.
            </p>
            <Button
              variant="default"
              onClick={runAnalysis}
              className="bg-primary hover:bg-primary-hover shadow-brand-sm w-full"
            >
              Run AI Analysis
            </Button>
          </div>
        </div>
      )}

      {state.phase === 'ready' && state.latest && (
        <AnalysisSummary
          result={state.latest}
          onRetry={runAnalysis}
        />
      )}
    </div>
  );
}

function PanelBrandHeader() {
  return (
    <div className="clinical-panel-header">
      <span
        className="clinical-mark"
        aria-hidden="true"
      >
        <svg
          viewBox="0 0 32 32"
          role="presentation"
        >
          <path d="M7 12V7h5M20 7h5v5M25 20v5h-5M12 25H7v-5" />
          <path d="M8 16h5l2-4 3 8 2-4h4" />
        </svg>
      </span>
      <div className="clinical-panel-title">
        <strong>Clinique Amina</strong>
        <span>Imaging review workspace</span>
      </div>
    </div>
  );
}

function InfoCard({
  icon,
  headline,
  detail,
}: {
  icon: React.ReactNode;
  headline: string;
  detail?: string;
}) {
  return (
    <div className="bg-info-bg border-info-border text-info-text shadow-brand-sm flex items-start gap-2 rounded-xl border p-3">
      {icon}
      <div>
        <div className="text-sm font-semibold">{headline}</div>
        {detail && <div className="mt-1 text-xs opacity-80">{detail}</div>}
      </div>
    </div>
  );
}

type UrgencyTone = 'success' | 'warning' | 'error' | 'neutral';

/**
 * Backend casing is inconsistent between the two urgency fields (triage_urgency
 * serializes via .name — likely upper-case — while per-recommendation urgency
 * serializes via .name.lower() — see reporting/guidelines.py). Normalize defensively
 * and fall back to a neutral tone for anything unrecognized rather than throwing.
 */
function urgencyTone(urgency: string | null | undefined): UrgencyTone {
  switch ((urgency ?? '').toLowerCase()) {
    case 'routine':
      return 'success';
    case 'prompt':
      return 'warning';
    case 'urgent':
      return 'error';
    default:
      return 'neutral';
  }
}

const urgencyToneClass: Record<UrgencyTone, string> = {
  success: 'bg-success-bg text-success-text border-success-border',
  warning: 'bg-warning-bg text-warning-text border-warning-border',
  error: 'bg-error-bg text-error-text border-error-border',
  neutral: 'bg-muted text-muted-foreground border-border',
};

function UrgencyBadge({ urgency }: { urgency: string | null | undefined }) {
  const tone = urgencyTone(urgency);
  return <Badge className={urgencyToneClass[tone]}>{urgency || 'unknown'}</Badge>;
}

function AnalysisSummary({ result, onRetry }: { result: AnalysisResult; onRetry: () => void }) {
  if (result.status === 'failed') {
    return (
      <div className="bg-error-bg border-error-border text-error-text rounded-md border p-3">
        <div className="text-sm font-semibold">Analysis failed</div>
        {result.error && <div className="mt-1 text-xs opacity-80">{result.error}</div>}
        <Button
          variant="default"
          onClick={onRetry}
          className="mt-3 w-full"
        >
          Retry AI Analysis
        </Button>
      </div>
    );
  }
  if (result.status !== 'complete') {
    return (
      <div className="text-muted-foreground flex items-center gap-2 text-sm">
        <Icons.LoadingOHIFMark className="h-5 w-5" />
        Analysis {result.status}…
      </div>
    );
  }

  const present = result.findings.filter(f => f.present);

  return (
    <div className="flex flex-col gap-3">
      {/* Trust banner — transparent by design, not fine print. Mirrors the same
          show-don't-hide pattern InvestigationalUseDialog already uses elsewhere
          in this app. */}
      <InfoCard
        icon={<Icons.NotificationInfo className="h-5 w-5" />}
        headline="AI-generated analysis — experimental, not a medical diagnosis."
        detail="Always confirm with a Clinique Amina clinician before acting on these results."
      />

      {result.triage_urgency && (
        <div className="flex items-center gap-2">
          <span className="text-muted-foreground text-xs">Overall urgency:</span>
          <UrgencyBadge urgency={result.triage_urgency} />
        </div>
      )}

      {result.verification_ok === false && (
        <div className="bg-warning-bg border-warning-border text-warning-text rounded-md border p-2 text-xs">
          Some of these findings couldn't be fully verified.
        </div>
      )}

      <PanelSection defaultOpen>
        <PanelSection.Header className="text-[13px] font-semibold tracking-wide">
          Findings
        </PanelSection.Header>
        <PanelSection.Content>
          <div className="flex flex-col gap-2 p-2">
            {present.length === 0 && (
              <div className="text-muted-foreground text-sm">No positive findings.</div>
            )}
            {present.map((finding, index) => (
              <FindingCard
                key={index}
                finding={finding}
              />
            ))}
          </div>
        </PanelSection.Content>
      </PanelSection>

      {result.recommendations.length > 0 && (
        <PanelSection defaultOpen>
          <PanelSection.Header className="text-[13px] font-semibold tracking-wide">
            Recommendations
          </PanelSection.Header>
          <PanelSection.Content>
            <div className="flex flex-col gap-2 p-2">
              {result.recommendations.map((rec, index) => (
                <RecommendationCard
                  key={index}
                  recommendation={rec}
                />
              ))}
            </div>
          </PanelSection.Content>
        </PanelSection>
      )}

      {result.report_text && (
        <PanelSection defaultOpen={false}>
          <PanelSection.Header className="text-[13px] font-semibold tracking-wide">
            Full Report
          </PanelSection.Header>
          <PanelSection.Content>
            <pre className="text-muted-foreground whitespace-pre-wrap p-2 text-xs">
              {result.report_text}
            </pre>
          </PanelSection.Content>
        </PanelSection>
      )}
    </div>
  );
}

/**
 * A confident finding (probability >= 0.7) gets the brand-teal accent bar and a
 * slightly stronger shadow — a visual "this one matters" cue that plain equal-
 * weight bordered rows didn't give. Below that threshold, a neutral bar keeps
 * lower-confidence findings from competing for attention.
 */
function FindingCard({ finding }: { finding: FindingInfo }) {
  const isConfident = (finding.probability ?? 0) >= 0.7;
  return (
    <div
      className={`bg-card shadow-brand-sm flex items-start gap-2.5 rounded-lg border-l-4 p-2.5 ${
        isConfident ? 'border-l-primary' : 'border-l-border'
      }`}
    >
      <div className="min-w-0 flex-1">
        <div className="flex items-center justify-between gap-2">
          <span className="text-foreground text-sm font-medium">{finding.label}</span>
          {finding.probability != null && (
            <Badge
              variant="secondary"
              className={isConfident ? 'bg-secondary text-primary' : undefined}
            >
              {Math.round(finding.probability * 100)}%
            </Badge>
          )}
        </div>
        {(finding.location || finding.laterality) && (
          <div className="text-muted-foreground mt-0.5 text-xs">
            {[finding.laterality, finding.location].filter(Boolean).join(' · ')}
          </div>
        )}
      </div>
    </div>
  );
}

const urgencyBorderClass: Record<UrgencyTone, string> = {
  success: 'border-l-success-text',
  warning: 'border-l-warning-text',
  error: 'border-l-error-text',
  neutral: 'border-l-border',
};

function RecommendationCard({ recommendation }: { recommendation: RecommendationInfo }) {
  const tone = urgencyTone(recommendation.urgency);
  return (
    <div
      className={`bg-card shadow-brand-sm flex items-start justify-between gap-2 rounded-lg border-l-4 p-2.5 ${urgencyBorderClass[tone]}`}
    >
      <span className="text-foreground text-sm">{recommendation.text}</span>
      <UrgencyBadge urgency={recommendation.urgency} />
    </div>
  );
}
