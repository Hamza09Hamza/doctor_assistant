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
  NotFoundError,
  SeriesInfo,
  AnalysisResult,
  FindingInfo,
  RecommendationInfo,
  LungNoduleDetectionResult,
} from './apiClient';
import { selectLungNoduleCandidate } from './lungNoduleCandidateEvents';

const POLL_INTERVAL_MS = 1500;

type SeriesDisplaySet = {
  SeriesInstanceUID?: string;
  Modality?: string;
  isOverlayDisplaySet?: boolean;
  referencedSeriesInstanceUID?: string;
  referencedDisplaySetInstanceUID?: string;
};

const OVERLAY_MODALITIES = new Set(['SEG', 'RTSTRUCT', 'SR', 'PR', 'PMAP']);

/** Resolve an overlay such as DICOM SEG back to the image series it annotates. */
export function resolveSourceSeriesInstanceUid(
  displaySets: SeriesDisplaySet[] | undefined,
  getDisplaySetByUID: (uid: string) => SeriesDisplaySet | undefined
): string | null {
  const first = displaySets?.[0];
  if (!first) {
    return null;
  }

  if (first.referencedSeriesInstanceUID) {
    return first.referencedSeriesInstanceUID;
  }
  if (first.referencedDisplaySetInstanceUID) {
    const referenced = getDisplaySetByUID(first.referencedDisplaySetInstanceUID);
    if (referenced?.SeriesInstanceUID) {
      return referenced.SeriesInstanceUID;
    }
  }

  const firstIsOverlay =
    first.isOverlayDisplaySet || OVERLAY_MODALITIES.has(String(first.Modality || '').toUpperCase());
  if (!firstIsOverlay && first.SeriesInstanceUID) {
    return first.SeriesInstanceUID;
  }

  const sourceImages = displaySets.find(displaySet => {
    const modality = String(displaySet.Modality || '').toUpperCase();
    return (
      displaySet.SeriesInstanceUID &&
      !displaySet.isOverlayDisplaySet &&
      !OVERLAY_MODALITIES.has(modality)
    );
  });
  return sourceImages?.SeriesInstanceUID ?? null;
}

/**
 * Read the source image series from OHIF's active display sets. A study can put a
 * DICOM SEG first (as LIDC-IDRI-0117 does), so using `displaySets[0].SeriesInstanceUID`
 * directly would query the API for the reader SEG and incorrectly report that the CT
 * was not imported.
 */
function useActiveSeriesInstanceUid(): string | null {
  const { servicesManager } = useSystem();
  const { displaySetService } = servicesManager.services;
  const [seriesInstanceUid, setSeriesInstanceUid] = useState<string | null>(null);

  const readActiveSeries = useCallback(() => {
    const displaySets = displaySetService.getActiveDisplaySets();
    setSeriesInstanceUid(
      resolveSourceSeriesInstanceUid(displaySets, uid => displaySetService.getDisplaySetByUID(uid))
    );
  }, [displaySetService]);

  useEffect(() => {
    readActiveSeries();
    const subscription = displaySetService.subscribe(
      displaySetService.EVENTS.DISPLAY_SETS_ADDED,
      readActiveSeries
    );
    return () => subscription.unsubscribe();
  }, [displaySetService, readActiveSeries]);

  return seriesInstanceUid;
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

type CandidateScanState =
  | { phase: 'idle' }
  | { phase: 'running' }
  | { phase: 'complete'; result: LungNoduleDetectionResult }
  | { phase: 'error'; message: string };

export default function DoctorAssistantPanel() {
  const seriesInstanceUid = useActiveSeriesInstanceUid();
  const [state, setState] = useState<PanelState>({ phase: 'loading' });
  const [candidateScan, setCandidateScan] = useState<CandidateScanState>({ phase: 'idle' });
  const candidateScanGeneration = useRef(0);

  useEffect(() => {
    let cancelled = false;
    if (!seriesInstanceUid) {
      setState({ phase: 'loading' });
      return;
    }
    candidateScanGeneration.current += 1;
    setCandidateScan({ phase: 'idle' });
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

  const scanForNoduleCandidates = useCallback(async () => {
    if (state.phase !== 'interactive-ready' || !state.detectorConfigured) {
      return;
    }
    const requestedSeriesId = state.series.id;
    const requestGeneration = candidateScanGeneration.current + 1;
    candidateScanGeneration.current = requestGeneration;
    setCandidateScan({ phase: 'running' });
    try {
      const result = await detectLungNodules(state.series.id);
      if (
        candidateScanGeneration.current === requestGeneration &&
        result.series_id === requestedSeriesId
      ) {
        setCandidateScan({ phase: 'complete', result });
      }
    } catch (error) {
      if (candidateScanGeneration.current === requestGeneration) {
        setCandidateScan({ phase: 'error', message: String(error) });
      }
    }
  }, [state]);

  if (state.phase === 'loading') {
    return (
      <div className="flex flex-col gap-3 p-3">
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
      <div className="flex flex-col gap-3 p-3">
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
      <div className="flex flex-col gap-3 p-3">
        <PanelBrandHeader />
        <div className="bg-error-bg border-error-border text-error-text rounded-md border p-3">
          <div className="text-sm font-semibold">Something went wrong while checking this scan</div>
          <div className="text-error-text/80 mt-1 font-mono text-xs">{state.message}</div>
        </div>
      </div>
    );
  }

  if (state.phase === 'interactive-ready') {
    return (
      <div className="flex flex-col gap-3 p-3">
        <PanelBrandHeader />
        <div className="text-muted-foreground text-xs">
          {state.series.modality} / {state.series.body_part}
        </div>
        <InfoCard
          icon={<Icons.NotificationInfo className="h-5 w-5" />}
          headline="MedSAM2 structure segmentation is ready"
          detail="Choose Segment structure (tight box), then box one visible structure on an axial slice. This outlines the prompt through the volume; it does not determine whether the structure is abnormal."
        />
        {state.detectorConfigured && (
          <div className="border-border bg-card shadow-brand rounded-xl border p-3">
            <div className="font-serif text-foreground text-sm font-semibold">
              Automatic lung-nodule shortlist
            </div>
            <p className="text-muted-foreground mt-1 text-xs leading-relaxed">
              Scans the complete chest CT for candidates. Expect false marks: this detector
              found 21 of 23 consensus nodules across 27 eligible LIDC scans whose CT
              SeriesInstanceUIDs were absent from LUNA16's published 888-series corpus,
              with 2.15 false candidates per scan at its fixed threshold.
            </p>

            {candidateScan.phase === 'idle' && (
              <Button
                variant="default"
                onClick={scanForNoduleCandidates}
                data-cy="scan-lung-nodule-candidates"
                className="mt-3 w-full"
              >
                Scan for nodule candidates
              </Button>
            )}

            {candidateScan.phase === 'running' && (
              <div
                className="bg-info-bg border-info-border text-info-text mt-3 flex items-start gap-2 rounded-md border p-2.5"
                data-cy="lung-nodule-scan-running"
              >
                <Icons.LoadingOHIFMark className="mt-0.5 h-4 w-4 shrink-0" />
                <div>
                  <div className="text-xs font-semibold">Scanning the complete CT…</div>
                  <div className="mt-0.5 text-[11px] opacity-80">
                    The Colab worker is locked until candidate detection finishes.
                  </div>
                </div>
              </div>
            )}

            {candidateScan.phase === 'error' && (
              <div className="bg-error-bg border-error-border text-error-text mt-3 rounded-md border p-2.5">
                <div className="text-xs font-semibold">Candidate scan failed</div>
                <div className="mt-1 font-mono text-[10px] opacity-80">
                  {candidateScan.message}
                </div>
                <Button
                  variant="secondary"
                  onClick={scanForNoduleCandidates}
                  className="mt-2 w-full"
                >
                  Retry candidate scan
                </Button>
              </div>
            )}

            {candidateScan.phase === 'complete' && (
              <CandidateShortlist
                result={candidateScan.result}
                onSelect={(candidate, candidateIndex) =>
                  selectLungNoduleCandidate({
                    seriesId: state.series.id,
                    candidateIndex,
                    candidate,
                  })
                }
                onRescan={scanForNoduleCandidates}
              />
            )}
          </div>
        )}
        {(state.modelVersion || state.detectorVersion) && (
          <div className="text-muted-foreground flex flex-col gap-0.5 font-mono text-[10px]">
            {state.detectorVersion && <span>{state.detectorVersion}</span>}
            {state.modelVersion && <span>{state.modelVersion}</span>}
          </div>
        )}
      </div>
    );
  }

  const series = state.series;

  return (
    <div className="flex flex-col gap-3 p-3">
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
            <h3 className="font-serif text-foreground text-[15px] font-semibold">
              Run AI Analysis
            </h3>
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

function CandidateShortlist({
  result,
  onSelect,
  onRescan,
}: {
  result: LungNoduleDetectionResult;
  onSelect: (candidate: LungNoduleDetectionResult['detections'][number], index: number) => void;
  onRescan: () => void;
}) {
  if (!result.detections.length) {
    return (
      <div className="mt-3">
        <div className="bg-muted border-border text-muted-foreground rounded-md border p-2.5 text-xs">
          No candidate crossed the fixed score threshold. This does not prove the scan is clear.
        </div>
        <Button
          variant="secondary"
          onClick={onRescan}
          className="mt-2 w-full"
        >
          Scan again
        </Button>
      </div>
    );
  }

  return (
    <div className="mt-3 flex flex-col gap-2" data-cy="lung-nodule-candidate-list">
      <div className="text-muted-foreground flex items-center justify-between text-[11px]">
        <span>
          {result.detections.length} candidate{result.detections.length === 1 ? '' : 's'} at score ≥{' '}
          {result.min_score.toFixed(2)}
        </span>
        <button
          type="button"
          onClick={onRescan}
          className="text-primary hover:text-primary-hover underline underline-offset-2"
        >
          Rescan
        </button>
      </div>
      {result.detections.map((candidate, index) => {
        const largestEdgeMm = Math.max(...candidate.size_whd_mm);
        return (
          <button
            key={`${candidate.seed_sop_instance_uid}-${index}`}
            type="button"
            data-cy={`lung-nodule-candidate-${index + 1}`}
            onClick={() => onSelect(candidate, index)}
            className="border-border bg-background hover:border-primary/70 hover:bg-primary/5 focus-visible:ring-primary group flex w-full items-center gap-3 rounded-lg border p-2.5 text-left transition-colors focus-visible:ring-2 focus-visible:outline-none"
          >
            <span className="border-accent text-accent flex h-7 w-7 shrink-0 items-center justify-center rounded-full border font-mono text-[11px] font-semibold">
              {index + 1}
            </span>
            <span className="min-w-0 flex-1">
              <span className="text-foreground block text-xs font-semibold">
                Review and outline candidate
              </span>
              <span className="text-muted-foreground mt-0.5 block text-[10.5px]">
                uncalibrated score {candidate.score.toFixed(3)} · box {largestEdgeMm.toFixed(1)} mm
              </span>
            </span>
            <span className="text-primary text-base transition-transform group-hover:translate-x-0.5">›</span>
          </button>
        );
      })}
      <div className="text-warning-text text-[10.5px] leading-relaxed">
        Candidates are places to inspect, not confirmed nodules. Selecting one jumps to its
        slice and asks MedSAM2 to outline it.
      </div>
    </div>
  );
}

/**
 * Brand strip at the top of every panel state — the findings panel is the one
 * surface a patient spends the most time looking at, so it carries the Clinique
 * Amina identity (serif wordmark, gold hairline) rather than staying purely
 * functional. `font-serif` resolves to Playfair Display (tailwind.config.js);
 * the gold hairline reuses --accent rather than a hardcoded color so it moves
 * with the palette if the brand tokens are retuned later.
 */
function PanelBrandHeader() {
  return (
    <div className="border-border/70 flex items-center gap-2.5 border-b pb-3">
      {/* Monogram mark, not a second copy of plain text — a gold ring around a
       * teal "A" reads as an actual brand mark rather than a relabeled OHIF
       * header. Pure CSS/inline-SVG, no external asset needed. */}
      <span className="border-accent bg-secondary text-primary shadow-brand-sm flex h-8 w-8 shrink-0 items-center justify-center rounded-full border-2">
        <span className="font-serif text-sm font-bold">A</span>
      </span>
      <div className="flex flex-col leading-tight">
        <span className="font-serif text-foreground text-[15px] font-semibold tracking-tight">
          Clinique Amina
        </span>
        <span className="text-muted-foreground text-[10.5px] tracking-wide uppercase">
          AI Imaging Review
        </span>
      </div>
    </div>
  );
}

function InfoCard({ icon, headline, detail }: { icon: React.ReactNode; headline: string; detail?: string }) {
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
  return (
    <Badge className={urgencyToneClass[tone]}>{urgency || 'unknown'}</Badge>
  );
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
        <PanelSection.Header className="font-serif text-[13px] font-semibold tracking-wide">
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
          <PanelSection.Header className="font-serif text-[13px] font-semibold tracking-wide">
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
          <PanelSection.Header className="font-serif text-[13px] font-semibold tracking-wide">
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
