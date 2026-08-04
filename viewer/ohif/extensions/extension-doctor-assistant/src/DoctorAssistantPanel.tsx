import React, { useEffect, useState, useCallback } from 'react';
import { useSystem } from '@ohif/core';
import { Button, Badge, PanelSection, Icons } from '@ohif/ui-next';
import {
  findSeriesByDicomUid,
  listSeriesAnalyses,
  submitSeriesAnalysis,
  getAnalysisResult,
  NotFoundError,
  SeriesInfo,
  AnalysisResult,
  FindingInfo,
  RecommendationInfo,
} from './apiClient';

const POLL_INTERVAL_MS = 1500;

/**
 * Read the active display set's SeriesInstanceUID the same way OHIF's own
 * `usePatientInfo` hook reads patient info — `displaySetService.getActiveDisplaySets()`
 * plus a `DISPLAY_SETS_ADDED` subscription (AGENTS.md: prefer service pub/sub over
 * useEffect polling). `displaySet.SeriesInstanceUID` is confirmed as a direct field
 * (extensions/default/src/MergeDataSource/index.ts). Reacting to a same-study viewport
 * switch (not just newly-added display sets) is a real gap worth confirming against a
 * multi-series study once this is running for real — flagged rather than assumed.
 */
function useActiveSeriesInstanceUid(): string | null {
  const { servicesManager } = useSystem();
  const { displaySetService } = servicesManager.services;
  const [seriesInstanceUid, setSeriesInstanceUid] = useState<string | null>(null);

  const readActiveSeries = useCallback(() => {
    const displaySets = displaySetService.getActiveDisplaySets();
    const displaySet = displaySets?.[0];
    setSeriesInstanceUid(displaySet?.SeriesInstanceUID ?? null);
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
  | { phase: 'ready'; series: SeriesInfo; latest: AnalysisResult | null }
  | { phase: 'running'; series: SeriesInfo; analysisId: string }
  | { phase: 'error'; message: string };

export default function DoctorAssistantPanel() {
  const seriesInstanceUid = useActiveSeriesInstanceUid();
  const [state, setState] = useState<PanelState>({ phase: 'loading' });

  useEffect(() => {
    let cancelled = false;
    if (!seriesInstanceUid) {
      setState({ phase: 'loading' });
      return;
    }
    setState({ phase: 'loading' });
    (async () => {
      try {
        const series = await findSeriesByDicomUid(seriesInstanceUid);
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

      {state.phase === 'ready' && state.latest && <AnalysisSummary result={state.latest} />}
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

function AnalysisSummary({ result }: { result: AnalysisResult }) {
  if (result.status === 'failed') {
    return (
      <div className="bg-error-bg border-error-border text-error-text rounded-md border p-3">
        <div className="text-sm font-semibold">Analysis failed</div>
        {result.error && <div className="mt-1 text-xs opacity-80">{result.error}</div>}
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
