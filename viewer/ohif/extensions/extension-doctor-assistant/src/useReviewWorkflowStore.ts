import { useCallback, useSyncExternalStore } from 'react';

import type { LungNoduleDetectionResult } from './apiClient';
import type {
  InteractiveReviewCompletedDetail,
  InteractiveReviewContext,
  InteractiveReviewFailedDetail,
} from './lungNoduleCandidateEvents';

export type CandidateScanState =
  | { phase: 'idle' }
  | { phase: 'running'; startedAt: number }
  | { phase: 'complete'; result: LungNoduleDetectionResult; elapsedSeconds: number }
  | { phase: 'error'; message: string };

export type InteractiveReviewState =
  | { phase: 'idle' }
  | ({ phase: 'running' } & InteractiveReviewContext)
  | ({ phase: 'complete' } & InteractiveReviewCompletedDetail)
  | ({ phase: 'error' } & InteractiveReviewFailedDetail);

export interface SeriesReviewWorkflowState {
  candidateScan: CandidateScanState;
  selectedCandidateIndex: number | null;
  review: InteractiveReviewState;
  candidateReviews: Record<number, InteractiveReviewCompletedDetail>;
  candidateAssessments: Record<number, 'supported' | 'dismissed' | 'uncertain'>;
  artifactSave: { phase: 'idle' | 'saving' | 'saved' | 'error'; message?: string };
}

const EMPTY_STATE: SeriesReviewWorkflowState = {
  candidateScan: { phase: 'idle' },
  selectedCandidateIndex: null,
  review: { phase: 'idle' },
  candidateReviews: {},
  candidateAssessments: {},
  artifactSave: { phase: 'idle' },
};

const stateBySeries = new Map<string, SeriesReviewWorkflowState>();
const listenersBySeries = new Map<string, Set<() => void>>();

function getSeriesState(seriesId: string | null): SeriesReviewWorkflowState {
  if (!seriesId) {
    return EMPTY_STATE;
  }
  if (!stateBySeries.has(seriesId)) {
    stateBySeries.set(seriesId, { ...EMPTY_STATE });
  }
  return stateBySeries.get(seriesId)!;
}

/** Read-only snapshot used by non-React workflow integrations and focused unit tests. */
export function getSeriesReviewWorkflowState(seriesId: string): SeriesReviewWorkflowState {
  return getSeriesState(seriesId);
}

/** Clear module state between isolated tests. Not used by the runtime workflow. */
export function resetReviewWorkflowStoreForTests(): void {
  stateBySeries.clear();
  listenersBySeries.clear();
}

export function updateSeriesReviewWorkflow(
  seriesId: string,
  patch:
    | Partial<SeriesReviewWorkflowState>
    | ((current: SeriesReviewWorkflowState) => Partial<SeriesReviewWorkflowState>)
): void {
  const current = getSeriesState(seriesId);
  const resolvedPatch = typeof patch === 'function' ? patch(current) : patch;
  stateBySeries.set(seriesId, { ...current, ...resolvedPatch });
  listenersBySeries.get(seriesId)?.forEach(listener => listener());
}

/**
 * Keeps costly detector output and the latest outline visible if OHIF swaps panel tabs
 * or remounts the component. State is isolated by backend series id so results can never
 * bleed into another active CT.
 */
export function useReviewWorkflowStore(seriesId: string | null) {
  const subscribe = useCallback(
    (listener: () => void) => {
      if (!seriesId) {
        return () => undefined;
      }
      const listeners = listenersBySeries.get(seriesId) ?? new Set();
      listeners.add(listener);
      listenersBySeries.set(seriesId, listeners);
      return () => {
        listeners.delete(listener);
        if (!listeners.size) {
          listenersBySeries.delete(seriesId);
        }
      };
    },
    [seriesId]
  );

  const getSnapshot = useCallback(() => getSeriesState(seriesId), [seriesId]);
  const state = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  const update = useCallback(
    (
      patch:
        | Partial<SeriesReviewWorkflowState>
        | ((current: SeriesReviewWorkflowState) => Partial<SeriesReviewWorkflowState>)
    ) => {
      if (seriesId) {
        updateSeriesReviewWorkflow(seriesId, patch);
      }
    },
    [seriesId]
  );

  return [state, update] as const;
}
