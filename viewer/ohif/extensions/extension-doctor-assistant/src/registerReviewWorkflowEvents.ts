import { eventTarget } from '@cornerstonejs/core';

import {
  INTERACTIVE_REVIEW_COMPLETED,
  INTERACTIVE_REVIEW_FAILED,
  INTERACTIVE_REVIEW_STARTED,
  type InteractiveReviewCompletedDetail,
  type InteractiveReviewContext,
  type InteractiveReviewFailedDetail,
} from './lungNoduleCandidateEvents';
import { updateSeriesReviewWorkflow } from './useReviewWorkflowStore';

/** Capture review results for the whole mode, including while the panel is unmounted. */
export default function registerReviewWorkflowEvents(): { unsubscribe: () => void } {
  const handleStarted = (evt: CustomEvent<InteractiveReviewContext>) => {
    updateSeriesReviewWorkflow(evt.detail.seriesId, {
      review: { phase: 'running', ...evt.detail },
    });
  };
  const handleCompleted = (evt: CustomEvent<InteractiveReviewCompletedDetail>) => {
    updateSeriesReviewWorkflow(evt.detail.seriesId, current => ({
      review: { phase: 'complete', ...evt.detail },
      candidateReviews:
        evt.detail.candidateNumber == null
          ? current.candidateReviews
          : {
              ...current.candidateReviews,
              [evt.detail.candidateNumber - 1]: evt.detail,
            },
      artifactSave: {
        phase: evt.detail.orthancStatus === 'published' ? 'saved' : 'idle',
      },
    }));
  };
  const handleFailed = (evt: CustomEvent<InteractiveReviewFailedDetail>) => {
    if (evt.detail.seriesId) {
      updateSeriesReviewWorkflow(evt.detail.seriesId, {
        review: { phase: 'error', ...evt.detail },
      });
    }
  };

  eventTarget.addEventListener(INTERACTIVE_REVIEW_STARTED, handleStarted);
  eventTarget.addEventListener(INTERACTIVE_REVIEW_COMPLETED, handleCompleted);
  eventTarget.addEventListener(INTERACTIVE_REVIEW_FAILED, handleFailed);
  return {
    unsubscribe: () => {
      eventTarget.removeEventListener(INTERACTIVE_REVIEW_STARTED, handleStarted);
      eventTarget.removeEventListener(INTERACTIVE_REVIEW_COMPLETED, handleCompleted);
      eventTarget.removeEventListener(INTERACTIVE_REVIEW_FAILED, handleFailed);
    },
  };
}
