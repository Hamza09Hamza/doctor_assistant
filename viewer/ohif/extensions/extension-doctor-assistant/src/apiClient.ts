/**
 * Plain fetch() calls to the doctor_assistant API (see ../../../../../api/). No
 * OHIF-specific HTTP plumbing is needed for this — DataSourceModule is for feeding
 * the *viewer* from a DICOMweb server, not for a panel's own calls to an unrelated
 * backend, so a normal fetch client is the right tool here.
 *
 * `doctorAssistantApiBaseUrl` is read from `window.config` (set in
 * platform/app/public/config/doctor_assistant.js) rather than hardcoded, so the API's
 * location stays a deployment concern, not a source-code one.
 */

declare global {
  interface Window {
    config?: { doctorAssistantApiBaseUrl?: string };
  }
}

function getApiBaseUrl(): string {
  return window.config?.doctorAssistantApiBaseUrl || 'http://localhost:8000';
}

export class NotFoundError extends Error {}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, init);
  if (response.status === 404) {
    throw new NotFoundError(path);
  }
  if (!response.ok) {
    const body = await response.text();
    throw new Error(`doctor_assistant API ${path} -> ${response.status}: ${body}`);
  }
  return response.json();
}

export interface SeriesInfo {
  id: string;
  study_id: string;
  dicom_series_uid: string;
  modality: string;
  body_part: string;
  analysis_eligible: boolean;
  ineligible_reason: string | null;
}

export interface AnalysisStatus {
  id: string;
  status: 'queued' | 'running' | 'complete' | 'failed';
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface FindingInfo {
  label: string;
  canonical_label: string | null;
  probability: number | null;
  present: boolean;
  laterality: string | null;
  location: string | null;
}

export interface RecommendationInfo {
  label: string;
  text: string;
  urgency: string;
}

export interface AnalysisResult extends AnalysisStatus {
  study_id: string;
  series_id: string | null;
  report_text: string | null;
  verification_ok: boolean | null;
  triage_urgency: string | null;
  findings: FindingInfo[];
  recommendations: RecommendationInfo[];
}

export function findSeriesByDicomUid(seriesInstanceUid: string): Promise<SeriesInfo> {
  return apiFetch(`/v1/series?dicom_series_uid=${encodeURIComponent(seriesInstanceUid)}`);
}

export function listSeriesAnalyses(seriesId: string): Promise<AnalysisStatus[]> {
  return apiFetch(`/v1/series/${seriesId}/analyses`);
}

export function submitSeriesAnalysis(seriesId: string): Promise<{ analysis_id: string; status: string }> {
  return apiFetch(`/v1/series/${seriesId}/analyses`, { method: 'POST' });
}

export function getAnalysisResult(analysisId: string): Promise<AnalysisResult> {
  return apiFetch(`/v1/analyses/${analysisId}`);
}
