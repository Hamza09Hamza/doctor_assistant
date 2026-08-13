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

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${getApiBaseUrl()}${path}`, init);
  if (response.status === 404) {
    throw new NotFoundError(path);
  }
  if (!response.ok) {
    const contentType = response.headers.get('content-type') || '';
    let detail = response.statusText || 'request failed';
    if (contentType.includes('application/json')) {
      const body = await response.json().catch(() => null);
      detail = body?.detail || body?.message || detail;
    } else {
      const body = await response.text();
      // Never pour an ngrok/proxy HTML document into the clinical panel.
      if (body && !body.trimStart().startsWith('<')) {
        detail = body.slice(0, 240);
      }
    }
    throw new ApiError(response.status, `${response.status}: ${detail}`);
  }
  const contentType = response.headers.get('content-type') || '';
  if (!contentType.includes('application/json')) {
    throw new ApiError(
      502,
      'The AI endpoint returned a web page instead of data. Confirm the remote API URL and restart the viewer.'
    );
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

export interface RuntimeHealth {
  status: string;
  mode: string;
  ready?: boolean;
  medsam2_configured: boolean;
  medsam2_loaded?: boolean;
  model_version: string | null;
  lung_nodule_detector_configured?: boolean;
  lung_nodule_detector_loaded?: boolean;
  lung_nodule_detector_version?: string | null;
}

export function getRuntimeHealth(): Promise<RuntimeHealth> {
  return apiFetch('/health');
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

export interface LungNoduleCandidate {
  candidate_id: string;
  score: number;
  center_lps_mm: [number, number, number];
  size_whd_mm: [number, number, number];
  seed_sop_instance_uid: string;
  box_xyxy: [number, number, number, number];
}

export interface LungNoduleDetectionResult {
  series_id: string;
  model_version: string;
  run_id: string;
  cache_status: 'hit' | 'miss';
  elapsed_ms: number;
  generated_at: string;
  source_fingerprint: string;
  model_fingerprint: string;
  cache_key: string;
  min_score: number;
  source_slice_count: number;
  detections: LungNoduleCandidate[];
}

/** Automatic CT candidate generation. A candidate is not a diagnosis. */
export function detectLungNodules(
  seriesId: string,
  options: { minScore?: number; force?: boolean } = {}
): Promise<LungNoduleDetectionResult> {
  return apiFetch(`/v1/series/${seriesId}/detect-lung-nodules`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      min_score: options.minScore,
      force: options.force ?? false,
    }),
  });
}

/** Recover the newest source-bound detector run after an API or panel restart. */
export function getLatestLungNoduleDetection(
  seriesId: string
): Promise<LungNoduleDetectionResult> {
  return apiFetch(`/v1/series/${seriesId}/detect-lung-nodules/latest`);
}

export interface SegmentBoxResult {
  sop_instance_uid: string;
  mask_rle: { size: [number, number]; counts: number[] };
  model_version: string;
}

/**
 * On-demand interactive segmentation: `boxXyxy` is a box in this SOP instance's own
 * pixel coordinates (row/col, not world/patient space) — the caller converts from a
 * viewport annotation before calling this. Not a queued analysis; the response comes
 * back in this same request.
 */
export function segmentBox(
  seriesId: string,
  sopInstanceUid: string,
  boxXyxy: [number, number, number, number]
): Promise<SegmentBoxResult> {
  return apiFetch(`/v1/series/${seriesId}/segment-box`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ sop_instance_uid: sopInstanceUid, box_xyxy: boxXyxy }),
  });
}

export interface SegmentVolumeSlice {
  sop_instance_uid: string;
  mask_rle: { size: [number, number]; counts: number[] };
}

export interface SegmentVolumeResult {
  seed_sop_instance_uid: string;
  masks: SegmentVolumeSlice[];
  source_slice_count: number;
  segmented_slice_count: number;
  voxel_count: number;
  volume_ml: number;
  axial_bbox_diagonal_mm: number;
  craniocaudal_extent_mm: number;
  model_version: string;
  dicom_seg_series_instance_uid: string;
  dicom_seg_sop_instance_uid: string;
  orthanc_status: 'published' | 'disabled' | 'failed';
  warning: string | null;
  dicom_seg_artifact?: {
    download_path: string;
    sha256: string;
    byte_length: number;
    series_instance_uid: string;
    sop_instance_uid: string;
  } | null;
  reference_comparison?: ReferenceComparison | null;
}

export interface ReferenceReaderComparison {
  reader_id: string;
  matched: boolean;
  segment_number: number | null;
  segment_label: string | null;
  prompt_overlap_voxels: number;
  dice: number | null;
  reference_voxel_count: number | null;
  reference_volume_ml: number | null;
  reference_segmented_slice_count: number | null;
}

export interface ReferenceComparison {
  reference_set: string;
  matching_method: string;
  reader_count: number;
  matched_reader_count: number;
  consensus_rule: string;
  consensus_reader_threshold: number;
  consensus_available: boolean;
  consensus_dice: number | null;
  consensus_voxel_count: number | null;
  consensus_volume_ml: number | null;
  consensus_segmented_slice_count: number | null;
  readers: ReferenceReaderComparison[];
}

/** Retrieve the standards-valid DICOM SEG produced by the remote runtime. */
export async function downloadDicomSegArtifact(downloadPath: string): Promise<Blob> {
  const response = await fetch(`${getApiBaseUrl()}${downloadPath}`);
  if (!response.ok) {
    throw new ApiError(response.status, `Could not download DICOM SEG (${response.status})`);
  }
  return response.blob();
}

/** Copy a remote inference artifact into the Mac's local Orthanc through OHIF's proxy. */
export async function saveDicomSegArtifactToLocalOrthanc(
  downloadPath: string
): Promise<{ ID?: string; ParentSeries?: string; Status?: string }> {
  const artifact = await downloadDicomSegArtifact(downloadPath);
  const response = await fetch('/local-orthanc-rest/instances', {
    method: 'POST',
    headers: { 'Content-Type': 'application/dicom' },
    body: artifact,
  });
  if (!response.ok) {
    throw new ApiError(response.status, `Local Orthanc rejected the DICOM SEG (${response.status})`);
  }
  return response.json();
}

/** Propagate a box through the complete DICOM stack and persist a DICOM SEG. */
export function segmentVolume(
  seriesId: string,
  sopInstanceUid: string,
  boxXyxy: [number, number, number, number],
  voi?: { windowCenter: number; windowWidth: number },
  segmentLabel = 'AI prompted structure'
): Promise<SegmentVolumeResult> {
  return apiFetch(`/v1/series/${seriesId}/segment-volume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      sop_instance_uid: sopInstanceUid,
      box_xyxy: boxXyxy,
      // Backend-neutral on purpose: the server may use native SAM2 MLX, a converted
      // MedSAM2 checkpoint, or the official CUDA MedSAM2 runtime.
      segment_label: segmentLabel,
      publish_to_orthanc: true,
      window_center: voi?.windowCenter,
      window_width: voi?.windowWidth,
    }),
  });
}
