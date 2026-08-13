import {
  ApiError,
  findSeriesByDicomUid,
  getRuntimeHealth,
  saveDicomSegArtifactToLocalOrthanc,
} from './apiClient';

const fetchMock = jest.fn();

function response({
  status = 200,
  contentType = 'application/json',
  json = {},
  text = '',
  blob,
}: {
  status?: number;
  contentType?: string;
  json?: unknown;
  text?: string;
  blob?: Blob;
}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    statusText: status >= 400 ? 'request failed' : 'OK',
    headers: {
      get: (name: string) => (name.toLowerCase() === 'content-type' ? contentType : null),
    },
    json: jest.fn().mockResolvedValue(json),
    text: jest.fn().mockResolvedValue(text),
    blob: jest.fn().mockResolvedValue(blob ?? new Blob()),
  };
}

describe('doctor assistant API client', () => {
  beforeEach(() => {
    fetchMock.mockReset();
    global.fetch = fetchMock;
    window.config = { doctorAssistantApiBaseUrl: 'https://remote-ai.example' };
  });

  it('encodes a DICOM series UID and uses the configured remote endpoint', async () => {
    const series = {
      id: 'series-a',
      study_id: 'study-a',
      dicom_series_uid: '1.2.3 & test',
      modality: 'ct',
      body_part: 'chest',
      analysis_eligible: true,
      ineligible_reason: null,
    };
    fetchMock.mockResolvedValue(response({ json: series }));

    await expect(findSeriesByDicomUid(series.dicom_series_uid)).resolves.toEqual(series);
    expect(fetchMock).toHaveBeenCalledWith(
      'https://remote-ai.example/v1/series?dicom_series_uid=1.2.3%20%26%20test',
      undefined
    );
  });

  it('turns a successful proxy HTML page into an actionable API error, not a JSON SyntaxError', async () => {
    fetchMock.mockResolvedValue(
      response({ contentType: 'text/html; charset=utf-8', text: '<!DOCTYPE html>' })
    );

    const request = getRuntimeHealth();
    await expect(request).rejects.toEqual(
      expect.objectContaining({
        name: 'ApiError',
        status: 502,
        message: expect.stringContaining('returned a web page instead of data'),
      })
    );
    await expect(request).rejects.toBeInstanceOf(ApiError);
  });

  it('downloads a remote DICOM SEG then sends the exact artifact to local Orthanc', async () => {
    const artifact = new Blob(['dicom-seg'], { type: 'application/dicom' });
    fetchMock
      .mockResolvedValueOnce(response({ contentType: 'application/dicom', blob: artifact }))
      .mockResolvedValueOnce(response({ json: { ID: 'orthanc-instance', Status: 'Success' } }));

    await expect(saveDicomSegArtifactToLocalOrthanc('/v1/artifacts/seg-sop-1')).resolves.toEqual({
      ID: 'orthanc-instance',
      Status: 'Success',
    });

    expect(fetchMock).toHaveBeenNthCalledWith(
      1,
      'https://remote-ai.example/v1/artifacts/seg-sop-1'
    );
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/local-orthanc-rest/instances', {
      method: 'POST',
      headers: { 'Content-Type': 'application/dicom' },
      body: artifact,
    });
  });
});
