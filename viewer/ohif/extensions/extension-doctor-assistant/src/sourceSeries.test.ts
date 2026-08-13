import { resolveSourceSeriesInstanceUid, type SeriesDisplaySet } from './sourceSeries';

const lookup = (sets: SeriesDisplaySet[]) => (uid: string) =>
  sets.find(displaySet => displaySet.displaySetInstanceUID === uid);

describe('resolveSourceSeriesInstanceUid', () => {
  it('returns the active image-series UID when CT is first', () => {
    const displaySets = [{ SeriesInstanceUID: 'ct-series', Modality: 'CT' }];

    expect(resolveSourceSeriesInstanceUid(displaySets, lookup(displaySets))).toBe('ct-series');
  });

  it('follows a SEG direct series reference instead of using the SEG series UID', () => {
    const displaySets = [
      {
        SeriesInstanceUID: 'reader-seg-series',
        Modality: 'SEG',
        isOverlayDisplaySet: true,
        referencedSeriesInstanceUID: 'source-ct-series',
      },
    ];

    expect(resolveSourceSeriesInstanceUid(displaySets, lookup(displaySets))).toBe(
      'source-ct-series'
    );
  });

  it('follows a SEG display-set reference to its source image series', () => {
    const displaySets = [
      {
        displaySetInstanceUID: 'seg-display-set',
        SeriesInstanceUID: 'seg-series',
        Modality: 'SEG',
        referencedDisplaySetInstanceUID: 'ct-display-set',
      },
      {
        displaySetInstanceUID: 'ct-display-set',
        SeriesInstanceUID: 'source-ct-series',
        Modality: 'CT',
      },
    ];

    expect(resolveSourceSeriesInstanceUid(displaySets, lookup(displaySets))).toBe(
      'source-ct-series'
    );
  });

  it('falls back to the first non-overlay image set when a hydrated SEG is first', () => {
    const displaySets = [
      { SeriesInstanceUID: 'seg-series', Modality: 'SEG' },
      { SeriesInstanceUID: 'structured-report', Modality: 'SR' },
      { SeriesInstanceUID: 'source-ct-series', Modality: 'ct' },
    ];

    expect(resolveSourceSeriesInstanceUid(displaySets, lookup(displaySets))).toBe(
      'source-ct-series'
    );
  });

  it('never treats an unreferenced overlay UID as the source volume', () => {
    const displaySets = [{ SeriesInstanceUID: 'seg-only', Modality: 'SEG' }];

    expect(resolveSourceSeriesInstanceUid(displaySets, lookup(displaySets))).toBeNull();
    expect(resolveSourceSeriesInstanceUid(undefined, lookup(displaySets))).toBeNull();
  });
});
