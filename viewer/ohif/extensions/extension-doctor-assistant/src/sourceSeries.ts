export type SeriesDisplaySet = {
  displaySetInstanceUID?: string;
  SeriesInstanceUID?: string;
  Modality?: string;
  madeInClient?: boolean;
  isOverlayDisplaySet?: boolean;
  referencedSeriesInstanceUID?: string;
  referencedDisplaySetInstanceUID?: string;
};

const OVERLAY_MODALITIES = new Set(['SEG', 'RTSTRUCT', 'SR', 'PR', 'PMAP']);

/**
 * Resolve an overlay such as DICOM SEG back to the source image series it annotates.
 *
 * The active OHIF display-set list is not guaranteed to put the source CT first. In
 * particular, hydrated LIDC reader SEG objects may be first, and their own series UID
 * must never be sent to the inference API as though it were the source volume.
 */
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

export default resolveSourceSeriesInstanceUid;
