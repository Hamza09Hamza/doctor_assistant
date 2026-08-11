import { cache, eventTarget, metaData, utilities as csUtils } from '@cornerstonejs/core';
import * as cornerstoneTools from '@cornerstonejs/tools';

import { ApiError, segmentBox, segmentVolume, type SegmentVolumeSlice } from '../apiClient';
import { decodeMaskRle } from '../segmentBoxRle';
import { MedSAMBoxTool } from './MedSAMBoxTool';

const { Enums: ToolEnums } = cornerstoneTools;

/**
 * A deliberately local, minimal copy of `@ohif/extension-cornerstone`'s own
 * `createSegmentationForViewport` (extensions/cornerstone/src/utils/
 * createSegmentationForViewport.ts) -- not re-exported from that package's public
 * `@ohif/extension-cornerstone` entry point, and this repo's own convention (see every
 * other extension's `@ohif/extension-cornerstone` import) is to depend only on that
 * public surface, not reach into another extension's internal `src/` path across the
 * package boundary. Per this repo's CLAUDE.md ("never modify core architecture, only as
 * a last resort"), extending that package's exports was rejected in favor of this small
 * self-contained duplicate.
 */
async function createLabelmapSegmentationForViewport(
  servicesManager: AppTypes.ServicesManager,
  viewportId: string,
  label: string
): Promise<string | undefined> {
  const { viewportGridService, displaySetService, segmentationService } = servicesManager.services;
  const { viewports } = viewportGridService.getState();
  const viewport = viewports.get(viewportId);
  const displaySetInstanceUID = viewport?.displaySetInstanceUIDs?.[0];
  if (!displaySetInstanceUID) {
    return undefined;
  }
  const displaySet = displaySetService.getDisplaySetByUID(displaySetInstanceUID);
  const segmentationId = crypto.randomUUID();
  const generatedSegmentationId = await segmentationService.createLabelmapForDisplaySet(displaySet, {
    label,
    segmentationId,
    segments: {},
  });
  await segmentationService.addSegmentationRepresentation(viewportId, {
    segmentationId,
    type: cornerstoneTools.Enums.SegmentationRepresentations.Labelmap,
  });
  return generatedSegmentationId;
}

/**
 * Wires the MedSAMBoxTool into the app: registers the tool class once, adds it to
 * every existing tool group, and reacts to a completed box by calling the backend and
 * painting the returned mask into a labelmap segmentation for the viewport that owns
 * it. Call once from the extension's `onModeEnter` (mirrors extensions/cornerstone's
 * own `onModeEnter`, which does the equivalent registration/event-wiring for its
 * built-in tools — see that file's module docstring).
 *
 * VERIFICATION STATUS: written against the real, installed @cornerstonejs/tools v5.6.8
 * / @cornerstonejs/core API (RectangleROITool subclassing, VoxelManager.setAtIJK,
 * SegmentationService.getLabelmapVolume, metaData.get('instance', ...),
 * csUtils.transformWorldToIndex — all grounded by reading the installed package
 * source/types, not guessed) but NOT yet run against a live `yarn dev` viewer. Per this
 * project's own evidentiary rule (see docs/MONAI_PATHOLOGY_EXPERTS_RESULTS.md's "two
 * coordinate bugs, both caught by looking, not by code review"), this must be treated as
 * unverified until someone actually draws a box in the running viewer and looks at
 * where the mask lands — do not report this as "wired" before that happens.
 */
export function registerMedSAMBoxTool({ servicesManager, seriesIdForSeriesUid }: {
  servicesManager: AppTypes.ServicesManager;
  /** Resolves a DICOM SeriesInstanceUID to this app's internal series id (the backend
   * route is keyed by the latter, not the DICOM UID) — see apiClient.findSeriesByDicomUid. */
  seriesIdForSeriesUid: (seriesInstanceUid: string) => Promise<string>;
}): { unsubscribe: () => void } {
  try {
    cornerstoneTools.addTool(MedSAMBoxTool);
  } catch {
    // already added in a previous mode-enter (addTool throws on a duplicate toolName)
  }

  const { toolGroupService, cornerstoneViewportService, segmentationService, uiNotificationService } =
    servicesManager.services;

  const addToolToGroup = (toolGroupId: string) => {
    const toolGroup = toolGroupService.getToolGroup(toolGroupId);
    if (!toolGroup) {
      return;
    }
    try {
      toolGroup.addTool(MedSAMBoxTool.toolName, {});
    } catch {
      // already added
    }
  };

  // Cover tool groups that already exist (mode re-entry) ...
  for (const toolGroupId of toolGroupService.getToolGroupIds()) {
    addToolToGroup(toolGroupId);
  }
  // ... and ones created later. OHIF creates tool groups lazily, only once a viewport
  // actually mounts (ToolGroupService.getToolGroupForViewport's create-on-demand
  // fallback) -- onModeEnter fires before any viewport exists, so the loop above alone
  // finds nothing on a fresh mode entry and the toolbar button stays permanently
  // "Not available on the current viewport." This is the actual fix for that, not the
  // loop above (kept only for the re-entry case). Mirrors this repo's own
  // CLAUDE.md guidance to prefer service pub/sub over a one-shot check.
  const toolGroupCreatedSubscription = toolGroupService.subscribe(
    toolGroupService.EVENTS.TOOLGROUP_CREATED,
    ({ toolGroupId }: { toolGroupId: string }) => addToolToGroup(toolGroupId)
  );

  const handleAnnotationCompleted = async (evt: { detail: { annotation: any } }) => {
    const { annotation } = evt.detail;
    if (annotation?.metadata?.toolName !== MedSAMBoxTool.toolName) {
      return;
    }

    try {
      const referencedImageId: string | undefined = annotation.metadata.referencedImageId;
      if (!referencedImageId) {
        throw new Error('MedSAMBox annotation has no referencedImageId (volume-viewport box drawing is not supported yet)');
      }

      const instance = metaData.get('instance', referencedImageId);
      const sopInstanceUid = instance?.SOPInstanceUID || instance?.SopInstanceUID;
      const seriesInstanceUid = instance?.SeriesInstanceUID;
      if (!sopInstanceUid || !seriesInstanceUid) {
        throw new Error(`could not resolve SOPInstanceUID/SeriesInstanceUID for ${referencedImageId}`);
      }

      // Find the live viewport currently displaying this image, to read its own
      // (cornerstone-computed, not hand-derived) world<->pixel geometry -- reusing
      // cornerstone's own transform rather than re-deriving DICOM orientation math is
      // deliberate: this project has already shipped two real coordinate bugs from
      // hand-rolled ImageOrientationPatient math (see notebooks/HANDOFF.md s3.2).
      const renderingEngine = cornerstoneViewportService.getRenderingEngine();
      const viewport = renderingEngine
        ?.getViewports()
        .find((vp: any) => vp.getCurrentImageId?.() === referencedImageId);
      if (!viewport) {
        throw new Error('no active viewport is currently displaying the annotated image');
      }
      // viewport.getImageData() returns an IImageData *wrapper* (dimensions, spacing,
      // scalarData, ...); the actual vtk-style ImageData with a real .worldToIndex()
      // method transformWorldToIndex() needs is nested at .imageData, not the wrapper
      // itself -- passing the wrapper directly is what raised
      // "imageData.worldToIndex is not a function".
      const imageDataWrapper = viewport.getImageData();
      const imageData = imageDataWrapper?.imageData;
      if (!imageData) {
        throw new Error('viewport has no imageData to convert the box into pixel coordinates');
      }

      const points: Array<[number, number, number]> = annotation.data.handles.points;
      const indices = points.map(world => csUtils.transformWorldToIndex(imageData, world));
      const iVals = indices.map(p => p[0]);
      const jVals = indices.map(p => p[1]);
      const boxXyxy: [number, number, number, number] = [
        Math.min(...iVals),
        Math.min(...jVals),
        Math.max(...iVals),
        Math.max(...jVals),
      ];

      const seriesId = await seriesIdForSeriesUid(seriesInstanceUid);
      let masks: SegmentVolumeSlice[];
      let resultLabel = 'MedSAM2 3D prompted segmentation';
      let successMessage: string;
      try {
        const voiRange = viewport.getProperties?.().voiRange;
        const voi =
          voiRange && Number.isFinite(voiRange.lower) && Number.isFinite(voiRange.upper)
            ? {
                windowCenter: (voiRange.lower + voiRange.upper) / 2,
                windowWidth: voiRange.upper - voiRange.lower,
              }
            : undefined;
        const volumeResult = await segmentVolume(seriesId, sopInstanceUid, boxXyxy, voi);
        resultLabel = volumeResult.model_version.startsWith('sam2-mlx:')
          ? 'SAM2 MLX 3D prompted segmentation'
          : 'MedSAM2 3D prompted segmentation';
        masks = volumeResult.masks;
        const persistence =
          volumeResult.orthanc_status === 'published'
            ? 'DICOM SEG saved to Orthanc.'
            : volumeResult.orthanc_status === 'failed'
              ? 'DICOM SEG created locally; Orthanc upload failed.'
              : 'DICOM SEG created locally.';
        successMessage =
          `${volumeResult.segmented_slice_count} slices · ` +
          `${volumeResult.volume_ml.toFixed(2)} mL · ` +
          `${volumeResult.axial_bbox_diagonal_mm.toFixed(1)} mm axial span. ${persistence}`;
      } catch (error) {
        // Keep the already-working 2D path available on machines where the larger
        // MedSAM2 runtime/checkpoint has not yet been installed. Only a capability
        // failure falls back; bad geometry/model errors must remain visible.
        if (!(error instanceof ApiError) || error.status !== 503) {
          throw error;
        }
        const sliceResult = await segmentBox(seriesId, sopInstanceUid, boxXyxy);
        masks = [
          {
            sop_instance_uid: sliceResult.sop_instance_uid,
            mask_rle: sliceResult.mask_rle,
          },
        ];
        resultLabel = 'MedSAM 2D box segmentation';
        successMessage = 'MedSAM2 is not configured, so this result covers the selected slice only.';
      }

      const segmentationId = await createLabelmapSegmentationForViewport(
        servicesManager,
        viewport.id,
        resultLabel
      );
      if (!segmentationId) {
        throw new Error('could not create a labelmap segmentation for this viewport');
      }
      const segmentIndex = 1;
      const referencedImageIds: string[] = viewport.getImageIds?.() || [];
      const labelmapImageIds = cornerstoneTools.segmentation.getLabelmapImageIds(segmentationId);
      if (!referencedImageIds.length || labelmapImageIds.length !== referencedImageIds.length) {
        throw new Error(
          `segmentation/source stack mismatch (${labelmapImageIds.length} labelmaps for ` +
            `${referencedImageIds.length} source images)`
        );
      }

      // Pair source and derived image IDs by the order used when
      // createLabelmapForDisplaySet created the labelmaps, then identify each source
      // slice by SOP UID. This remains correct whether OHIF displays the stack head to
      // foot or foot to head; no inferred k-direction is involved.
      const labelmapBySopUid = new Map<string, string>();
      referencedImageIds.forEach((sourceImageId, index) => {
        const sourceInstance = metaData.get('instance', sourceImageId);
        const sourceSopUid = sourceInstance?.SOPInstanceUID || sourceInstance?.SopInstanceUID;
        if (sourceSopUid) {
          labelmapBySopUid.set(sourceSopUid, labelmapImageIds[index]);
        }
      });

      for (const slice of masks) {
        const labelmapImageId = labelmapBySopUid.get(slice.sop_instance_uid);
        const labelmapImage = labelmapImageId ? cache.getImage(labelmapImageId) : undefined;
        if (!labelmapImage?.voxelManager) {
          throw new Error(`no labelmap image for segmented SOP ${slice.sop_instance_uid}`);
        }
        const flatMask = decodeMaskRle(slice.mask_rle);
        const [maskHeight, maskWidth] = slice.mask_rle.size;
        for (let col = 0; col < maskWidth; col++) {
          for (let row = 0; row < maskHeight; row++) {
            if (flatMask[row + col * maskHeight]) {
              labelmapImage.voxelManager.setAtIJK(col, row, 0, segmentIndex);
            }
          }
        }
      }
      cornerstoneTools.segmentation.triggerSegmentationEvents.triggerSegmentationDataModified(
        segmentationId
      );
      uiNotificationService?.show({
        title: resultLabel,
        message: successMessage,
        type: masks.length > 1 ? 'success' : 'warning',
      });
    } catch (error) {
      console.error('MedSAMBoxTool: segmentation failed', error);
      uiNotificationService?.show({
        title: 'Interactive segmentation failed',
        message: error instanceof Error ? error.message : String(error),
        type: 'error',
      });
    }
  };

  eventTarget.addEventListener(ToolEnums.Events.ANNOTATION_COMPLETED, handleAnnotationCompleted);

  return {
    unsubscribe: () => {
      eventTarget.removeEventListener(ToolEnums.Events.ANNOTATION_COMPLETED, handleAnnotationCompleted);
      toolGroupCreatedSubscription.unsubscribe();
    },
  };
}

export default registerMedSAMBoxTool;
