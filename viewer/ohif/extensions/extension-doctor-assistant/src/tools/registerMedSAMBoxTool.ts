import { cache, eventTarget, metaData, utilities as csUtils } from '@cornerstonejs/core';
import * as cornerstoneTools from '@cornerstonejs/tools';

import { ApiError, segmentBox, segmentVolume, type SegmentVolumeSlice } from '../apiClient';
import {
  LUNG_NODULE_CANDIDATE_SELECTED,
  type LungNoduleCandidateSelectedDetail,
} from '../lungNoduleCandidateEvents';
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

  const {
    toolGroupService,
    viewportGridService,
    cornerstoneViewportService,
    segmentationService,
    uiNotificationService,
  } = servicesManager.services;

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

  let inferenceInProgress = false;

  const handleAnnotationCompleted = async (evt: { detail: { annotation: any } }) => {
    const { annotation } = evt.detail;
    if (annotation?.metadata?.toolName !== MedSAMBoxTool.toolName) {
      return;
    }

    if (inferenceInProgress) {
      uiNotificationService?.show({
        title: 'MedSAM2 is already working',
        message: 'Wait for the current Colab segmentation to finish before drawing another box.',
        type: 'warning',
        duration: 5000,
      });
      return;
    }

    inferenceInProgress = true;
    const activeToolGroups = toolGroupService
      .getToolGroupIds()
      .map((toolGroupId: string) => toolGroupService.getToolGroup(toolGroupId))
      .filter(
        (toolGroup: any) =>
          toolGroup?.getToolOptions(MedSAMBoxTool.toolName)?.mode === ToolEnums.ToolModes.Active
      );
    const activeToolGroupBindings = new Map(
      activeToolGroups.map((toolGroup: any) => [
        toolGroup,
        [...(toolGroup.getToolOptions(MedSAMBoxTool.toolName)?.bindings || [])],
      ])
    );
    activeToolGroups.forEach((toolGroup: any) =>
      toolGroup.setToolDisabled(MedSAMBoxTool.toolName)
    );
    const loadingNotificationId = uiNotificationService?.show({
      title: 'Segmenting selected structure…',
      message: 'Colab is processing the CT volume. Please wait; only one prompt runs at a time.',
      type: 'loading',
      autoClose: false,
      allowDuplicates: true,
    });

    try {
      // RectangleROI may omit referencedImageId for a volume viewport even though it
      // has an exact current source slice. Cornerstone already knows which viewport
      // owns the annotation from its frame-of-reference and view-plane metadata; use
      // that public utility, then ask the viewport for its current image instead of
      // reimplementing slice geometry.
      const activeViewportId = viewportGridService.getActiveViewportId();
      const activeViewport = activeViewportId
        ? cornerstoneViewportService.getCornerstoneViewport(activeViewportId)
        : undefined;
      const viewport =
        cornerstoneTools.utilities.getViewportForAnnotation(annotation) || activeViewport;
      const firstBoxPoint = annotation.data.handles.points[0];
      const volumeId = viewport?.getVolumeId?.();
      const imageVolume = volumeId ? cache.getVolume(volumeId) : undefined;
      const viewPlaneNormal =
        annotation.metadata.viewPlaneNormal || viewport?.getCamera?.().viewPlaneNormal;
      const referencedImageId: string | undefined =
        annotation.metadata.referencedImageId ||
        viewport?.getCurrentImageId?.() ||
        (imageVolume && viewPlaneNormal
          ? csUtils.getClosestImageId(imageVolume, firstBoxPoint, viewPlaneNormal, {
              ignoreSpacing: true,
            })
          : undefined);
      if (!referencedImageId) {
        throw new Error('could not resolve the source slice for the MedSAM2 box');
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
      const candidateNumber: number | undefined =
        annotation.metadata.doctorAssistantCandidateNumber;
      const candidateScore: number | undefined =
        annotation.metadata.doctorAssistantCandidateScore;
      const isDetectorCandidate = Number.isInteger(candidateNumber);
      let masks: SegmentVolumeSlice[];
      let resultLabel = 'MedSAM2 prompted structure';
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
        const volumeResult = await segmentVolume(
          seriesId,
          sopInstanceUid,
          boxXyxy,
          voi,
          isDetectorCandidate
            ? `AI pulmonary nodule candidate ${candidateNumber}`
            : 'AI prompted structure'
        );
        const segmenterName = volumeResult.model_version.startsWith('sam2-mlx:')
          ? 'SAM2 MLX'
          : 'MedSAM2';
        resultLabel = isDetectorCandidate
          ? `Nodule candidate ${candidateNumber} · ${segmenterName} outline`
          : `${segmenterName} prompted structure`;
        masks = volumeResult.masks;
        const persistence =
          volumeResult.orthanc_status === 'published'
            ? 'DICOM SEG saved to Orthanc.'
            : volumeResult.orthanc_status === 'failed'
              ? 'DICOM SEG created in the API runtime; Orthanc upload failed.'
              : 'DICOM SEG created in the API runtime.';
        const candidatePrefix = isDetectorCandidate && Number.isFinite(candidateScore)
          ? `Uncalibrated detector score ${candidateScore!.toFixed(3)}. `
          : '';
        successMessage = candidatePrefix +
          `${volumeResult.segmented_slice_count} slices · ` +
          `${volumeResult.volume_ml.toFixed(2)} mL · ` +
          `${volumeResult.axial_bbox_diagonal_mm.toFixed(1)} mm axial span. ${persistence} ` +
          'This mask follows the box prompt; it does not confirm an anomaly.';
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
        resultLabel = 'MedSAM prompted structure (2D)';
        successMessage =
          'This result covers the selected slice only and does not confirm an anomaly.';
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
        type: 'warning',
      });
    } catch (error) {
      console.error('MedSAMBoxTool: segmentation failed', error);
      uiNotificationService?.show({
        title: 'Interactive segmentation failed',
        message: error instanceof Error ? error.message : String(error),
        type: 'error',
      });
    } finally {
      if (loadingNotificationId) {
        uiNotificationService?.hide(loadingNotificationId);
      }
      activeToolGroups.forEach((toolGroup: any) => {
        // Restore the exact bindings we captured. ToolGroup.setToolMode has a default
        // empty options object, so calling it with only (name, Active) does not use its
        // internal restoreToolOptions and would reactivate an unusable, unbound tool.
        toolGroup.setToolActive(MedSAMBoxTool.toolName, {
          bindings: activeToolGroupBindings.get(toolGroup) || [],
        });
      });
      inferenceInProgress = false;
    }
  };

  const handleCandidateSelected = async (
    evt: CustomEvent<LungNoduleCandidateSelectedDetail>
  ) => {
    const { seriesId: selectedSeriesId, candidate, candidateIndex } = evt.detail;
    if (inferenceInProgress) {
      uiNotificationService?.show({
        title: 'MedSAM2 is already working',
        message: 'Wait for the current Colab segmentation to finish before selecting another candidate.',
        type: 'warning',
        duration: 5000,
      });
      return;
    }
    const activeViewportId = viewportGridService.getActiveViewportId();
    const viewport = activeViewportId
      ? cornerstoneViewportService.getCornerstoneViewport(activeViewportId)
      : undefined;
    if (!viewport) {
      uiNotificationService?.show({
        title: 'Could not open candidate',
        message: 'Select the axial CT viewport, then choose the candidate again.',
        type: 'error',
      });
      return;
    }

    try {
      const imageIds: string[] = viewport.getImageIds?.() || [];
      const imageIndex = imageIds.findIndex(imageId => {
        const instance = metaData.get('instance', imageId);
        return (
          instance?.SOPInstanceUID === candidate.seed_sop_instance_uid ||
          instance?.SopInstanceUID === candidate.seed_sop_instance_uid
        );
      });
      if (imageIndex < 0) {
        throw new Error('the selected candidate slice is not present in the active CT viewport');
      }

      if (typeof viewport.jumpToWorld === 'function') {
        viewport.jumpToWorld(candidate.center_lps_mm);
        viewport.render?.();
      } else {
        await csUtils.jumpToSlice(viewport.element, { imageIndex });
      }
      // Volume viewports can keep the same vtk ImageData object across slices. Wait
      // one animation frame so Cornerstone has applied the requested focal-plane
      // change before we build and submit the synthetic box annotation.
      await new Promise<void>(resolve => requestAnimationFrame(() => resolve()));
      const currentImageId = viewport.getCurrentImageId?.();
      if (currentImageId) {
        const currentInstance = metaData.get('instance', currentImageId);
        const currentSopUid =
          currentInstance?.SOPInstanceUID || currentInstance?.SopInstanceUID;
        if (currentSopUid && currentSopUid !== candidate.seed_sop_instance_uid) {
          throw new Error('viewer did not reach the detector candidate source slice');
        }
      }
      const imageData = viewport.getImageData?.()?.imageData;
      if (!imageData) {
        throw new Error('viewport has no imageData for candidate geometry');
      }

      // Convert the backend's source-pixel box into the same world-point shape a
      // RectangleROI annotation carries. Reusing the ordinary annotation handler below
      // keeps candidate and manual prompts on one geometry/painting path.
      const centerIndex = csUtils.transformWorldToIndex(
        imageData,
        candidate.center_lps_mm
      );
      const k = centerIndex[2];
      const [x0, y0, x1, y1] = candidate.box_xyxy;
      const points = [
        [x0, y0, k],
        [x1, y0, k],
        [x0, y1, k],
        [x1, y1, k],
      ].map(index => csUtils.transformIndexToWorld(imageData, index));
      const camera = viewport.getCamera?.();
      const referencedImageId = imageIds[imageIndex];
      const instance = metaData.get('instance', referencedImageId);
      if (!instance) {
        throw new Error('candidate source metadata is unavailable after slice navigation');
      }
      const activeSeriesUid = instance?.SeriesInstanceUID;
      if (!activeSeriesUid) {
        throw new Error('candidate source series metadata is unavailable');
      }
      const activeSeriesId = await seriesIdForSeriesUid(activeSeriesUid);
      if (activeSeriesId !== selectedSeriesId) {
        throw new Error('the selected candidate belongs to a different active series');
      }

      await handleAnnotationCompleted({
        detail: {
          annotation: {
            metadata: {
              toolName: MedSAMBoxTool.toolName,
              referencedImageId,
              FrameOfReferenceUID:
                instance?.FrameOfReferenceUID || viewport.getFrameOfReferenceUID?.(),
              viewPlaneNormal: camera?.viewPlaneNormal,
              viewUp: camera?.viewUp,
              doctorAssistantCandidateNumber: candidateIndex + 1,
              doctorAssistantCandidateScore: candidate.score,
            },
            data: { handles: { points } },
          },
        },
      });
    } catch (error) {
      uiNotificationService?.show({
        title: 'Could not open candidate',
        message: error instanceof Error ? error.message : String(error),
        type: 'error',
      });
    }
  };

  eventTarget.addEventListener(ToolEnums.Events.ANNOTATION_COMPLETED, handleAnnotationCompleted);
  eventTarget.addEventListener(LUNG_NODULE_CANDIDATE_SELECTED, handleCandidateSelected);

  return {
    unsubscribe: () => {
      eventTarget.removeEventListener(ToolEnums.Events.ANNOTATION_COMPLETED, handleAnnotationCompleted);
      eventTarget.removeEventListener(LUNG_NODULE_CANDIDATE_SELECTED, handleCandidateSelected);
      toolGroupCreatedSubscription.unsubscribe();
    },
  };
}

export default registerMedSAMBoxTool;
