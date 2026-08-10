import { RectangleROITool } from '@cornerstonejs/tools';

/**
 * A plain `RectangleROITool` under its own tool name — the exact "subclass an existing
 * annotation tool to get a distinct toolName" pattern `@cornerstonejs/tools` itself uses
 * for `RectangleROIThresholdTool` (extends `RectangleROITool`, only overrides
 * `toolName`). No behavior is overridden here: the box-*drawing* interaction (drag,
 * handles, rendering) is entirely RectangleROITool's own, already-shipped code.
 *
 * What happens once the box is drawn (calling the backend, painting the result) is NOT
 * this tool's job — it's wired externally via an `Events.ANNOTATION_COMPLETED` listener
 * scoped to this toolName (see `registerMedSAMBoxTool.ts`), the same
 * draw-then-react-to-completion split `RectangleROIThresholdTool`'s own callers use.
 *
 * Named "box", not "point"/"click", deliberately: MedSAM's released checkpoint was
 * fine-tuned exclusively on box prompts (see experts/medsam_interactive.py's module
 * docstring) — a single-click point prompt would be untested, off-label use of this
 * checkpoint.
 */
export class MedSAMBoxTool extends RectangleROITool {
  static toolName = 'MedSAMBox';
}

export default MedSAMBoxTool;
