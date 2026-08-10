/**
 * Decoder for the COCO-style RLE the backend returns from POST
 * /v1/series/{id}/segment-box (see experts/medsam_interactive.py::encode_binary_mask_rle
 * on the Python side — this is its exact inverse, kept as a pure function so it's
 * unit-testable without a live viewport/backend).
 *
 * Column-major (Fortran) order: `counts[0]` is always a background run (possibly
 * zero-length, when the mask starts on foreground) so a decoder never special-cases
 * "starts true."
 */

export interface MaskRLE {
  size: [number, number]; // [height, width]
  counts: number[];
}

export function decodeMaskRle(rle: MaskRLE): Uint8Array {
  const [height, width] = rle.size;
  const flat = new Uint8Array(height * width);

  let pos = 0;
  let value = 0;
  for (const run of rle.counts) {
    if (value) {
      flat.fill(1, pos, pos + run);
    }
    pos += run;
    value = value ? 0 : 1;
  }
  return flat; // column-major: flat[row + col * height] === mask[row, col]
}

export function maskValueAt(flat: Uint8Array, height: number, row: number, col: number): number {
  return flat[row + col * height];
}
