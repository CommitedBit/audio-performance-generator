import type { Clip } from '../types/timeline';

/** Seconds. No trim goes shorter: a clip trimmed to nothing could never be grabbed to lengthen it again. */
export const MIN_CLIP_SECONDS = 0.1;

export type TrimSide = 'left' | 'right';

const clamp = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);

/**
 * A clip's placement after dragging one trim handle `delta` seconds from
 * where the drag began (positive is rightwards).
 *
 * Always computed from the clip as it was when the drag began, not step by
 * step, so rounding cannot build up over a long drag.
 */
export function trimClip(
  clip: Clip,
  side: TrimSide,
  delta: number
): Pick<Clip, 'start' | 'offset' | 'duration'> {
  const { start, offset, duration } = clip;
  // A clip already shorter than the minimum keeps its length rather than
  // being stretched by the first touch of a handle.
  const minLength = Math.min(MIN_CLIP_SECONDS, duration);

  if (side === 'left') {
    // start moves with offset, so every sound stays at the moment it was
    // placed. Moving offset alone shifted the whole clip's audio earlier by
    // the amount trimmed. The edge can go back out as far as the start of
    // the source, but not before 0 on the timeline.
    const d = clamp(delta, -Math.min(offset, start), duration - minLength);
    return { start: start + d, offset: offset + d, duration: duration - d };
  }

  // Without a known source length, the current end is the only one known to
  // have audio behind it.
  const sourceEnd = clip.sourceDuration ?? offset + duration;
  return { start, offset, duration: clamp(duration + delta, minLength, sourceEnd - offset) };
}
