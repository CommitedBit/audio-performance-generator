import { describe, expect, it } from 'vitest';
import type { Clip } from '../types/timeline';
import { MIN_CLIP_SECONDS, trimClip } from './trim';

// 2 s into the timeline, playing 1 s..5 s of a 6 s source.
function clip(extra: Partial<Clip> = {}): Clip {
  return {
    id: 'c1',
    trackId: 'track-voice',
    blobId: 'b1',
    start: 2,
    offset: 1,
    duration: 4,
    sourceDuration: 6,
    ...extra,
  };
}

/** The timeline time at which the source's first sample would play. */
const anchor = (c: Pick<Clip, 'start' | 'offset'>) => c.start - c.offset;

describe('left trim', () => {
  it('moves start with offset, so the audio stays where it was in time', () => {
    const before = clip();
    const after = trimClip(before, 'left', 1.5);
    expect(after).toEqual({ start: 3.5, offset: 2.5, duration: 2.5 });
    expect(anchor(after)).toBe(anchor(before));
  });

  it('extends back out as far as the start of the source, and no further', () => {
    expect(trimClip(clip(), 'left', -0.5)).toEqual({ start: 1.5, offset: 0.5, duration: 4.5 });
    expect(trimClip(clip(), 'left', -10)).toEqual({ start: 1, offset: 0, duration: 5 });
  });

  it('extends back out no further than 0 on the timeline', () => {
    expect(trimClip(clip({ start: 0.5, offset: 2 }), 'left', -10)).toEqual({
      start: 0,
      offset: 1.5,
      duration: 4.5,
    });
  });

  it('never trims below the minimum length', () => {
    const after = trimClip(clip(), 'left', 10);
    expect(after.duration).toBeCloseTo(MIN_CLIP_SECONDS);
    expect(after.start + after.duration).toBeCloseTo(6); // the right edge stays put
    expect(anchor(after)).toBeCloseTo(anchor(clip()));
  });
});

describe('right trim', () => {
  it('extends back out as far as sourceDuration, and no further', () => {
    expect(trimClip(clip(), 'right', 0.5)).toEqual({ start: 2, offset: 1, duration: 4.5 });
    const after = trimClip(clip(), 'right', 10);
    expect(after).toEqual({ start: 2, offset: 1, duration: 5 });
    expect(after.offset + after.duration).toBe(6);
  });

  it('never trims below the minimum length', () => {
    expect(trimClip(clip(), 'right', -10)).toEqual({ start: 2, offset: 1, duration: MIN_CLIP_SECONDS });
  });

  // A clip made without sourceDuration: nothing says audio exists past its end.
  it('cannot extend past the current end when the source length is unknown', () => {
    expect(trimClip(clip({ sourceDuration: undefined }), 'right', 1).duration).toBe(4);
  });
});

// A blip of a sound effect. The old trim snapped it up to 0.1 s on the first
// touch of a handle, and on the left that pushed offset below 0, which
// AudioBufferSourceNode.start() rejects with a RangeError.
it('does not stretch a clip already shorter than the minimum', () => {
  const blip = clip({ start: 1, offset: 0, duration: 0.0625, sourceDuration: 0.0625 });
  expect(trimClip(blip, 'left', 0.5)).toEqual({ start: 1, offset: 0, duration: 0.0625 });
  expect(trimClip(blip, 'right', -0.5)).toEqual({ start: 1, offset: 0, duration: 0.0625 });
});
