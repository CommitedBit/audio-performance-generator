import { afterEach, describe, expect, it, vi } from 'vitest';
import useProjectStore from './useProjectStore';

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('addClip', () => {
  // Plain-HTTP LAN access (http://gpu-box:8080) is not a secure context, so
  // crypto.randomUUID is undefined there; getRandomValues still exists.
  it('works without crypto.randomUUID', () => {
    const real = globalThis.crypto;
    vi.stubGlobal('crypto', { getRandomValues: real.getRandomValues.bind(real) });

    const { addClip } = useProjectStore.getState();
    addClip('track-music', 'blob-a', 4);
    addClip('track-music', 'blob-b', 2);

    const clips = useProjectStore.getState().project.clips;
    expect(clips.map(c => c.blobId)).toEqual(['blob-a', 'blob-b']);
    for (const c of clips) expect(c.id).toMatch(UUID_V4);
    expect(clips[0].id).not.toBe(clips[1].id);
    // Appended after the track's last clip, as before.
    expect(clips[1].start).toBe(4);
  });
});
