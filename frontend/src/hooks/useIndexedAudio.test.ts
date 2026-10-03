import { afterEach, describe, expect, it, vi } from 'vitest';
import { set } from 'idb-keyval';
import { saveBlob } from './useIndexedAudio';

// No IndexedDB outside a browser; the key it is given is what matters here.
vi.mock('idb-keyval', () => ({ get: vi.fn(), set: vi.fn(async () => {}) }));

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('saveBlob', () => {
  // The first thing to break on plain-HTTP LAN access: it runs right after the
  // audio is fetched, so the generation was lost at the last step.
  it('works without crypto.randomUUID', async () => {
    const real = globalThis.crypto;
    vi.stubGlobal('crypto', { getRandomValues: real.getRandomValues.bind(real) });

    const blob = new Blob(['RIFF']);
    const id = await saveBlob(blob);

    expect(id).toMatch(UUID_V4);
    expect(set).toHaveBeenCalledWith(id, blob);
  });
});
