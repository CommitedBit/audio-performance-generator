import { afterEach, describe, expect, it, vi } from 'vitest';
import { get, set } from 'idb-keyval';
import { loadClipAudio, saveBlob } from './useIndexedAudio';

// No IndexedDB outside a browser; the key it is given is what matters here.
vi.mock('idb-keyval', () => ({ get: vi.fn(), set: vi.fn(async () => {}) }));

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

afterEach(() => {
  vi.clearAllMocks();
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

  // IndexedDB full or unavailable: the generation was thrown away, though
  // the server kept the audio and the clip could fetch it back from there.
  it('still gives a key when the write fails but the server has a copy', async () => {
    vi.mocked(set).mockRejectedValueOnce(new DOMException('full', 'QuotaExceededError'));
    await expect(saveBlob(new Blob(['RIFF']), 'a1')).resolves.toMatch(UUID_V4);
  });

  // With no server copy, this one is the only one there is.
  it('throws when the write fails and the server has no copy', async () => {
    const full = new DOMException('full', 'QuotaExceededError');
    vi.mocked(set).mockRejectedValueOnce(full).mockRejectedValueOnce(full);
    await expect(saveBlob(new Blob(['RIFF']))).rejects.toThrow('full');
    await expect(saveBlob(new Blob(['RIFF']), null)).rejects.toThrow('full');
  });
});

describe('loadClipAudio', () => {
  /** The server, holding audio id a1. Returns the URLs it was asked for. */
  function server(): string[] {
    const asked: string[] = [];
    vi.stubGlobal('fetch', async (url: string) => {
      asked.push(url);
      return url === '/api/v1/audio/a1'
        ? new Response(new Blob(['from-server']))
        : new Response(null, { status: 404, statusText: 'Not Found' });
    });
    return asked;
  }

  it('uses the cached copy without asking the server', async () => {
    const asked = server();
    vi.mocked(get).mockResolvedValueOnce(new Blob(['cached']));

    const blob = await loadClipAudio({ blobId: 'k1', audioId: 'a1' });

    expect(await blob.text()).toBe('cached');
    expect(get).toHaveBeenCalledWith('k1');
    expect(asked).toEqual([]);
  });

  // Evicted by the browser or cleared with site data: it used to be "audio
  // missing" for good, with the audio still on the server.
  it('fetches the audio back from the server once the cache has lost it, and caches it again', async () => {
    const asked = server();
    vi.mocked(get).mockResolvedValueOnce(undefined);

    const blob = await loadClipAudio({ blobId: 'k1', audioId: 'a1' });

    expect(await blob.text()).toBe('from-server');
    expect(asked).toEqual(['/api/v1/audio/a1']);
    expect(set).toHaveBeenCalledWith('k1', blob);
  });

  it('treats a cache that cannot be opened as an empty one', async () => {
    server();
    vi.mocked(get).mockRejectedValueOnce(new DOMException('no IndexedDB here', 'InvalidStateError'));

    const blob = await loadClipAudio({ blobId: 'k1', audioId: 'a1' });
    expect(await blob.text()).toBe('from-server');
  });

  it('still returns the audio when caching it again fails', async () => {
    server();
    vi.mocked(get).mockResolvedValueOnce(undefined);
    vi.mocked(set).mockRejectedValueOnce(new DOMException('full', 'QuotaExceededError'));

    const blob = await loadClipAudio({ blobId: 'k1', audioId: 'a1' });
    expect(await blob.text()).toBe('from-server');
  });

  it('says which copy is missing when neither exists', async () => {
    const asked = server();
    vi.mocked(get).mockResolvedValueOnce(undefined).mockResolvedValueOnce(undefined);

    await expect(loadClipAudio({ blobId: 'k1' })).rejects.toThrow(
      'audio missing: not cached in this browser, and the clip has no server copy'
    );
    expect(asked).toEqual([]);
    // The server deleted it too.
    await expect(loadClipAudio({ blobId: 'k1', audioId: 'deleted' })).rejects.toMatchObject({
      name: 'ApiError',
      status: 404,
    });
  });
});
