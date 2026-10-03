import { get, set } from 'idb-keyval';
import { fetchAudioById } from '../api/client';
import { newId } from '../lib/id';
import type { Clip } from '../types/timeline';

/**
 * Cache freshly generated audio in this browser, and return its key.
 *
 * When the server holds the audio too (`audioId`), a failed write is not
 * fatal: loadClipAudio fetches the audio back by that id. It used to throw
 * regardless, and so threw away a finished generation wherever IndexedDB is
 * full or cannot be opened, although the server had kept the audio. With no
 * server copy this one is the only one, so failing to keep it still throws.
 */
export async function saveBlob(blob: Blob, audioId?: string | null): Promise<string> {
  const id = newId();
  try {
    await set(id, blob);
  } catch (err) {
    if (!audioId) throw err;
  }
  return id;
}

export async function getBlob(id: string): Promise<Blob | undefined> {
  return await get<Blob>(id);
}

/**
 * A clip's audio: this browser's cached copy, or else the server's.
 *
 * IndexedDB is only a cache. The browser may evict it under storage pressure
 * and clearing site data empties it; while it held the only reference to the
 * audio, every clip made before that failed with "audio missing" for good,
 * although the server still had it. A cache that cannot be opened at all
 * (older Firefox private windows) counts as an empty one.
 */
export async function loadClipAudio(clip: Pick<Clip, 'blobId' | 'audioId'>): Promise<Blob> {
  const cached = await getBlob(clip.blobId).catch(() => undefined);
  if (cached) return cached;
  if (!clip.audioId) {
    throw new Error('audio missing: not cached in this browser, and the clip has no server copy');
  }
  const blob = await fetchAudioById(clip.audioId);
  // Cached again under the same key, so the next Play stays local. Best
  // effort: the audio is in hand either way.
  await set(clip.blobId, blob).catch(() => {});
  return blob;
}
