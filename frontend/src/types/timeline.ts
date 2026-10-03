export type TrackType = 'voice' | 'sfx' | 'music';

export interface Track {
  id: string;
  type: TrackType;
  name: string;
}

// The optional fields are read with a default wherever they are used, so a
// clip made without them -- before they existed, or by an import that cannot
// know them -- still plays and trims.
export interface Clip {
  id: string;
  trackId: string;
  /** IndexedDB key of this browser's copy of the audio. Only a cache: see audioId. */
  blobId: string;
  /**
   * The server's id for the audio (GET /v1/audio/{audioId}). The browser may
   * evict IndexedDB and clearing site data empties it, so this is the copy
   * the audio is fetched back from.
   */
  audioId?: string;
  /** What the clip shows on the timeline: the prompt it was generated from. */
  label?: string;
  start: number;
  /** seconds into the source audio where playback should begin */
  offset: number;
  duration: number;
  /**
   * Length of the whole source audio, seconds: how far the trim handles can
   * extend back out. Absent, a clip cannot be lengthened past its current end.
   */
  sourceDuration?: number;
  /** linear; absent means 1 */
  gain?: number;
  /** seconds; absent means no fade */
  fadeIn?: number;
  /** seconds; absent means no fade */
  fadeOut?: number;
}

export interface Project {
  id: string;
  name: string;
  tracks: Track[];
  clips: Clip[];
}
