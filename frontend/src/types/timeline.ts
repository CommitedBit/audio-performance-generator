export type TrackType = 'voice' | 'sfx' | 'music';

export interface Track {
  id: string;
  type: TrackType;
  name: string;
}

export interface Clip {
  id: string;
  trackId: string;
  blobId: string;
  start: number;
  /** seconds into the source audio where playback should begin */
  offset: number;
  duration: number;
}

export interface Project {
  id: string;
  name: string;
  tracks: Track[];
  clips: Clip[];
}
