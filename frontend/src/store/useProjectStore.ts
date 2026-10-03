import { create } from 'zustand';
import type { Project, Clip, TrackType } from '../types/timeline';

export interface ProjectState {
  project: Project;
  selectedClipId?: string;
  playing: boolean;
  addClip: (trackId: string, blobId: string, duration: number) => void;
  moveClip: (clipId: string, start: number) => void;
  /** update an existing clip with a partial set of fields */
  updateClip: (id: string, partial: Partial<Clip>) => void;
  setPlaying: (playing: boolean) => void;
}

const useProjectStore = create<ProjectState>(set => ({
  project: {
    id: 'project1',
    name: 'New Project',
    // One track per generated type. The 'sfx' and 'music' TrackType variants
    // existed but were unreachable: nothing could ever create a track for them.
    tracks: [
      { id: 'track-voice', type: 'voice', name: 'Voice' },
      { id: 'track-sfx', type: 'sfx', name: 'SFX' },
      { id: 'track-music', type: 'music', name: 'Music' },
    ],
    clips: [],
  },
  selectedClipId: undefined,
  playing: false,
  addClip: (trackId, blobId, duration) =>
    set(state => {
      const clipsForTrack = state.project.clips.filter(c => c.trackId === trackId);
      const start = clipsForTrack.reduce((acc, c) => Math.max(acc, c.start + c.duration), 0);
      const clip: Clip = {
        id: crypto.randomUUID(),
        trackId,
        blobId,
        start,
        offset: 0,
        duration,
      };
      return {
        project: { ...state.project, clips: [...state.project.clips, clip] },
      };
    }),
  moveClip: (clipId, start) =>
    set(state => ({
      project: {
        ...state.project,
        clips: state.project.clips.map(c =>
          c.id === clipId ? { ...c, start } : c
        ),
      },
    })),
  updateClip: (id, partial) =>
    set(state => ({
      project: {
        ...state.project,
        clips: state.project.clips.map(c =>
          c.id === id ? { ...c, ...partial } : c
        ),
      },
    })),
  setPlaying: playing => set({ playing }),
}));

export default useProjectStore;

// Pure helper: returns clips ordered by start time.
// NOTE: do NOT pass this to useProjectStore() as a selector. It allocates a new
// array per call, and zustand v5 compares snapshots with Object.is, so an
// unstable selector re-renders forever. Call it inside a useMemo instead.
export const orderedClips = (clips: Clip[]) =>
  [...clips].sort((a, b) => a.start - b.start);

/** The track that generated audio of a given type belongs on. */
export const trackIdForType = (project: Project, type: TrackType): string | undefined =>
  project.tracks.find(t => t.type === type)?.id;
