import { useEffect, useMemo, useState } from 'react';
import { ApiError } from '../api/client';
import useProjectStore, { orderedClips } from '../store/useProjectStore';
import type { Clip, TrackType } from '../types/timeline';
import { loadClipAudio } from '../hooks/useIndexedAudio';
import { AudioEngine, runPlayback } from '../engine/AudioEngine';
import { type TrimSide, trimClip } from '../lib/trim';

const PIXELS_PER_SEC = 100;

const colors: Record<TrackType, string> = {
  voice: '#3b82f6', // blue-500
  sfx: '#f59e0b',   // amber-500
  music: '#10b981', // emerald-500
};

// One engine, and so one AudioContext, for the page. Made per Timeline mount,
// each visit to the editor opened another context and none was ever closed.
const engine = new AudioEngine(loadClipAudio);

export default function Timeline() {
  const project = useProjectStore(s => s.project);
  const rawClips = useProjectStore(s => s.project.clips);
  const clips = useMemo(() => orderedClips(rawClips), [rawClips]);
  const updateClip = useProjectStore(s => s.updateClip);
  const playing = useProjectStore(s => s.playing);
  const setPlaying = useProjectStore(s => s.setPlaying);
  const [error, setError] = useState<string | null>(null);

  // The engine outlives this component; leaving the editor stops what it plays.
  useEffect(() => () => engine.stop(), []);

  const play = () => {
    if (playing) return;
    setError(null);
    // Nothing is awaited before engine.play, so it runs inside this click
    // and can resume the AudioContext.
    runPlayback(() => engine.play(clips), setPlaying).catch((err: unknown) => {
      setError(err instanceof ApiError ? `${err.message} (HTTP ${err.status})` : String(err));
    });
  };

  const handleMove = (
    e: React.PointerEvent<HTMLDivElement>,
    clipId: string,
    startPx: number
  ) => {
    e.preventDefault();
    const startX = e.clientX;
    const target = e.currentTarget as HTMLElement;
    let newLeft = startPx;
    let raf = 0 as number | null;
    const move = (ev: PointerEvent) => {
      const delta = ev.clientX - startX;
      newLeft = Math.max(0, startPx + delta);
      if (!raf)
        raf = requestAnimationFrame(() => {
          target.style.left = `${newLeft}px`;
          raf = 0;
        });
    };
    const up = () => {
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', up);
      updateClip(clipId, { start: newLeft / PIXELS_PER_SEC });
    };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up);
  };

  const handleTrim = (e: React.PointerEvent<HTMLDivElement>, clip: Clip, side: TrimSide) => {
    e.stopPropagation();
    e.preventDefault();
    const startX = e.clientX;
    const el = e.currentTarget.parentElement as HTMLElement;
    let patch: ReturnType<typeof trimClip> | null = null;
    const move = (ev: PointerEvent) => {
      patch = trimClip(clip, side, (ev.clientX - startX) / PIXELS_PER_SEC);
      // A left trim moves the clip's left edge, not just its width.
      el.style.left = `${patch.start * PIXELS_PER_SEC}px`;
      el.style.width = `${patch.duration * PIXELS_PER_SEC}px`;
    };
    const up = () => {
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', up);
      if (patch) updateClip(clip.id, patch);
    };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up);
  };

  return (
    <div className="panel">
      <div className="row">
        <button onClick={play} disabled={playing || clips.length === 0}>Play</button>
        <button onClick={() => engine.stop()} disabled={!playing}>Stop</button>
        <span className="muted small">
          {clips.length} clip{clips.length === 1 ? '' : 's'}
        </span>
      </div>
      {error && <p className="error">{error}</p>}
      <div className="timeline">
        {project.tracks.map(track => (
          <div key={track.id} className="track">
            <span className="track-label">{track.name}</span>
            {project.clips.filter(c => c.trackId === track.id).map(c => {
              const left = c.start * PIXELS_PER_SEC;
              const width = c.duration * PIXELS_PER_SEC;
              return (
                <div
                  key={c.id}
                  className="clip"
                  title={c.label}
                  style={{
                    position: 'absolute',
                    left,
                    width,
                    top: 8,
                    height: 48,
                    background: colors[track.type],
                  }}
                  onPointerDown={e => handleMove(e, c.id, left)}
                >
                  {c.label && <span className="clip-label">{c.label}</span>}
                  <div
                    className="left-handle"
                    style={{
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      width: 6,
                      height: '100%',
                      cursor: 'ew-resize',
                      background: '#fff',
                    }}
                    onPointerDown={e => handleTrim(e, c, 'left')}
                  />
                  <div
                    className="right-handle"
                    style={{
                      position: 'absolute',
                      right: 0,
                      top: 0,
                      width: 6,
                      height: '100%',
                      cursor: 'ew-resize',
                      background: '#fff',
                    }}
                    onPointerDown={e => handleTrim(e, c, 'right')}
                  />
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
