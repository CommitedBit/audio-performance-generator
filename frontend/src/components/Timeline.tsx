import { useMemo, useRef } from 'react';
import useProjectStore, { orderedClips } from '../store/useProjectStore';
import type { TrackType } from '../types/timeline';
import { getBlob } from '../hooks/useIndexedAudio';
import { AudioEngine } from '../engine/AudioEngine';

const PIXELS_PER_SEC = 100;

const colors: Record<TrackType, string> = {
  voice: '#3b82f6', // blue-500
  sfx: '#f59e0b',   // amber-500
  music: '#10b981', // emerald-500
};

export default function Timeline() {
  const project = useProjectStore(s => s.project);
  const rawClips = useProjectStore(s => s.project.clips);
  const clips = useMemo(() => orderedClips(rawClips), [rawClips]);
  const updateClip = useProjectStore(s => s.updateClip);
  const playing = useProjectStore(s => s.playing);
  const setPlaying = useProjectStore(s => s.setPlaying);

  const engineRef = useRef<AudioEngine | null>(null);
  if (!engineRef.current) engineRef.current = new AudioEngine(getBlob);

  const play = async () => {
    if (playing) return;
    setPlaying(true);
    await engineRef.current!.play(clips);
    const last = clips.reduce((m, c) => Math.max(m, c.start + c.duration), 0);
    setTimeout(() => setPlaying(false), last * 1000);
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

  const handleTrim = (
    e: React.PointerEvent<HTMLDivElement>,
    clipId: string,
    side: 'left' | 'right',
    origWidth: number,
    origOffset: number
  ) => {
    e.stopPropagation();
    e.preventDefault();
    const startX = e.clientX;
    let newWidth = origWidth;
    let newOffset = origOffset;
    const move = (ev: PointerEvent) => {
      const deltaPx = ev.clientX - startX;
      if (side === 'left') {
        const deltaSec = Math.max(0, deltaPx / PIXELS_PER_SEC);
        newWidth = Math.max(0.1, origWidth - deltaSec);
        newOffset = origOffset + (origWidth - newWidth);
      } else {
        const deltaSec = Math.min(0, deltaPx / PIXELS_PER_SEC);
        newWidth = Math.max(0.1, origWidth + deltaSec);
      }
      const el = (e.target as HTMLElement).parentElement as HTMLElement;
      el.style.width = `${newWidth * PIXELS_PER_SEC}px`;
    };
    const up = () => {
      document.removeEventListener('pointermove', move);
      document.removeEventListener('pointerup', up);
      const patch = side === 'left'
        ? { offset: newOffset, duration: newWidth }
        : { duration: newWidth };
      updateClip(clipId, patch);
    };
    document.addEventListener('pointermove', move);
    document.addEventListener('pointerup', up);
  };

  return (
    <div className="panel">
      <div className="row">
        <button onClick={play} disabled={playing || clips.length === 0}>Play</button>
        <span className="muted small">
          {clips.length} clip{clips.length === 1 ? '' : 's'}
        </span>
      </div>
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
                    onPointerDown={e => handleTrim(e, c.id, 'left', c.duration, c.offset)}
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
                    onPointerDown={e => handleTrim(e, c.id, 'right', c.duration, c.offset)}
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
