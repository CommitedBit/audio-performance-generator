import type { Clip } from '../types/timeline';
import { scheduleClip } from './schedule';

/** A clip's audio, from wherever it is kept. Rejects when it cannot be had. */
export type LoadAudio = (clip: Clip) => Promise<Blob>;

/**
 * How far past "now" the timeline's 0 is put. The clock keeps running while
 * the clips are scheduled, so a clip due at exactly "now" would start late,
 * and out of step with the clips scheduled after it.
 */
const START_LEAD = 0.1;

interface Playback {
  voices: { source: AudioBufferSourceNode; gain: GainNode }[];
  /** settles the play() that started this playback */
  end: () => void;
}

export class AudioEngine {
  // One context for the engine's life, made by the first play() so that it
  // is born inside a click: a context made at page load starts suspended
  // under browser autoplay policy.
  private ctx: AudioContext | null = null;
  // blobId -> decoded audio. The promise is kept rather than the buffer, so
  // clips that share a source decode it once.
  private buffers = new Map<string, Promise<AudioBuffer>>();
  private loadAudio: LoadAudio;
  private current: Playback | null = null;

  // NOTE: written as an explicit field rather than a constructor parameter
  // property because tsconfig sets `erasableSyntaxOnly`, which forbids the
  // `constructor(private loadAudio: ...)` shorthand (it emits runtime code).
  constructor(loadAudio: LoadAudio) {
    this.loadAudio = loadAudio;
  }

  /**
   * Play `clips` from the top of the timeline, stopping whatever was playing.
   *
   * Resolves when the last clip has finished or stop() is called. Rejects,
   * having scheduled nothing, when any clip's audio cannot be loaded.
   *
   * Call it straight from the Play click, with no await before it: it resumes
   * the context, and browsers only allow that during a user gesture.
   */
  async play(clips: Clip[]): Promise<void> {
    const ctx = this.context();
    // Before the first await, so still inside the gesture. Never resumed, a
    // context made outside one stayed suspended, and Play was silent with no
    // error anywhere.
    void ctx.resume();
    this.stop();

    let end = () => {};
    const ended = new Promise<void>(resolve => (end = resolve));
    const playback: Playback = { voices: [], end };
    this.current = playback;

    try {
      // Every buffer is decoded before t0 is read. Decoded one at a time
      // inside the scheduling loop, t0 slipped into the past while later
      // clips decoded, and each of them started late by however long the
      // decodes before it had taken.
      const buffers = await Promise.race([
        Promise.all(clips.map(c => this.buffer(ctx, c))),
        ended,
      ]);
      if (!buffers) return; // stop() was called while loading

      const t0 = ctx.currentTime + START_LEAD;
      let remaining = 0;
      clips.forEach((clip, i) => {
        const plan = scheduleClip(clip, t0, buffers[i].duration);
        if (!plan) return;
        // A GainNode per clip carries its level and fades.
        const gain = ctx.createGain();
        for (const p of plan.gain) {
          if (p.kind === 'set') gain.gain.setValueAtTime(p.value, p.time);
          else gain.gain.linearRampToValueAtTime(p.value, p.time);
        }
        const source = ctx.createBufferSource();
        source.buffer = buffers[i];
        source.connect(gain).connect(ctx.destination);
        source.onended = () => {
          if (--remaining === 0) end();
        };
        source.start(plan.when, plan.offset, plan.duration);
        remaining += 1;
        playback.voices.push({ source, gain });
      });
      if (remaining > 0) await ended;
    } finally {
      if (this.current === playback) this.stop();
    }
  }

  /** Silence everything play() scheduled, and settle that play(). */
  stop(): void {
    const playback = this.current;
    if (!playback) return;
    this.current = null;
    // The sources are kept for this. A started source plays to its end
    // whatever becomes of the code that started it, so a Stop that has not
    // kept them has nothing to stop.
    for (const { source, gain } of playback.voices) {
      try {
        source.stop();
      } catch {
        /* already ended; older WebKit throws on stop() then. Silent either way. */
      }
      gain.disconnect();
    }
    playback.end();
  }

  private context(): AudioContext {
    this.ctx ??= new AudioContext();
    return this.ctx;
  }

  private buffer(ctx: AudioContext, clip: Clip): Promise<AudioBuffer> {
    let buffer = this.buffers.get(clip.blobId);
    if (!buffer) {
      buffer = this.loadAudio(clip)
        .then(blob => blob.arrayBuffer())
        .then(data => ctx.decodeAudioData(data));
      this.buffers.set(clip.blobId, buffer);
      // A failure is not remembered, so the next Play tries again: the
      // server it had to be fetched from may be back by then.
      const loading = buffer;
      loading.catch(() => {
        if (this.buffers.get(clip.blobId) === loading) this.buffers.delete(clip.blobId);
      });
    }
    return buffer;
  }
}

/**
 * Run one playback with `playing` set for exactly as long as it lasts.
 *
 * `play` is called before anything is awaited, so it still runs inside the
 * click that started it (see AudioEngine.play). The flag is cleared in a
 * finally: it used to be cleared by a timer armed after play() returned, so
 * a play() that threw -- audio missing -- left it set and the Play button
 * disabled until a reload.
 */
export async function runPlayback(
  play: () => Promise<void>,
  setPlaying: (playing: boolean) => void
): Promise<void> {
  setPlaying(true);
  try {
    await play();
  } finally {
    setPlaying(false);
  }
}
