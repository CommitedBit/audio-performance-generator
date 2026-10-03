import type { Clip } from '../types/timeline';

/** One step of a clip's gain automation, applied in order. */
export interface GainPoint {
  /** 'set' jumps to `value` at `time`; 'ramp' moves linearly to it from the point before. */
  kind: 'set' | 'ramp';
  value: number;
  time: number;
}

/** When and how one clip plays, in AudioContext time. */
export interface ClipSchedule {
  /** context time the clip starts sounding */
  when: number;
  /** seconds into the source */
  offset: number;
  /** seconds of the source to play */
  duration: number;
  gain: GainPoint[];
}

/**
 * Where a clip lands on the context clock when the timeline's 0 is at `t0`.
 *
 * `sourceLength` is the decoded buffer's length. The clip is clamped to it:
 * the clip was sized from the server's measurement, which need not match
 * what this browser decodes to the sample, and a fade-out timed past the real
 * end of the audio is never heard.
 *
 * Returns null when nothing of the clip is left to play.
 */
export function scheduleClip(clip: Clip, t0: number, sourceLength: number): ClipSchedule | null {
  const offset = Math.min(Math.max(clip.offset, 0), sourceLength);
  const duration = Math.min(clip.duration, sourceLength - offset);
  if (!(duration > 0)) return null;

  const when = t0 + clip.start;
  const end = when + duration;
  const level = Math.max(clip.gain ?? 1, 0);
  let fadeIn = Math.max(clip.fadeIn ?? 0, 0);
  let fadeOut = Math.max(clip.fadeOut ?? 0, 0);
  // Fades longer than the clip (it was trimmed after they were set) shrink in
  // proportion until they meet. Left to overlap, the fade-out would begin
  // before the fade-in ends, and Web Audio sorts automation by time, so the
  // two ramps would interleave into neither shape.
  if (fadeIn + fadeOut > duration) {
    const scale = duration / (fadeIn + fadeOut);
    fadeIn *= scale;
    fadeOut *= scale;
  }

  const gain: GainPoint[] =
    fadeIn > 0
      ? [
          { kind: 'set', value: 0, time: when },
          { kind: 'ramp', value: level, time: when + fadeIn },
        ]
      : [{ kind: 'set', value: level, time: when }];
  if (fadeOut > 0) {
    // Pinned at full level first: a ramp runs from the point before it, so
    // without this the fade-out would start where the fade-in ended (or at
    // `when`) and stretch across the whole clip.
    //
    // Never before the fade-in's end. Where the fades meet, `end - fadeOut`
    // can round to an ulp below `when + fadeIn` (t0 0.1, 0.7 s clip, fades
    // 0.3 and 0.4 gives 0.3999999999999999 against 0.4), and Web Audio sorts
    // the pin ahead of the fade-in's ramp: the gain then holds 0 for the
    // whole fade-in and jumps to full, a click where the fade should be.
    gain.push(
      { kind: 'set', value: level, time: Math.max(end - fadeOut, when + fadeIn) },
      { kind: 'ramp', value: 0, time: end }
    );
  }
  return { when, offset, duration, gain };
}
