import { describe, expect, it } from 'vitest';
import type { Clip } from '../types/timeline';
import { scheduleClip } from './schedule';

function clip(extra: Partial<Clip> = {}): Clip {
  return { id: 'c1', trackId: 'track-voice', blobId: 'b1', start: 2, offset: 1, duration: 4, ...extra };
}

describe('scheduleClip', () => {
  it('starts the clip at t0 + start and plays offset to offset + duration', () => {
    expect(scheduleClip(clip(), 10, 8)).toMatchObject({ when: 12, offset: 1, duration: 4 });
  });

  // Every clip made before gain and fades existed has neither field.
  it('plays at unity with no fades when gain and fades are absent', () => {
    expect(scheduleClip(clip(), 10, 8)?.gain).toEqual([{ kind: 'set', value: 1, time: 12 }]);
  });

  it("holds the clip's gain", () => {
    expect(scheduleClip(clip({ gain: 0.5 }), 10, 8)?.gain).toEqual([{ kind: 'set', value: 0.5, time: 12 }]);
  });

  it('ramps a fade-in up from silence to the gain', () => {
    expect(scheduleClip(clip({ gain: 0.5, fadeIn: 1.5 }), 10, 8)?.gain).toEqual([
      { kind: 'set', value: 0, time: 12 },
      { kind: 'ramp', value: 0.5, time: 13.5 },
    ]);
  });

  it('ramps a fade-out down to silence exactly as the clip ends', () => {
    expect(scheduleClip(clip({ gain: 0.5, fadeOut: 1 }), 10, 8)?.gain).toEqual([
      { kind: 'set', value: 0.5, time: 12 },
      { kind: 'set', value: 0.5, time: 15 },
      { kind: 'ramp', value: 0, time: 16 },
    ]);
  });

  // Trimmed to 2 s after 3 s of fade-in and 1 s of fade-out were set.
  it('shrinks fades longer than the clip in proportion, so they meet', () => {
    const plan = scheduleClip(clip({ duration: 2, fadeIn: 3, fadeOut: 1 }), 10, 8)!;
    expect(plan.gain).toEqual([
      { kind: 'set', value: 0, time: 12 },
      { kind: 'ramp', value: 1, time: 13.5 },
      { kind: 'set', value: 1, time: 13.5 },
      { kind: 'ramp', value: 0, time: 14 },
    ]);
    const times = plan.gain.map(p => p.time);
    expect(times).toEqual([...times].sort((a, b) => a - b));
  });

  // The first Play puts t0 at 0.1. With fades that meet, end - fadeOut came
  // out an ulp before the fade-in's end, Web Audio sorted the fade-out's pin
  // ahead of the fade-in's ramp, and the fade-in was silence then a click.
  it('never pins the fade-out before the fade-in has ended, whatever the rounding', () => {
    const times = (c: Partial<Clip>, t0: number) => scheduleClip(clip(c), t0, 100)!.gain.map(p => p.time);
    const ascending = (ts: number[]) => ts.every((t, i) => i === 0 || t >= ts[i - 1]);

    expect(ascending(times({ start: 0, offset: 0, duration: 0.7, fadeIn: 0.3, fadeOut: 0.4 }, 0.1))).toBe(true);

    // Fades that meet exactly, and fades long enough to be scaled, across a
    // spread of start times.
    const shapes = [
      { duration: 1, fadeIn: 0.5, fadeOut: 0.5 },
      { duration: 0.7, fadeIn: 0.3, fadeOut: 0.4 },
      { duration: 6.92, fadeIn: 1.38, fadeOut: 9.51 },
    ];
    for (const shape of shapes) {
      for (let t0 = 0; t0 < 50; t0 += 0.01) {
        const ts = times({ ...shape, start: 0, offset: 0 }, t0);
        if (!ascending(ts)) expect.fail(`t0 ${t0} ${JSON.stringify(shape)}: ${ts.join(', ')}`);
      }
    }
  });

  // The clip was sized from the server's measurement; the browser decoded
  // less. Its fade-out has to end where the audio really does.
  it('clamps the clip to the decoded audio, fade-out included', () => {
    const plan = scheduleClip(clip({ offset: 1, duration: 4, fadeOut: 1 }), 10, 3.5)!;
    expect(plan).toMatchObject({ when: 12, offset: 1, duration: 2.5 });
    expect(plan.gain.at(-1)).toEqual({ kind: 'ramp', value: 0, time: 14.5 });
  });

  it('schedules nothing when the offset is past the end of the decoded audio', () => {
    expect(scheduleClip(clip({ offset: 5 }), 10, 4)).toBeNull();
  });
});
