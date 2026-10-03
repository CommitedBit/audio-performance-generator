import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { Clip } from '../types/timeline';
import { AudioEngine, runPlayback } from './AudioEngine';
import { scheduleClip } from './schedule';

// -- a fake Web Audio graph -----------------------------------------------------
//
// Node has no AudioContext. The fake records what the engine asks of it and
// keeps a clock that only moves when a test says so, so a slow decode can be
// made to cost time.

/** Seconds of context time each decodeAudioData call takes. */
let decodeCost = 0;

class FakeNode {
  outputs: unknown[] = [];
  connect<T>(node: T): T {
    this.outputs.push(node);
    return node;
  }
  disconnect() {
    this.outputs = [];
  }
}

class FakeParam {
  events: [string, number, number][] = [];
  setValueAtTime(value: number, time: number) {
    this.events.push(['set', value, time]);
  }
  linearRampToValueAtTime(value: number, time: number) {
    this.events.push(['ramp', value, time]);
  }
}

class FakeGain extends FakeNode {
  gain = new FakeParam();
}

class FakeSource extends FakeNode {
  ctx: FakeContext;
  buffer: { duration: number } | null = null;
  onended: (() => void) | null = null;
  /** `now` is the context clock when start() was called. */
  started: { when: number; offset: number; duration: number; now: number } | null = null;
  stopped = false;
  private ended = false;
  constructor(ctx: FakeContext) {
    super();
    this.ctx = ctx;
  }
  start(when: number, offset: number, duration: number) {
    this.started = { when, offset, duration, now: this.ctx.currentTime };
  }
  stop() {
    this.stopped = true;
    this.finish();
  }
  /** The audio has run out (or been stopped): 'ended' fires, once and asynchronously, as in a browser. */
  finish() {
    if (this.ended) return;
    this.ended = true;
    queueMicrotask(() => this.onended?.());
  }
  /** When the audio really begins: a time already past means "now". */
  get sounds() {
    return Math.max(this.started!.when, this.started!.now);
  }
}

class FakeContext {
  static made: FakeContext[] = [];
  currentTime = 0;
  destination = new FakeNode();
  sources: FakeSource[] = [];
  resume = vi.fn(async () => {});
  constructor() {
    FakeContext.made.push(this);
  }
  /** The "audio" is its own length in seconds, as text. */
  async decodeAudioData(data: ArrayBuffer) {
    this.currentTime += decodeCost;
    return { duration: Number(new TextDecoder().decode(data)) };
  }
  createBufferSource() {
    const s = new FakeSource(this);
    this.sources.push(s);
    return s;
  }
  createGain() {
    return new FakeGain();
  }
}

const ctx = () => FakeContext.made[0];

/** Every scheduled clip runs out of audio. */
const finishAll = () => ctx().sources.forEach(s => s.finish());

function clip(extra: Partial<Clip> = {}): Clip {
  return { id: 'c1', trackId: 'track-voice', blobId: 'b1', start: 0, offset: 0, duration: 2, ...extra };
}

/** Every clip's source is 10 s long. */
const tenSeconds = vi.fn(async () => new Blob(['10']));

/** Let the engine get as far as it can: loading, decoding, scheduling. */
const settle = () => new Promise(resolve => setTimeout(resolve, 0));

beforeEach(() => {
  FakeContext.made = [];
  decodeCost = 0;
  vi.stubGlobal('AudioContext', FakeContext);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

// -- AudioContext ---------------------------------------------------------------

describe('the AudioContext', () => {
  // One made at page load is born suspended under autoplay policy.
  it('is not made until the first play()', () => {
    new AudioEngine(tenSeconds);
    expect(FakeContext.made).toHaveLength(0);
  });

  // Browsers only let a page start audio during a user gesture. The old
  // engine never resumed its context, so Play could be silent with no error.
  it('is resumed before play() first yields, while still inside the click', () => {
    const engine = new AudioEngine(tenSeconds);
    void engine.play([clip()]);
    expect(ctx().resume).toHaveBeenCalledOnce();
    engine.stop();
  });

  // Resumed by every play(), not only the one that made it: a browser can
  // suspend the context again later (iOS Safari does on an interruption), and
  // only a Play click is allowed to bring it back.
  it('is the same one for every play(), and every play() resumes it', async () => {
    const engine = new AudioEngine(tenSeconds);
    for (let i = 0; i < 3; i++) {
      const done = engine.play([clip()]);
      expect(ctx().resume).toHaveBeenCalledTimes(i + 1);
      await settle();
      finishAll();
      await done;
    }
    expect(FakeContext.made).toHaveLength(1);
    expect(ctx().sources).toHaveLength(3);
  });
});

// -- scheduling -----------------------------------------------------------------

describe('play', () => {
  // Voice and a music bed both at 0, and an effect 0.5 s in. Decoding takes
  // 1 s each; the old engine read the clock before decoding, so the bed came
  // in a second after the voice.
  it('decodes everything before reading the clock, so a slow decode delays all clips together', async () => {
    decodeCost = 1;
    const clips = [
      clip({ id: 'voice', blobId: 'v' }),
      clip({ id: 'bed', blobId: 'm' }),
      clip({ id: 'door', blobId: 'd', start: 0.5 }),
    ];
    const engine = new AudioEngine(tenSeconds);
    void engine.play(clips);
    await settle();

    const [voice, bed, door] = ctx().sources;
    for (const s of ctx().sources) expect(s.started!.when).toBeGreaterThan(s.started!.now);
    expect(bed.sounds).toBe(voice.sounds);
    expect(door.sounds - voice.sounds).toBeCloseTo(0.5);
    engine.stop();
  });

  // The old engine scheduled each clip as it loaded, so the clips before a
  // missing one played on under the error.
  it("schedules nothing when any clip's audio cannot be loaded", async () => {
    const load = vi.fn(async (c: Clip) => {
      if (c.blobId === 'gone') throw new Error('audio missing');
      return new Blob(['10']);
    });
    const engine = new AudioEngine(load);
    await expect(engine.play([clip(), clip({ id: 'c2', blobId: 'gone' })])).rejects.toThrow('audio missing');
    expect(ctx().sources).toHaveLength(0);
  });

  it('tries a failed load again on the next play()', async () => {
    const load = vi.fn(async () => new Blob(['10'])).mockRejectedValueOnce(new Error('server unreachable'));
    const engine = new AudioEngine(load);
    await expect(engine.play([clip()])).rejects.toThrow('server unreachable');

    const done = engine.play([clip()]);
    await settle();
    expect(ctx().sources).toHaveLength(1);
    finishAll();
    await done;
  });

  it('decodes a source shared by two clips once', async () => {
    const engine = new AudioEngine(tenSeconds);
    tenSeconds.mockClear();
    const done = engine.play([clip(), clip({ id: 'c2', start: 3 })]);
    await settle();
    expect(tenSeconds).toHaveBeenCalledOnce();
    expect(ctx().sources).toHaveLength(2);
    finishAll();
    await done;
  });

  // The buffers are decoded all together and matched back to their clips by
  // index. Off by one, or the plan's offset and duration dropped on the way
  // to start(), and a clip plays the wrong audio, or ignores its trims: a
  // left-trimmed clip would play its source from 0.
  it("starts each clip's own audio at the clip's offset, for the clip's duration", async () => {
    const lengths: Record<string, number> = { a: 5, b: 9 };
    const load = async (c: Clip) => new Blob([String(lengths[c.blobId])]);
    const clips = [
      clip({ id: 'a', blobId: 'a', start: 0, offset: 1, duration: 2 }),
      // 4 s from 3 s in: all of it from b's 9 s, but only 2 s of a's 5 s.
      clip({ id: 'b', blobId: 'b', start: 1, offset: 3, duration: 4 }),
    ];
    const engine = new AudioEngine(load);
    void engine.play(clips);
    await settle();

    const [a, b] = ctx().sources;
    expect(a.buffer).toEqual({ duration: 5 });
    expect(b.buffer).toEqual({ duration: 9 });
    const t0 = a.started!.when;
    expect(a.started).toMatchObject({ when: t0, offset: 1, duration: 2 });
    expect(b.started).toMatchObject({ when: t0 + 1, offset: 3, duration: 4 });
    engine.stop();
  });

  it("sends each clip through its own GainNode carrying the clip's gain and fades", async () => {
    const faded = clip({ start: 1, duration: 4, gain: 0.5, fadeIn: 0.5, fadeOut: 1 });
    const engine = new AudioEngine(tenSeconds);
    void engine.play([faded, clip({ id: 'c2' })]);
    await settle();

    const [a, b] = ctx().sources;
    const gain = a.outputs[0] as FakeGain;
    expect(gain).toBeInstanceOf(FakeGain);
    expect(gain.outputs).toEqual([ctx().destination]);
    expect(b.outputs[0]).not.toBe(gain);

    const t0 = a.started!.when - faded.start;
    const plan = scheduleClip(faded, t0, 10)!;
    expect(gain.gain.events).toEqual(plan.gain.map(p => [p.kind, p.value, p.time]));
    expect(gain.gain.events.at(-1)).toEqual(['ramp', 0, a.started!.when + 4]);
    engine.stop();
  });

  // The playing flag follows this promise, so it must not settle while
  // anything is still sounding.
  it('settles only when the last clip has ended', async () => {
    const engine = new AudioEngine(tenSeconds);
    let settled = false;
    const done = engine.play([clip(), clip({ id: 'c2', blobId: 'b2', start: 1 })]).then(() => (settled = true));
    await settle();
    const [a, b] = ctx().sources;

    a.finish();
    await settle();
    expect(settled).toBe(false);

    b.finish();
    await done;
    expect(settled).toBe(true);
  });
});

// -- stop -----------------------------------------------------------------------

describe('stop', () => {
  it('stops every scheduled source and settles play()', async () => {
    const engine = new AudioEngine(tenSeconds);
    const done = engine.play([clip(), clip({ id: 'c2', blobId: 'b2', start: 5 })]);
    await settle();
    expect(ctx().sources).toHaveLength(2);

    engine.stop();
    for (const s of ctx().sources) expect(s.stopped).toBe(true);
    await expect(done).resolves.toBeUndefined();
  });

  // A server fetch for an evicted clip can take a while. Stop pressed
  // meanwhile must win: nothing may start once it lands.
  it('pressed while audio is still loading, starts nothing', async () => {
    let release = () => {};
    const slow = () => new Promise<Blob>(resolve => (release = () => resolve(new Blob(['10']))));
    const engine = new AudioEngine(slow);
    const done = engine.play([clip()]);
    await settle();

    engine.stop();
    await done;
    release();
    await settle();
    expect(ctx().sources).toHaveLength(0);
  });

  it('is what a second play() does to the first, and the second still plays', async () => {
    const engine = new AudioEngine(tenSeconds);
    const first = engine.play([clip()]);
    await settle();
    void engine.play([clip({ id: 'c2', blobId: 'b2' })]);
    await expect(first).resolves.toBeUndefined();
    expect(ctx().sources[0].stopped).toBe(true);

    // The first play()'s cleanup must leave the second alone: stopping
    // whatever is current, it stopped the second while it was still loading,
    // and that one settled having played nothing.
    await settle();
    expect(ctx().sources).toHaveLength(2);
    expect(ctx().sources[1].stopped).toBe(false);
    engine.stop();
  });
});

// -- runPlayback ----------------------------------------------------------------

describe('runPlayback', () => {
  it('sets playing while play() runs and clears it when it settles', async () => {
    const setPlaying = vi.fn();
    let finish = () => {};
    const done = runPlayback(() => new Promise<void>(resolve => (finish = resolve)), setPlaying);
    expect(setPlaying.mock.calls).toEqual([[true]]);
    finish();
    await done;
    expect(setPlaying.mock.calls).toEqual([[true], [false]]);
  });

  // It used to stay set forever, and the Play button with it.
  it('clears playing when play() throws', async () => {
    const setPlaying = vi.fn();
    await expect(
      runPlayback(() => Promise.reject(new Error('audio missing')), setPlaying)
    ).rejects.toThrow('audio missing');
    expect(setPlaying).toHaveBeenLastCalledWith(false);
  });

  // An await before play() would end the click's user activation first.
  it('calls play() before awaiting anything', () => {
    const play = vi.fn(async () => {});
    void runPlayback(play, () => {});
    expect(play).toHaveBeenCalledOnce();
  });
});
