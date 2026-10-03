// @vitest-environment jsdom
//
// The wiring between the Timeline and the engine and trim math, which have
// their own tests in plain Node: what Play, Stop, leaving the editor and a
// drag on a trim handle actually do.
import { act } from 'react';
import { type Root, createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { get } from 'idb-keyval';
import useProjectStore from '../store/useProjectStore';
import type { Clip } from '../types/timeline';
import Timeline from './Timeline';

(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

// jsdom has no IndexedDB. Each test says what the cache holds.
vi.mock('idb-keyval', () => ({ get: vi.fn(), set: vi.fn(async () => {}) }));

// -- a fake Web Audio graph, just enough to play through -------------------------

/** Every source made in the current test, oldest first. */
let sources: FakeSource[] = [];
/** Every context made in this file. Not reset: the engine is the page's. */
let contextsMade = 0;
/** Every resume() of any context in this file. Not reset either, for the same reason. */
let resumes = 0;

class FakeNode {
  connect<T>(node: T): T {
    return node;
  }
  disconnect() {}
}

class FakeSource extends FakeNode {
  buffer: unknown = null;
  onended: (() => void) | null = null;
  started = false;
  stopped = false;
  start() {
    this.started = true;
  }
  stop() {
    this.stopped = true;
    queueMicrotask(() => this.onended?.());
  }
}

class FakeContext {
  currentTime = 0;
  destination = new FakeNode();
  constructor() {
    contextsMade += 1;
  }
  async resume() {
    resumes += 1;
  }
  async decodeAudioData() {
    return { duration: 10 };
  }
  createBufferSource() {
    const s = new FakeSource();
    sources.push(s);
    return s;
  }
  createGain() {
    const param = { setValueAtTime() {}, linearRampToValueAtTime() {} };
    return Object.assign(new FakeNode(), { gain: param });
  }
}

/** Cached audio. jsdom's Blob has no arrayBuffer(), which is all the engine asks of it. */
const AUDIO = { arrayBuffer: async () => new ArrayBuffer(8) } as unknown as Blob;

// -----------------------------------------------------------------------------

const initialStore = useProjectStore.getState();

function clip(extra: Partial<Clip> = {}): Clip {
  return {
    id: 'c1',
    trackId: 'track-voice',
    blobId: 'b1',
    label: 'Who goes there?',
    start: 2,
    offset: 0,
    duration: 4,
    sourceDuration: 4,
    ...extra,
  };
}

function withClips(...clips: Clip[]) {
  useProjectStore.setState(s => ({ project: { ...s.project, clips } }));
}

const storedClip = () => useProjectStore.getState().project.clips[0];

// The engine is the page's, shared by every test here, so each Timeline is
// unmounted afterwards: that stops anything a test left playing.
let roots: Root[] = [];

async function renderTimeline() {
  const host = document.body.appendChild(document.createElement('div'));
  const root = createRoot(host);
  roots.push(root);
  await act(async () => root.render(<Timeline />));
  return { host, root };
}

function button(host: HTMLElement, text: string): HTMLButtonElement {
  const el = [...host.querySelectorAll('button')].find(b => b.textContent === text);
  if (!el) throw new Error(`no "${text}" button`);
  return el;
}

/** Click a button, then let the engine get as far as it can: loading, decoding, scheduling. */
async function click(host: HTMLElement, text: string) {
  await act(async () => {
    button(host, text).click();
    await new Promise(resolve => setTimeout(resolve, 0));
  });
}

/**
 * Drag `handle` sideways by `dx` pixels, stopping short of letting go.
 * React listens for pointerdown; the drag itself is followed on document.
 */
async function dragBy(handle: Element, dx: number) {
  await act(async () => {
    handle.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, clientX: 500 }));
    document.dispatchEvent(new MouseEvent('pointermove', { clientX: 500 + dx }));
  });
}

async function letGo() {
  await act(async () => document.dispatchEvent(new MouseEvent('pointerup')));
}

beforeEach(() => {
  sources = [];
  vi.stubGlobal('AudioContext', FakeContext);
  vi.mocked(get).mockResolvedValue(AUDIO);
});

afterEach(() => {
  for (const root of roots) act(() => root.unmount());
  roots = [];
  vi.unstubAllGlobals();
  vi.mocked(get).mockReset();
  document.body.innerHTML = '';
  useProjectStore.setState(initialStore, true);
});

describe('Timeline playback', () => {
  // The playing flag was cleared by a timer armed after play() returned, so
  // when play() threw it stayed set: Play disabled until a reload, and the
  // error nowhere on screen.
  it('gives Play back and shows why when a clip cannot be played', async () => {
    vi.mocked(get).mockResolvedValue(undefined); // evicted, and no server copy
    // A blobId no other test plays. The engine is the page's and keeps what
    // it has decoded, so after any test that played b1 it would never look
    // in the emptied cache, and this passed or failed by test order.
    withClips(clip({ blobId: 'never-played' }));
    const { host } = await renderTimeline();

    await click(host, 'Play');

    expect(button(host, 'Play').disabled).toBe(false);
    expect(host.querySelector('.error')?.textContent).toContain('audio missing');
  });

  // The sources were never kept, so nothing could stop them once started.
  it('Stop silences what Play started and gives Play back', async () => {
    withClips(clip(), clip({ id: 'c2', blobId: 'b2', start: 7 }));
    const { host } = await renderTimeline();

    await click(host, 'Play');
    expect(sources).toHaveLength(2);
    expect(sources.every(s => s.started)).toBe(true);
    expect(button(host, 'Play').disabled).toBe(true);

    await click(host, 'Stop');
    expect(sources.every(s => s.stopped)).toBe(true);
    expect(button(host, 'Play').disabled).toBe(false);
  });

  // The engine, and its AudioContext, used to be made per Timeline mount, so
  // every visit to the editor opened another context and none was closed.
  it('plays every visit to the editor through one AudioContext', async () => {
    withClips(clip());
    for (let visit = 0; visit < 3; visit++) {
      const { host, root } = await renderTimeline();
      await click(host, 'Play');
      act(() => root.unmount());
    }
    expect(contextsMade).toBe(1);
    expect(sources).toHaveLength(3);
  });

  it('stops playing when the editor is left', async () => {
    withClips(clip());
    const { host, root } = await renderTimeline();
    await click(host, 'Play');

    act(() => root.unmount());
    expect(sources.map(s => s.stopped)).toEqual([true]);
  });

  // Browsers only let a page resume audio during a user gesture, which ends
  // at the first await. Deferred past it, Play could be silent with no error.
  // Every Play, not just the first: a browser can suspend the context again
  // later (iOS Safari does on an interruption), and only a click resumes it.
  it('resumes the AudioContext inside every Play click itself', async () => {
    withClips(clip());
    const { host } = await renderTimeline();

    for (let press = 0; press < 2; press++) {
      // Counted from before the click: the context is the page's, and earlier
      // tests have resumed it already.
      const before = resumes;
      let resumedInClick = false;
      await act(async () => {
        button(host, 'Play').click();
        resumedInClick = resumes > before; // nothing awaited since the click
        await new Promise(resolve => setTimeout(resolve, 0));
      });
      expect(resumedInClick).toBe(true);
      await click(host, 'Stop');
    }
  });

  // The cache fallback was only tested inside loadClipAudio. The Timeline is
  // where it has to be wired in: with a loader that reads IndexedDB alone,
  // an evicted clip was "audio missing" for good.
  it('plays a clip whose cached copy is gone from the server copy', async () => {
    vi.mocked(get).mockResolvedValue(undefined);
    const asked: string[] = [];
    vi.stubGlobal('fetch', async (url: string) => {
      asked.push(url);
      return { ok: true, status: 200, statusText: 'OK', blob: async () => AUDIO } as unknown as Response;
    });
    // A blobId no other test plays, so the engine has not decoded it already.
    withClips(clip({ blobId: 'evicted', audioId: 'a1' }));
    const { host } = await renderTimeline();

    await click(host, 'Play');

    expect(asked).toEqual(['/api/v1/audio/a1']);
    expect(sources).toHaveLength(1);
    expect(sources[0].started).toBe(true);
    expect(host.querySelector('.error')).toBeNull();
  });
});

describe('Timeline trim handles', () => {
  // 100 px is 1 s. The left edge used to stay put while offset moved, so the
  // clip's audio slid 1 s earlier in time instead of losing its first second.
  it('a left trim moves the start with the offset', async () => {
    withClips(clip({ start: 2, offset: 0, duration: 4 }));
    const { host } = await renderTimeline();
    const el = host.querySelector<HTMLElement>('.clip')!;

    await dragBy(host.querySelector('.left-handle')!, 100);
    // The edge follows the pointer during the drag, not just once it lands.
    expect([el.style.left, el.style.width]).toEqual(['300px', '300px']);

    await letGo();
    expect(storedClip()).toMatchObject({ start: 3, offset: 1, duration: 3 });
  });

  // The handles could only ever shrink a clip: a trim could not be undone.
  it('a right trim extends back out as far as the source goes', async () => {
    withClips(clip({ start: 0, offset: 0, duration: 2, sourceDuration: 5 }));
    const { host } = await renderTimeline();

    await dragBy(host.querySelector('.right-handle')!, 1000);
    await letGo();

    expect(storedClip()).toMatchObject({ start: 0, offset: 0, duration: 5 });
  });

  it("shows the clip's label", async () => {
    withClips(clip());
    const { host } = await renderTimeline();
    expect(host.querySelector('.clip')?.textContent).toBe('Who goes there?');
  });
});
