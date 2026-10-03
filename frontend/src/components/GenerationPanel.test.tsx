// @vitest-environment jsdom
//
// The only test file with a DOM: the panel has to be rendered and unmounted.
// The client behaviour it relies on is tested in plain Node in client.test.ts.
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { JobInfo, ModelsResponse } from '../api/client';
import GenerationPanel from './GenerationPanel';

// Tells React it is under test, so it flushes work inside act() and does not
// warn about updates made there.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const MODELS: ModelsResponse = {
  device: 'cuda',
  providers: [
    {
      id: 'ace-step',
      name: 'ACE-Step',
      capability: 'music',
      license: 'Apache-2.0',
      requires_gpu: true,
      remote: false,
      description: '',
      available: true,
      unavailable_reason: '',
      loaded: true,
      voices: [],
      params: [],
    },
  ],
  defaults: { voice: null, music: 'ace-step', sfx: null },
};

const QUEUED: JobInfo = {
  id: 'music:j1',
  kind: 'music',
  status: 'queued',
  progress: 0,
  message: '',
  audio_id: null,
  audio_url: null,
  error: null,
  queue_position: 1,
  meta: {},
};

const JOB_URL = '/api/v1/jobs/music:j1';

function json(body: unknown): Response {
  return new Response(JSON.stringify(body), { headers: { 'Content-Type': 'application/json' } });
}

/** A server whose music job stays queued; records each request as "METHOD url". */
function queuedServer(): string[] {
  const calls: string[] = [];
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init: RequestInit = {}) => {
    // A real fetch never sends a request whose signal has already aborted.
    if (init.signal?.aborted) throw init.signal.reason;
    const call = `${init.method ?? 'GET'} ${String(input)}`;
    calls.push(call);
    if (call === 'GET /api/v1/models') return json(MODELS);
    if (init.method === 'DELETE') return json({ ...QUEUED, status: 'cancelled' });
    return json(QUEUED); // the POST, and every poll
  });
  return calls;
}

function byText<T extends HTMLElement>(host: HTMLElement, selector: string, text: string): T {
  const el = [...host.querySelectorAll<T>(selector)].find(e => e.textContent?.trim() === text);
  if (!el) throw new Error(`no ${selector} reading "${text}"`);
  return el;
}

/** Type into a React-controlled field: React only sees a value set through the prototype's setter. */
function typeInto(field: HTMLTextAreaElement, value: string) {
  Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')!.set!.call(field, value);
  field.dispatchEvent(new Event('input', { bubbles: true }));
}

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

describe('GenerationPanel', () => {
  // Leaving the editor mid-generation used to stop the polling only, and the
  // queued job still took its turn on the GPU for audio nobody would collect.
  it('cancels the server job when it unmounts mid-generation', async () => {
    vi.useFakeTimers();
    const calls = queuedServer();
    const host = document.body.appendChild(document.createElement('div'));
    const root = createRoot(host);

    await act(async () => root.render(<GenerationPanel />));
    await act(() => vi.advanceTimersByTimeAsync(0)); // the model list arrives

    await act(async () => byText<HTMLButtonElement>(host, 'button', 'Music').click());
    await act(async () => typeInto(host.querySelector('textarea')!, 'rain on a tin roof'));
    await act(async () => byText<HTMLButtonElement>(host, 'button', 'Generate Music').click());
    await act(() => vi.advanceTimersByTimeAsync(600)); // submitted, one poll done
    expect(calls).toContain('POST /api/v1/audio/music');
    expect(host.textContent).toContain('queued behind 1');

    act(() => root.unmount());
    await vi.advanceTimersByTimeAsync(30_000);
    expect(calls.filter(c => c.startsWith('DELETE '))).toEqual([`DELETE ${JOB_URL}`]);
    // And it stopped watching: no poll after the unmount.
    expect(calls.at(-1)).toBe(`DELETE ${JOB_URL}`);
  });
});
