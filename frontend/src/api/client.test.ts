import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  ApiError,
  type JobInfo,
  type JobStatus,
  cancelJob,
  fetchAudio,
  fetchAudioById,
  fetchModels,
  generate,
  generateAudio,
  getApiKey,
  getJob,
  setApiKey,
  uploadVoiceReference,
  waitForJob,
} from './client';

// -- fakes ---------------------------------------------------------------------

interface Call {
  url: string;
  method: string;
  headers: Headers;
  body: RequestInit['body'];
}

type Handler = (url: string, init: RequestInit) => Response | Promise<Response>;

/**
 * Replace fetch with `handler`, recording every request that reaches the
 * server. Like the real fetch, a request rejects with the signal's reason when
 * its signal aborts.
 */
function mockFetch(handler: Handler): Call[] {
  const calls: Call[] = [];
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input);
    const signal = init.signal;
    // A real fetch never sends a request whose signal is already aborted, so
    // it is not recorded: a cancel made with the spent signal must not count
    // as having reached the server.
    if (signal?.aborted) return Promise.reject(signal.reason);
    calls.push({ url, method: init.method ?? 'GET', headers: new Headers(init.headers), body: init.body });
    return new Promise<Response>((resolve, reject) => {
      signal?.addEventListener('abort', () => reject(signal.reason), { once: true });
      Promise.resolve()
        .then(() => handler(url, init))
        .then(resolve, reject);
    });
  });
  return calls;
}

function json(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { 'Content-Type': 'application/json' },
  });
}

/** What fetch does when nothing answers. */
function offline(): Promise<Response> {
  return Promise.reject(new TypeError('Failed to fetch'));
}

/** A request that never completes, so an abort can land mid-flight. */
function hang(): Promise<Response> {
  return new Promise(() => {});
}

/** Answer successive calls with each step in turn, repeating the last one. */
function sequence(...steps: (() => Response | Promise<Response>)[]): Handler {
  let i = 0;
  return () => steps[Math.min(i++, steps.length - 1)]();
}

function job(status: JobStatus, extra: Partial<JobInfo> = {}): JobInfo {
  return {
    id: 'music:j1',
    kind: 'music',
    status,
    progress: 0,
    message: '',
    audio_id: null,
    audio_url: null,
    error: null,
    queue_position: null,
    meta: {},
    ...extra,
  };
}

/** Track a promise's outcome without leaving a rejection unhandled while timers run. */
function outcome<T>(p: Promise<T>) {
  const o: { settled: boolean; value?: T; error?: unknown } = { settled: false };
  p.then(
    value => Object.assign(o, { settled: true, value }),
    error => Object.assign(o, { settled: true, error })
  );
  return o;
}

function memoryStorage(): Storage {
  const m = new Map<string, string>();
  return {
    get length() {
      return m.size;
    },
    clear: () => m.clear(),
    getItem: k => m.get(k) ?? null,
    key: i => [...m.keys()][i] ?? null,
    removeItem: k => void m.delete(k),
    setItem: (k, v) => void m.set(k, String(v)),
  };
}

const JOB_URL = '/api/v1/jobs/music:j1';
const polls = (calls: Call[]) => calls.filter(c => c.method === 'GET' && c.url === JOB_URL);
const cancels = (calls: Call[]) => calls.filter(c => c.method === 'DELETE' && c.url === JOB_URL);

beforeEach(() => {
  vi.stubGlobal('localStorage', memoryStorage());
  // The key also lives in module memory (the no-storage fallback); clear it
  // so one test's key never leaks into the next.
  setApiKey('');
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

// -- access key ----------------------------------------------------------------

describe('X-API-Key', () => {
  it('is not sent when no key is set', async () => {
    const calls = mockFetch(() => json({ device: 'cpu', providers: [], defaults: {} }));
    await fetchModels();
    expect(calls[0].headers.has('X-API-Key')).toBe(false);
  });

  it('is sent on every kind of request', async () => {
    setApiKey('s3cret');
    const calls = mockFetch(url =>
      url.includes('/v1/audio/abc') ? new Response(new Blob(['RIFF'])) : json(job('done'))
    );

    await fetchModels();
    await generate('music', { prompt: 'rain' });
    await getJob('music:j1');
    await cancelJob('music:j1');
    await fetchAudio(job('done', { audio_url: '/v1/audio/abc' }));
    await fetchAudioById('abc');

    expect(calls).toHaveLength(6);
    expect(calls[5].url).toBe('/api/v1/audio/abc');
    for (const c of calls) expect(c.headers.get('X-API-Key')).toBe('s3cret');
    // Adding the key must not drop the headers the request already had.
    expect(calls[1].headers.get('Content-Type')).toBe('application/json');
  });

  it('leaves multipart uploads for the browser to frame', async () => {
    setApiKey('s3cret');
    const calls = mockFetch(() => json({ id: 'v1', name: 'narrator', bytes: 4, created_at: 0 }));
    await uploadVoiceReference(new File(['RIFF'], 'narrator.wav'), 'narrator');

    expect(calls[0].body).toBeInstanceOf(FormData);
    expect(calls[0].headers.get('X-API-Key')).toBe('s3cret');
    // Setting Content-Type ourselves would lose the multipart boundary.
    expect(calls[0].headers.has('Content-Type')).toBe(false);
  });

  it('is kept for the page when storage is unavailable', async () => {
    const blocked = () => {
      throw new DOMException('denied', 'SecurityError');
    };
    vi.stubGlobal('localStorage', { getItem: blocked, setItem: blocked, removeItem: blocked });
    setApiKey('s3cret');
    const calls = mockFetch(() => json({ device: 'cpu', providers: [], defaults: {} }));
    await fetchModels();
    expect(calls[0].headers.get('X-API-Key')).toBe('s3cret');
  });

  it('stops being sent once cleared', async () => {
    setApiKey('s3cret');
    setApiKey('');
    const calls = mockFetch(() => json({ device: 'cpu', providers: [], defaults: {} }));
    await fetchModels();
    expect(calls[0].headers.has('X-API-Key')).toBe(false);
  });

  // Checked on the stored key: Headers strips the whitespace from a header
  // value by itself, so the request alone would pass without the trim.
  it('is stored trimmed', () => {
    setApiKey('  s3cret  ');
    expect(getApiKey()).toBe('s3cret');
  });

  // Untrimmed, a pasted blank went out as an empty X-API-Key, and the 401
  // blamed a key the user never entered instead of asking for one.
  it('treats a blank key as no key', async () => {
    setApiKey('   ');
    const calls = mockFetch(() => json({ detail: 'missing or invalid API key' }, 401));
    await expect(fetchModels()).rejects.toMatchObject({
      status: 401,
      message: 'this server requires an API key - enter it on the Server page',
    });
    expect(calls[0].headers.has('X-API-Key')).toBe(false);
  });
});

describe('401', () => {
  const unauthorized = () => json({ detail: 'missing or invalid API key' }, 401);

  it('asks for a key when none is set', async () => {
    mockFetch(unauthorized);
    await expect(fetchModels()).rejects.toMatchObject({
      name: 'ApiError',
      status: 401,
      message: 'this server requires an API key - enter it on the Server page',
    });
  });

  it('blames the key when one was sent', async () => {
    setApiKey('wrong');
    mockFetch(unauthorized);
    await expect(fetchModels()).rejects.toMatchObject({
      status: 401,
      message: 'the server rejected the API key - check it on the Server page',
    });
  });

  it('says the same when fetching audio', async () => {
    mockFetch(unauthorized);
    await expect(fetchAudio(job('done', { audio_url: '/v1/audio/abc' }))).rejects.toMatchObject({
      status: 401,
      message: 'this server requires an API key - enter it on the Server page',
    });
  });
});

// -- request errors ------------------------------------------------------------

describe('request errors', () => {
  it('names an unreachable server', async () => {
    mockFetch(offline);
    await expect(fetchModels()).rejects.toMatchObject({
      name: 'ApiError',
      status: 0,
      message: 'cannot reach the model server at /api - is it running?',
    });
  });

  it('surfaces FastAPI detail with the status', async () => {
    mockFetch(() => json({ detail: 'seconds must be at most 22' }, 422));
    await expect(generate('sfx', { prompt: 'door' })).rejects.toMatchObject({
      status: 422,
      message: 'seconds must be at most 22',
    });
  });

  it('keeps an abort an AbortError, not "cannot reach the server"', async () => {
    mockFetch(hang);
    const ac = new AbortController();
    const pending = getJob('music:j1', ac.signal);
    ac.abort();

    const err = await pending.catch((e: unknown) => e);
    expect(err).toBeInstanceOf(DOMException);
    expect((err as DOMException).name).toBe('AbortError');
    expect(err).not.toBeInstanceOf(ApiError);
  });
});

// -- waitForJob ----------------------------------------------------------------

describe('waitForJob', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('returns a finished job without polling', async () => {
    const calls = mockFetch(() => json(job('running')));
    const done = job('done', { audio_url: '/v1/audio/abc' });
    await expect(waitForJob(done)).resolves.toBe(done);
    expect(calls).toHaveLength(0);
  });

  it('polls until the job is done, reporting each update', async () => {
    const calls = mockFetch(
      sequence(
        () => json(job('queued', { queue_position: 1 })),
        () => json(job('running', { progress: 0.5 })),
        () => json(job('done', { audio_url: '/v1/audio/abc' }))
      )
    );
    const seen: JobStatus[] = [];
    const result = outcome(waitForJob(job('queued'), j => seen.push(j.status)));

    // The first poll waits rather than asking again straight away.
    await vi.advanceTimersByTimeAsync(499);
    expect(calls).toHaveLength(0);

    await vi.advanceTimersByTimeAsync(10_000);
    expect(result.value?.status).toBe('done');
    expect(result.value?.audio_url).toBe('/v1/audio/abc');
    expect(seen).toEqual(['queued', 'running', 'done']);
    expect(polls(calls)).toHaveLength(3);
  });

  it('turns a failed job into an ApiError carrying its message', async () => {
    mockFetch(() => json(job('error', { error: 'CUDA out of memory' })));
    const result = outcome(waitForJob(job('running')));
    await vi.advanceTimersByTimeAsync(1000);
    expect(result.error).toMatchObject({ name: 'ApiError', status: 500, message: 'CUDA out of memory' });
  });

  it('reports a job cancelled on the server', async () => {
    mockFetch(() => json(job('cancelled')));
    const result = outcome(waitForJob(job('queued')));
    await vi.advanceTimersByTimeAsync(1000);
    expect(result.error).toMatchObject({ status: 499, message: 'generation was cancelled' });
  });

  // One dropped poll used to fail a generation still running on the server.
  it('rides out a poll the network dropped', async () => {
    mockFetch(sequence(offline, () => json(job('running')), () => json(job('done'))));
    const result = outcome(waitForJob(job('queued')));
    await vi.advanceTimersByTimeAsync(30_000);
    expect(result.error).toBeUndefined();
    expect(result.value?.status).toBe('done');
  });

  it('rides out any 5xx: 500, 502, 503 and 504', async () => {
    const calls = mockFetch(
      sequence(
        () => new Response('Internal Server Error', { status: 500 }),
        () => json({ detail: 'cannot reach music' }, 502),
        () => new Response('<html>Service Unavailable</html>', { status: 503 }),
        () => json({ detail: 'music timed out after 5.0s' }, 504),
        () => json(job('done'))
      )
    );
    const result = outcome(waitForJob(job('running')));
    await vi.advanceTimersByTimeAsync(30_000);
    expect(result.error).toBeUndefined();
    expect(result.value?.status).toBe('done');
    expect(polls(calls)).toHaveLength(5);
  });

  it('gives up once polls have failed for about a minute', async () => {
    const times: number[] = [];
    const calls = mockFetch(() => {
      times.push(Date.now());
      return offline();
    });
    const result = outcome(waitForJob(job('running')));

    await vi.advanceTimersByTimeAsync(59_000);
    expect(result.settled).toBe(false);

    await vi.advanceTimersByTimeAsync(20_000);
    expect(result.error).toBeInstanceOf(ApiError);
    expect(result.error).toMatchObject({ status: 0 });
    // Backed off, not hammering a dead server.
    expect(polls(calls).length).toBeGreaterThan(3);
    expect(polls(calls).length).toBeLessThan(15);
    // The wait grows, but stops at 10 s: an uncapped doubling would leave a
    // recovered server unasked for half a minute or more.
    const gaps = times.slice(1).map((t, i) => t - times[i]);
    expect(gaps).toEqual([...gaps].sort((a, b) => a - b));
    expect(Math.max(...gaps)).toBe(10_000);

    // And stopped for good.
    const total = calls.length;
    await vi.advanceTimersByTimeAsync(600_000);
    expect(calls).toHaveLength(total);
  });

  it('measures the outage from the last successful poll, not in total', async () => {
    // Two outages of about 50 s with answers in between: each is inside the
    // limit even though together they are well over it.
    const start = Date.now();
    mockFetch(() => {
      const t = Date.now() - start;
      if (t < 50_000) return offline();
      if (t < 65_000) return json(job('running'));
      if (t < 115_000) return offline();
      return json(job('done'));
    });
    const result = outcome(waitForJob(job('running')));
    await vi.advanceTimersByTimeAsync(300_000);
    expect(result.error).toBeUndefined();
    expect(result.value?.status).toBe('done');
  });

  it('treats a 4xx as an answer, not an outage', async () => {
    const calls = mockFetch(() => json({ detail: 'unknown job' }, 404));
    const result = outcome(waitForJob(job('running')));
    await vi.advanceTimersByTimeAsync(120_000);
    expect(result.error).toMatchObject({ name: 'ApiError', status: 404, message: 'unknown job' });
    expect(polls(calls)).toHaveLength(1);
  });

  it('does not retry a rejected key', async () => {
    setApiKey('stale');
    const calls = mockFetch(() => json({ detail: 'missing or invalid API key' }, 401));
    const result = outcome(waitForJob(job('running')));
    await vi.advanceTimersByTimeAsync(120_000);
    expect(result.error).toMatchObject({ status: 401, message: expect.stringContaining('API key') });
    expect(polls(calls)).toHaveLength(1);
  });

  it('rejects with an AbortError when aborted', async () => {
    mockFetch(() => json(job('running')));
    const ac = new AbortController();
    const result = outcome(waitForJob(job('running'), undefined, ac.signal));
    await vi.advanceTimersByTimeAsync(2000);
    ac.abort();
    await vi.advanceTimersByTimeAsync(5000);
    expect(result.error).toBeInstanceOf(DOMException);
    expect((result.error as DOMException).name).toBe('AbortError');
  });

  describe('cancelOnAbort', () => {
    it('cancels the server job when aborted between polls', async () => {
      const calls = mockFetch(() => json(job('queued')));
      const ac = new AbortController();
      const result = outcome(waitForJob(job('queued'), undefined, ac.signal, { cancelOnAbort: true }));
      await vi.advanceTimersByTimeAsync(600); // one poll done, now sleeping

      ac.abort();
      // No timer advance: an abort cuts the backoff short.
      await vi.advanceTimersByTimeAsync(0);
      expect((result.error as DOMException | undefined)?.name).toBe('AbortError');
      expect(cancels(calls)).toHaveLength(1);
      expect(cancels(calls)[0].url).toBe(JOB_URL);
    });

    it('cancels the server job when aborted mid-poll', async () => {
      const calls = mockFetch((_url, init) => (init.method === 'DELETE' ? json(job('cancelled')) : hang()));
      const ac = new AbortController();
      const result = outcome(waitForJob(job('queued'), undefined, ac.signal, { cancelOnAbort: true }));
      await vi.advanceTimersByTimeAsync(600); // the first poll is in flight
      expect(polls(calls)).toHaveLength(1);

      ac.abort();
      await vi.advanceTimersByTimeAsync(0);
      expect((result.error as DOMException | undefined)?.name).toBe('AbortError');
      expect(cancels(calls)).toHaveLength(1);
    });

    // A job already inside torch answers 409. Vitest fails the run on an
    // unhandled rejection, so these also prove the cancel's failure is caught.
    it('does not surface a 409 from the cancel', async () => {
      const calls = mockFetch((_url, init) =>
        init.method === 'DELETE'
          ? json({ detail: 'job is running and can no longer be cancelled' }, 409)
          : json(job('running'))
      );
      const ac = new AbortController();
      const result = outcome(waitForJob(job('running'), undefined, ac.signal, { cancelOnAbort: true }));
      await vi.advanceTimersByTimeAsync(600);
      ac.abort();
      await vi.advanceTimersByTimeAsync(0);
      expect((result.error as DOMException | undefined)?.name).toBe('AbortError');
      expect(cancels(calls)).toHaveLength(1);
    });

    it('does not surface an unreachable server on cancel', async () => {
      const calls = mockFetch((_url, init) => (init.method === 'DELETE' ? offline() : json(job('running'))));
      const ac = new AbortController();
      const result = outcome(waitForJob(job('running'), undefined, ac.signal, { cancelOnAbort: true }));
      await vi.advanceTimersByTimeAsync(600);
      ac.abort();
      await vi.advanceTimersByTimeAsync(0);
      expect((result.error as DOMException | undefined)?.name).toBe('AbortError');
      expect(cancels(calls)).toHaveLength(1);
    });

    it('is off by default: aborting only stops watching', async () => {
      const calls = mockFetch(() => json(job('running')));
      const ac = new AbortController();
      const result = outcome(waitForJob(job('running'), undefined, ac.signal));
      await vi.advanceTimersByTimeAsync(600);
      ac.abort();
      await vi.advanceTimersByTimeAsync(5000);
      expect(result.settled).toBe(true);
      expect(cancels(calls)).toHaveLength(0);
    });
  });
});

// -- generateAudio -------------------------------------------------------------

describe('generateAudio', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  const AUDIO_URL = '/api/v1/audio/abc';
  const audio = () => new Response(new Blob(['RIFF']));

  it('fetches spoken audio straight away, without polling', async () => {
    const calls = mockFetch(url =>
      url === AUDIO_URL ? audio() : json(job('done', { id: 'voice:j2', kind: 'voice', audio_url: '/v1/audio/abc' }))
    );
    const seen: JobStatus[] = [];
    const { job: done, blob } = await generateAudio('voice', { prompt: 'hello' }, undefined, j =>
      seen.push(j.status)
    );

    expect(done.status).toBe('done');
    expect(await blob.text()).toBe('RIFF');
    expect(seen).toEqual(['done']);
    expect(calls.map(c => `${c.method} ${c.url}`)).toEqual(['POST /api/v1/audio/speech', `GET ${AUDIO_URL}`]);
  });

  it('waits for a queued job, then fetches what it produced', async () => {
    const calls = mockFetch(
      sequence(
        () => json(job('queued', { queue_position: 1 })),
        () => json(job('running', { progress: 0.5 })),
        () => json(job('done', { audio_url: '/v1/audio/abc' })),
        audio
      )
    );
    const seen: JobStatus[] = [];
    const result = outcome(generateAudio('music', { prompt: 'rain' }, undefined, j => seen.push(j.status)));
    await vi.advanceTimersByTimeAsync(10_000);

    expect(result.error).toBeUndefined();
    expect(result.value?.job.audio_url).toBe('/v1/audio/abc');
    expect(await result.value?.blob.text()).toBe('RIFF');
    expect(seen).toEqual(['queued', 'running', 'done']);
    expect(calls.map(c => `${c.method} ${c.url}`)).toEqual([
      'POST /api/v1/audio/music',
      `GET ${JOB_URL}`,
      `GET ${JOB_URL}`,
      `GET ${AUDIO_URL}`,
    ]);
  });

  // Voice holds the POST until the audio is made, so for voice that one
  // request is the whole wait. An abort that did not reach it let the
  // generation finish anyway, and the panel saved a clip after the user left.
  it('stops when the caller aborts before the server has answered', async () => {
    const calls = mockFetch(url =>
      url === AUDIO_URL
        ? audio()
        : new Promise<Response>(resolve => {
            const done = job('done', { id: 'voice:j2', kind: 'voice', audio_url: '/v1/audio/abc' });
            setTimeout(() => resolve(json(done)), 3000);
          })
    );
    const ac = new AbortController();
    const result = outcome(generateAudio('voice', { prompt: 'hello' }, ac.signal));
    await vi.advanceTimersByTimeAsync(1000);

    ac.abort();
    await vi.advanceTimersByTimeAsync(5000); // past when the server would have answered
    expect((result.error as DOMException | undefined)?.name).toBe('AbortError');
    expect(calls.map(c => `${c.method} ${c.url}`)).toEqual(['POST /api/v1/audio/speech']);
  });

  // Without the cancel, the queued job still took its turn on the GPU for
  // audio nobody would collect. GenerationPanel.test.tsx checks the panel's
  // unmount gets here.
  it('cancels the server job when the caller aborts while it waits', async () => {
    const calls = mockFetch((_url, init) => {
      if (init.method === 'POST') return json(job('queued', { queue_position: 2 }));
      if (init.method === 'DELETE') return json(job('cancelled'));
      return json(job('queued', { queue_position: 1 }));
    });
    const ac = new AbortController();
    const result = outcome(generateAudio('music', { prompt: 'rain' }, ac.signal));
    await vi.advanceTimersByTimeAsync(600); // submitted, one poll done
    expect(polls(calls)).toHaveLength(1);

    ac.abort();
    await vi.advanceTimersByTimeAsync(0);
    expect((result.error as DOMException | undefined)?.name).toBe('AbortError');
    expect(cancels(calls)).toHaveLength(1);
  });
});
