/**
 * Client for the local model server.
 *
 * Replaces the old direct-to-ElevenLabs call. Nothing here knows which model
 * is behind an endpoint: the server advertises its providers via /v1/models
 * and the UI renders whatever it finds, so adding a model needs no frontend
 * change.
 */

const API_BASE: string = import.meta.env.VITE_API_BASE ?? '/api';

export type Capability = 'voice' | 'music' | 'sfx';

export interface VoiceInfo {
  id: string;
  name: string;
  description: string;
  cloned: boolean;
}

export interface ParamSpec {
  name: string;
  type: 'float' | 'int' | 'str' | 'bool';
  default: number | string | boolean;
  minimum: number | null;
  maximum: number | null;
  description: string;
}

export interface ProviderInfo {
  id: string;
  name: string;
  capability: Capability;
  license: string;
  requires_gpu: boolean;
  /** Runs on a third-party service (ElevenLabs): text leaves the machine. */
  remote: boolean;
  description: string;
  available: boolean;
  unavailable_reason: string;
  loaded: boolean;
  voices: VoiceInfo[];
  params: ParamSpec[];
}

export interface ModelsResponse {
  device: string;
  providers: ProviderInfo[];
  defaults: Record<Capability, string | null>;
}

export type JobStatus = 'queued' | 'running' | 'done' | 'error' | 'cancelled';

export interface JobInfo {
  id: string;
  kind: string;
  status: JobStatus;
  progress: number;
  message: string;
  audio_id: string | null;
  audio_url: string | null;
  error: string | null;
  queue_position: number | null;
  meta: Record<string, unknown>;
}

export interface GenerateOptions {
  prompt: string;
  provider?: string;
  voiceId?: string;
  seconds?: number;
  seed?: number;
  params?: Record<string, unknown>;
}

export class ApiError extends Error {
  // Written as an explicit field, not a constructor parameter property:
  // tsconfig sets `erasableSyntaxOnly`, which forbids the shorthand.
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

// -- access key ---------------------------------------------------------------
//
// When the server sets API_KEY, every request must carry it. The user enters
// it once on the Server page and it is kept in this browser.
//
// It is deliberately NOT injected by nginx. A proxy that attaches the key to
// every request authenticates anonymous callers along with everyone else, so
// setting API_KEY would protect nothing on an exposed port.

const KEY_STORAGE = 'apg.apiKey';
// Fallback when storage is unavailable (some private modes throw on access):
// the key then lasts for this page load instead of being silently dropped.
let memoryKey = '';

export function getApiKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? memoryKey;
  } catch {
    return memoryKey;
  }
}

export function setApiKey(key: string): void {
  memoryKey = key.trim();
  try {
    if (memoryKey) localStorage.setItem(KEY_STORAGE, memoryKey);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {
    /* storage unavailable; memoryKey still applies for this page */
  }
}

function withAuth(init?: RequestInit): RequestInit {
  const key = getApiKey();
  if (!key) return init ?? {};
  // Headers() accepts every HeadersInit shape and leaves FormData uploads
  // alone, so multipart boundaries are still set by the browser.
  const headers = new Headers(init?.headers);
  headers.set('X-API-Key', key);
  return { ...init, headers };
}

function authError(): ApiError {
  return new ApiError(
    401,
    getApiKey()
      ? 'the server rejected the API key - check it on the Server page'
      : 'this server requires an API key - enter it on the Server page'
  );
}

function isAbortError(err: unknown): boolean {
  return err instanceof DOMException && err.name === 'AbortError';
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, withAuth(init));
  } catch (err) {
    // An abort is the caller's own doing. Rewritten as "cannot reach" it was
    // indistinguishable from a dead server, so a caller could neither stay
    // quiet about a cancel nor retry only real outages.
    if (isAbortError(err)) throw err;
    // A dead backend is the most likely failure in local dev, so name it
    // rather than surfacing a bare "Failed to fetch".
    throw new ApiError(0, `cannot reach the model server at ${API_BASE} - is it running?`);
  }

  if (res.status === 401) throw authError();
  if (!res.ok) {
    // FastAPI puts the useful text in `detail`; keep the status either way.
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body?.detail === 'string' ? body.detail : JSON.stringify(body?.detail ?? body);
    } catch {
      /* non-JSON error body; statusText is the best we have */
    }
    throw new ApiError(res.status, detail);
  }

  return (await res.json()) as T;
}

export function fetchModels(signal?: AbortSignal): Promise<ModelsResponse> {
  return request<ModelsResponse>('/v1/models', { signal });
}

function generateBody(opts: GenerateOptions): string {
  return JSON.stringify({
    prompt: opts.prompt,
    provider: opts.provider,
    voice_id: opts.voiceId,
    seconds: opts.seconds,
    seed: opts.seed,
    params: opts.params ?? {},
  });
}

const JSON_HEADERS = { 'Content-Type': 'application/json' };

/** Voice generation is quick, so the server holds the request until it is done. */
export function generateSpeech(opts: GenerateOptions, signal?: AbortSignal): Promise<JobInfo> {
  return request<JobInfo>('/v1/audio/speech', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: generateBody(opts),
    signal,
  });
}

/** Music and SFX return a queued job immediately; poll it with waitForJob. */
export function generateMusic(opts: GenerateOptions, signal?: AbortSignal): Promise<JobInfo> {
  return request<JobInfo>('/v1/audio/music', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: generateBody(opts),
    signal,
  });
}

export function generateSfx(opts: GenerateOptions, signal?: AbortSignal): Promise<JobInfo> {
  return request<JobInfo>('/v1/audio/sfx', {
    method: 'POST',
    headers: JSON_HEADERS,
    body: generateBody(opts),
    signal,
  });
}

export function generate(
  capability: Capability,
  opts: GenerateOptions,
  signal?: AbortSignal
): Promise<JobInfo> {
  if (capability === 'voice') return generateSpeech(opts, signal);
  if (capability === 'music') return generateMusic(opts, signal);
  return generateSfx(opts, signal);
}

export function getJob(id: string, signal?: AbortSignal): Promise<JobInfo> {
  return request<JobInfo>(`/v1/jobs/${id}`, { signal });
}

export function cancelJob(id: string): Promise<JobInfo> {
  return request<JobInfo>(`/v1/jobs/${id}`, { method: 'DELETE' });
}

/**
 * Cancel a job nobody is waiting for any more, best effort.
 *
 * A 409 means it is already running (work inside torch cannot be interrupted)
 * and a network error means there is no server to tell. The caller has moved
 * on either way and has nowhere to show either, so neither is an error.
 */
function abandonJob(id: string): void {
  void cancelJob(id).catch(() => {
    /* 409 or unreachable: nothing more to do, and nowhere to report it */
  });
}

const TERMINAL: JobStatus[] = ['done', 'error', 'cancelled'];

// Music generation runs for minutes, so polling backs off from 500ms to 3s
// rather than hammering the server for the whole run.
const POLL_FIRST_MS = 500;
const POLL_MAX_MS = 3000;
// A failed poll backs off further, up to this.
const RETRY_MAX_MS = 10_000;
// How long polls may keep failing before the job is given up on: long enough
// to ride out a gateway or nginx restart or a Wi-Fi drop, short enough that a
// server that is really gone is reported instead of polled forever.
const OUTAGE_LIMIT_MS = 60_000;

/**
 * Whether a failed poll is worth repeating.
 *
 * Status 0 is "could not reach the server", and a 5xx is mostly the gateway
 * saying the same of a model service (502/504) or a server mid-restart (503),
 * which clear on their own. A 4xx is an answer -- a 404 after a model service
 * restarted means the job is gone, a 401 means the key is wrong -- and asking
 * again cannot change it.
 */
function isTransient(err: unknown): boolean {
  return err instanceof ApiError && (err.status === 0 || err.status >= 500);
}

/** A delay that an abort cuts short, so a cancel never waits out a backoff. */
function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException('aborted', 'AbortError'));
      return;
    }
    const onAbort = () => {
      clearTimeout(timer);
      reject(new DOMException('aborted', 'AbortError'));
    };
    const timer = setTimeout(() => {
      signal?.removeEventListener('abort', onAbort);
      resolve();
    }, ms);
    signal?.addEventListener('abort', onAbort, { once: true });
  });
}

export interface WaitOptions {
  /**
   * Cancel the server job when `signal` aborts. For callers whose abort means
   * the result is no longer wanted, not merely that one view stopped watching.
   */
  cancelOnAbort?: boolean;
}

/**
 * Poll a job to completion.
 *
 * A failed poll is retried while it looks transient (see isTransient), until
 * none has succeeded for OUTAGE_LIMIT_MS. One dropped request used to fail a
 * generation that was still running fine on the server.
 */
export async function waitForJob(
  job: JobInfo,
  onUpdate?: (job: JobInfo) => void,
  signal?: AbortSignal,
  { cancelOnAbort = false }: WaitOptions = {}
): Promise<JobInfo> {
  let current = job;
  let delay = POLL_FIRST_MS;
  let failingSince: number | null = null;

  try {
    while (!TERMINAL.includes(current.status)) {
      await sleep(delay, signal);
      try {
        current = await getJob(current.id, signal);
      } catch (err) {
        if (!isTransient(err)) throw err;
        failingSince ??= Date.now();
        if (Date.now() - failingSince >= OUTAGE_LIMIT_MS) throw err;
        delay = Math.min(delay * 2, RETRY_MAX_MS);
        continue;
      }
      failingSince = null;
      delay = Math.min(delay * 1.4, POLL_MAX_MS);
      onUpdate?.(current);
    }
  } catch (err) {
    // Stopping the poll alone leaves a queued job to take its turn on the GPU
    // and produce audio nobody collects.
    if (cancelOnAbort && isAbortError(err)) abandonJob(current.id);
    throw err;
  }

  if (current.status === 'error') {
    throw new ApiError(500, current.error ?? 'generation failed');
  }
  if (current.status === 'cancelled') {
    throw new ApiError(499, 'generation was cancelled');
  }
  return current;
}

/** Fetch the finished audio so it can be cached locally and decoded. */
export async function fetchAudio(job: JobInfo): Promise<Blob> {
  if (!job.audio_url) throw new ApiError(500, 'job finished without producing audio');
  const res = await fetch(`${API_BASE}${job.audio_url}`, withAuth());
  if (res.status === 401) throw authError();
  if (!res.ok) throw new ApiError(res.status, `could not fetch audio: ${res.statusText}`);
  return await res.blob();
}

/**
 * One clip's audio, start to finish: submit, wait for the job, fetch the file.
 *
 * Aborting `signal` means the caller no longer wants the result (the panel
 * unmounted), so a job still waiting on the server is cancelled rather than
 * left to take its turn on the GPU. The sequence lives here, not in the
 * component, so its abort and cancel cases are tested without a DOM; the
 * panel's own test only checks that unmounting gets here.
 */
export async function generateAudio(
  capability: Capability,
  opts: GenerateOptions,
  signal?: AbortSignal,
  onUpdate?: (job: JobInfo) => void
): Promise<{ job: JobInfo; blob: Blob }> {
  let job = await generate(capability, opts, signal);
  onUpdate?.(job);
  // Voice returns finished work; music and sfx come back queued.
  if (job.status !== 'done') {
    job = await waitForJob(job, onUpdate, signal, { cancelOnAbort: true });
  }
  return { job, blob: await fetchAudio(job) };
}

// -- voice clone references ---------------------------------------------------

export interface VoiceReference {
  id: string;
  name: string;
  bytes: number;
  created_at: number;
}

export function listVoiceReferences(): Promise<{ voices: VoiceReference[] }> {
  return request<{ voices: VoiceReference[] }>('/v1/voices');
}

export function uploadVoiceReference(file: File, name: string): Promise<VoiceReference> {
  const form = new FormData();
  form.append('file', file);
  form.append('name', name);
  return request<VoiceReference>('/v1/voices', { method: 'POST', body: form });
}

export function deleteVoiceReference(id: string): Promise<{ deleted: string }> {
  return request<{ deleted: string }>(`/v1/voices/${id}`, { method: 'DELETE' });
}

/** ok: a local model can serve it · cloud: only a cloud provider · stub: placeholder only · down: nothing. */
export type CapabilityStatus = 'ok' | 'cloud' | 'stub' | 'down';

export interface HealthInfo {
  status: 'ok' | 'stub' | 'degraded' | 'down';
  device: string;
  providers_available: number;
  providers_total: number;
  capabilities: Partial<Record<Capability, { status: CapabilityStatus; available: string[] }>>;
  gpu: { name: string; vram_total_gb: number; vram_free_gb: number } | null;
  /** Gateway only. */
  upstreams?: Record<string, 'up' | 'down'> | null;
  /** Gateway only: the models REQUIRED_PROVIDERS says a working install needs. */
  required?: Record<string, { status: 'ok' | 'unavailable' | 'missing'; reason: string }> | null;
}

export function fetchHealth(signal?: AbortSignal): Promise<HealthInfo> {
  return request<HealthInfo>('/health', { signal });
}
