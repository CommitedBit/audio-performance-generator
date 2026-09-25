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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, withAuth(init));
  } catch {
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

const TERMINAL: JobStatus[] = ['done', 'error', 'cancelled'];

/**
 * Poll a job to completion.
 *
 * Music generation runs for minutes, so this backs off from 500ms to 3s rather
 * than hammering the server for the whole run.
 */
export async function waitForJob(
  job: JobInfo,
  onUpdate?: (job: JobInfo) => void,
  signal?: AbortSignal
): Promise<JobInfo> {
  let current = job;
  let delay = 500;

  while (!TERMINAL.includes(current.status)) {
    if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
    await new Promise(resolve => setTimeout(resolve, delay));
    delay = Math.min(delay * 1.4, 3000);
    current = await getJob(current.id, signal);
    onUpdate?.(current);
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

export interface HealthInfo {
  status: 'ok' | 'loading' | 'degraded';
  device: string;
  providers_available: number;
  providers_total: number;
  gpu: { name: string; vram_total_gb: number; vram_free_gb: number } | null;
}

export function fetchHealth(signal?: AbortSignal): Promise<HealthInfo> {
  return request<HealthInfo>('/health', { signal });
}
