import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ApiError,
  type Capability,
  type JobInfo,
  type ProviderInfo,
  generateAudio,
} from '../api/client';
import { allFor, useModels } from '../hooks/useModels';
import { saveBlob } from '../hooks/useIndexedAudio';
import { formatElapsed } from '../lib/format';
import useProjectStore, { trackIdForType } from '../store/useProjectStore';

const CAPABILITIES: { id: Capability; label: string; placeholder: string }[] = [
  { id: 'voice', label: 'Voice', placeholder: 'Text to speak…' },
  { id: 'sfx', label: 'SFX', placeholder: 'A heavy wooden door creaking open…' },
  { id: 'music', label: 'Music', placeholder: 'Slow melancholic piano, sparse, reverb…' },
];

function formatRange(lo: number, hi: number): string {
  const n = (x: number) => String(Math.round(x * 100) / 100);
  if (Number.isFinite(lo) && Number.isFinite(hi)) return `${n(lo)}–${n(hi)}`;
  if (Number.isFinite(hi)) return `up to ${n(hi)}`;
  return `at least ${n(lo)}`;
}

export default function GenerationPanel() {
  const addClip = useProjectStore(s => s.addClip);
  const project = useProjectStore(s => s.project);
  const { models, loading, error: modelsError, refresh } = useModels();

  const [capability, setCapability] = useState<Capability>('voice');
  const [text, setText] = useState('');
  // The user's explicit picks. The provider and voice actually used are
  // derived from these during render, below.
  const [chosenProviderId, setProviderId] = useState<string>('');
  const [chosenVoiceId, setVoiceId] = useState<string>('');
  const [seconds, setSeconds] = useState<number>(8);
  const [job, setJob] = useState<JobInfo | null>(null);
  const [busy, setBusy] = useState(false);
  // Wall-clock time since Generate was pressed, ticking while busy: a first
  // model load can take minutes, and a silent spinner looks hung.
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!busy) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [busy]);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const providers = useMemo(() => allFor(models, capability), [models, capability]);
  const usable = useMemo(() => providers.filter(p => p.available), [providers]);

  // Keep the user's pick while it can serve this capability; otherwise follow
  // the server's default, rather than pinning a provider the UI happens to have
  // listed first. Never fall back to a cloud provider on our own: the server
  // deliberately has no default when only cloud can serve, so sending text out
  // stays an explicit choice. Derived during render -- an effect copying it
  // into state rendered a stale provider first.
  const providerId = usable.some(p => p.id === chosenProviderId)
    ? chosenProviderId
    : (models?.defaults?.[capability] ?? '') || usable.find(p => !p.remote)?.id || '';

  const onlyCloud = usable.length > 0 && usable.every(p => p.remote);

  const provider: ProviderInfo | undefined = useMemo(
    () => providers.find(p => p.id === providerId),
    [providers, providerId]
  );

  const voiceId = provider?.voices.some(v => v.id === chosenVoiceId)
    ? chosenVoiceId
    : provider?.voices[0]?.id ?? '';

  // Duration bounds come from the provider, not a fixed range: ACE-Step takes
  // 10-600 s, ElevenLabs SFX at most 22 s. A provider with no `seconds` param
  // takes no duration at all, so the control is hidden and nothing is sent --
  // matching the server, which ignores seconds for such providers.
  const secondsSpec = useMemo(
    () => provider?.params.find(p => p.name === 'seconds'),
    [provider]
  );
  const minSeconds = secondsSpec?.minimum ?? 0;
  const maxSeconds = secondsSpec?.maximum ?? Infinity;

  // On switching provider, keep the chosen length if it still fits, otherwise
  // take the new provider's default. Keyed on provider only, so typing a
  // partial number (e.g. "3" on the way to "30") is never clobbered. Adjusted
  // during render when the provider changes (React's pattern for state that
  // follows a prop), not in an effect, which rendered the old length first.
  const [secondsFor, setSecondsFor] = useState<ProviderInfo | undefined>(undefined);
  if (provider !== secondsFor) {
    setSecondsFor(provider);
    if (secondsSpec) {
      const lo = secondsSpec.minimum ?? -Infinity;
      const hi = secondsSpec.maximum ?? Infinity;
      if (!(seconds >= lo && seconds <= hi)) setSeconds(Number(secondsSpec.default));
    }
  }

  useEffect(() => () => abortRef.current?.abort(), []);

  const handleGenerate = async () => {
    const prompt = text.trim();
    if (!prompt || !provider) return;

    if (secondsSpec && !(Number.isFinite(seconds) && seconds >= minSeconds && seconds <= maxSeconds)) {
      // Checked here rather than left to the server so an out-of-range value
      // never creates a job that can only fail.
      setError(`Length must be ${formatRange(minSeconds, maxSeconds)} seconds for ${provider.name}`);
      return;
    }

    const trackId = trackIdForType(project, capability);
    if (!trackId) {
      setError(`no ${capability} track exists`);
      return;
    }

    const ac = new AbortController();
    abortRef.current = ac;
    setStartedAt(Date.now());
    setNow(Date.now());
    setBusy(true);
    setError(null);
    setJob(null);

    try {
      // An abort (the panel unmounting) abandons the result, and generateAudio
      // cancels the server job with it rather than leave it to hold the GPU.
      const { job: finished, blob } = await generateAudio(
        capability,
        {
          prompt,
          provider: provider.id,
          voiceId: capability === 'voice' ? voiceId : undefined,
          seconds: secondsSpec ? seconds : undefined,
        },
        ac.signal,
        setJob
      );
      // Best effort when the server kept the audio: see saveBlob.
      const blobId = await saveBlob(blob, finished.audio_id);

      // Trust the server's measured duration; fall back to decoding only if it
      // is missing, since decoding costs a full pass over the audio.
      let duration = Number(finished.meta?.duration ?? 0);
      if (!duration || Number.isNaN(duration)) {
        const ctx = new AudioContext();
        try {
          const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
          duration = buf.duration;
        } finally {
          void ctx.close();
        }
      }

      // audio_id is what the clip's audio is fetched back by once the browser
      // has dropped its cached copy.
      addClip(trackId, blobId, duration, { audioId: finished.audio_id ?? undefined, label: prompt });
      setText('');
      setJob(null);
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === 'AbortError') return;
      // The old panel swallowed this into console.error, leaving a dead button
      // and no explanation. Put it on screen.
      setError(err instanceof ApiError ? `${err.message} (HTTP ${err.status})` : String(err));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  };

  const active = CAPABILITIES.find(c => c.id === capability)!;
  const blocked = busy || !provider?.available || !text.trim();

  return (
    <div className="panel">
      <div className="row">
        {CAPABILITIES.map(c => (
          <button
            key={c.id}
            onClick={() => setCapability(c.id)}
            className={capability === c.id ? 'tab tab-active' : 'tab'}
            disabled={busy}
          >
            {c.label}
          </button>
        ))}
        <span className="spacer" />
        {models && <span className="muted">device: {models.device}</span>}
      </div>

      {loading && <p className="muted">Discovering models…</p>}

      {modelsError && (
        <p className="error">
          {modelsError} <button onClick={refresh}>Retry</button>
        </p>
      )}

      <textarea
        value={text}
        onChange={e => setText(e.target.value)}
        rows={4}
        placeholder={active.placeholder}
        disabled={busy}
      />

      <div className="row">
        <label>
          Model
          <select value={providerId} onChange={e => setProviderId(e.target.value)} disabled={busy}>
            {providerId === '' && providers.length > 0 && (
              <option value="" disabled>
                Choose a model…
              </option>
            )}
            {providers.map(p => (
              <option key={p.id} value={p.id} disabled={!p.available}>
                {p.name}
                {p.remote ? ' (cloud)' : ''}
                {p.available ? '' : ' — unavailable'}
              </option>
            ))}
            {providers.length === 0 && <option value="">none</option>}
          </select>
        </label>

        {capability === 'voice' && provider && provider.voices.length > 0 && (
          <label>
            Voice
            <select value={voiceId} onChange={e => setVoiceId(e.target.value)} disabled={busy}>
              {provider.voices.map(v => (
                <option key={v.id} value={v.id}>
                  {v.name}
                  {v.cloned ? ' (cloned)' : ''}
                </option>
              ))}
            </select>
          </label>
        )}

        {secondsSpec && (
          <label>
            Length
            <input
              type="number"
              min={secondsSpec.minimum ?? undefined}
              max={secondsSpec.maximum ?? undefined}
              step="any"
              value={seconds}
              onChange={e => setSeconds(Number(e.target.value))}
              disabled={busy}
              aria-label="Length in seconds"
            />
            s <span className="muted small">({formatRange(minSeconds, maxSeconds)})</span>
          </label>
        )}

        <button onClick={handleGenerate} disabled={blocked}>
          {busy ? 'Generating…' : `Generate ${active.label}`}
        </button>
      </div>

      {onlyCloud && providerId === '' && (
        <p className="warn">
          No local model can generate {active.label.toLowerCase()} right now. A cloud model is
          available, but choosing it sends your text to that service.
        </p>
      )}

      {provider && !provider.available && (
        <p className="warn">
          {provider.name} is unavailable: {provider.unavailable_reason}
        </p>
      )}

      {provider?.available && (
        <p className="muted small">
          {provider.description} · licence: {provider.license}
        </p>
      )}

      {job && job.status !== 'done' && (
        <p className="muted">
          {job.status}
          {job.queue_position ? ` (queued behind ${job.queue_position})` : ''}
          {job.message ? ` — ${job.message}` : ''}
          {busy && startedAt !== null ? ` · ${formatElapsed((now - startedAt) / 1000)}` : ''}
        </p>
      )}

      {error && <p className="error">{error}</p>}
    </div>
  );
}
