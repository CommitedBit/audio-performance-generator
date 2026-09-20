import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ApiError,
  type Capability,
  type JobInfo,
  type ProviderInfo,
  fetchAudio,
  generate,
  waitForJob,
} from '../api/client';
import { allFor, useModels } from '../hooks/useModels';
import { saveBlob } from '../hooks/useIndexedAudio';
import useProjectStore, { trackIdForType } from '../store/useProjectStore';

const CAPABILITIES: { id: Capability; label: string; placeholder: string }[] = [
  { id: 'voice', label: 'Voice', placeholder: 'Text to speak…' },
  { id: 'sfx', label: 'SFX', placeholder: 'A heavy wooden door creaking open…' },
  { id: 'music', label: 'Music', placeholder: 'Slow melancholic piano, sparse, reverb…' },
];

export default function GenerationPanel() {
  const addClip = useProjectStore(s => s.addClip);
  const project = useProjectStore(s => s.project);
  const { models, loading, error: modelsError, refresh } = useModels();

  const [capability, setCapability] = useState<Capability>('voice');
  const [text, setText] = useState('');
  const [providerId, setProviderId] = useState<string>('');
  const [voiceId, setVoiceId] = useState<string>('');
  const [seconds, setSeconds] = useState<number>(8);
  const [job, setJob] = useState<JobInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const providers = useMemo(() => allFor(models, capability), [models, capability]);
  const usable = useMemo(() => providers.filter(p => p.available), [providers]);

  // Follow the server's default whenever the capability changes, rather than
  // pinning a provider the UI happens to have listed first.
  useEffect(() => {
    const preferred = models?.defaults?.[capability] ?? '';
    const stillValid = usable.some(p => p.id === providerId);
    if (!stillValid) setProviderId(preferred || usable[0]?.id || '');
  }, [capability, models, usable, providerId]);

  const provider: ProviderInfo | undefined = useMemo(
    () => providers.find(p => p.id === providerId),
    [providers, providerId]
  );

  useEffect(() => {
    if (!provider) return;
    const stillValid = provider.voices.some(v => v.id === voiceId);
    if (!stillValid) setVoiceId(provider.voices[0]?.id ?? '');
  }, [provider, voiceId]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const handleGenerate = async () => {
    const prompt = text.trim();
    if (!prompt || !provider) return;

    const trackId = trackIdForType(project, capability);
    if (!trackId) {
      setError(`no ${capability} track exists`);
      return;
    }

    const ac = new AbortController();
    abortRef.current = ac;
    setBusy(true);
    setError(null);
    setJob(null);

    try {
      let current = await generate(
        capability,
        {
          prompt,
          provider: provider.id,
          voiceId: capability === 'voice' ? voiceId : undefined,
          seconds: capability === 'voice' ? undefined : seconds,
        },
        ac.signal
      );
      setJob(current);

      // Voice returns finished work; music and sfx come back queued.
      if (current.status !== 'done') {
        current = await waitForJob(current, setJob, ac.signal);
      }

      const blob = await fetchAudio(current);
      const blobId = await saveBlob(blob);

      // Trust the server's measured duration; fall back to decoding only if it
      // is missing, since decoding costs a full pass over the audio.
      let duration = Number(current.meta?.duration ?? 0);
      if (!duration || Number.isNaN(duration)) {
        const ctx = new AudioContext();
        try {
          const buf = await ctx.decodeAudioData(await blob.arrayBuffer());
          duration = buf.duration;
        } finally {
          void ctx.close();
        }
      }

      addClip(trackId, blobId, duration);
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
            {providers.map(p => (
              <option key={p.id} value={p.id} disabled={!p.available}>
                {p.name}
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

        {capability !== 'voice' && (
          <label>
            Length
            <input
              type="number"
              min={1}
              max={120}
              value={seconds}
              onChange={e => setSeconds(Number(e.target.value))}
              disabled={busy}
            />
            s
          </label>
        )}

        <button onClick={handleGenerate} disabled={blocked}>
          {busy ? 'Generating…' : `Generate ${active.label}`}
        </button>
      </div>

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
        </p>
      )}

      {error && <p className="error">{error}</p>}
    </div>
  );
}
