import { useEffect, useRef, useState } from 'react';
import {
  type HealthInfo,
  type VoiceReference,
  deleteVoiceReference,
  fetchHealth,
  listVoiceReferences,
  uploadVoiceReference,
} from '../api/client';
import { useModels } from '../hooks/useModels';

/**
 * Server status and voice-clone reference management.
 *
 * This replaces the old page whose only control was an ElevenLabs API key
 * input backed by localStorage. Credentials now live in the server's
 * environment, so there is nothing secret for the browser to hold.
 */
export default function SettingsPage() {
  const { models, loading, error, refresh } = useModels();
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [voices, setVoices] = useState<VoiceReference[]>([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const ac = new AbortController();
    fetchHealth(ac.signal).then(setHealth).catch(() => setHealth(null));
    return () => ac.abort();
  }, [models]);

  const reloadVoices = () =>
    listVoiceReferences()
      .then(r => setVoices(r.voices))
      .catch(() => setVoices([]));

  useEffect(() => {
    void reloadVoices();
  }, []);

  const onUpload = async () => {
    const file = fileRef.current?.files?.[0];
    if (!file) return;
    setBusy(true);
    setNotice(null);
    try {
      const rec = await uploadVoiceReference(file, file.name.replace(/\.[^.]+$/, ''));
      setNotice(`Added reference "${rec.name}"`);
      if (fileRef.current) fileRef.current.value = '';
      await reloadVoices();
      refresh();
    } catch (err: unknown) {
      setNotice(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  const onDelete = async (id: string) => {
    setBusy(true);
    try {
      await deleteVoiceReference(id);
      await reloadVoices();
      refresh();
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="panel">
      <h2>Model server</h2>

      {loading && <p className="muted">Contacting server…</p>}
      {error && (
        <p className="error">
          {error} <button onClick={refresh}>Retry</button>
        </p>
      )}

      {health && (
        <p className="muted">
          status <strong>{health.status}</strong> · device <strong>{health.device}</strong> ·{' '}
          {health.providers_available}/{health.providers_total} providers available
          {health.gpu && (
            <>
              {' '}
              · {health.gpu.name} ({health.gpu.vram_free_gb}/{health.gpu.vram_total_gb} GB free)
            </>
          )}
        </p>
      )}

      {models && (
        <table className="providers">
          <thead>
            <tr>
              <th>Provider</th>
              <th>Type</th>
              <th>Licence</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {models.providers.map(p => (
              <tr key={p.id} className={p.available ? '' : 'dim'}>
                <td>
                  {p.name}
                  {p.loaded && <span className="badge">loaded</span>}
                </td>
                <td>{p.capability}</td>
                <td className="small">{p.license}</td>
                <td>{p.available ? 'ready' : p.unavailable_reason}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h2>Voice references</h2>
      <p className="muted small">
        Uploaded samples are used for zero-shot voice cloning. Only clone a voice you have
        permission to use.
      </p>

      <div className="row">
        <input ref={fileRef} type="file" accept="audio/*" disabled={busy} />
        <button onClick={onUpload} disabled={busy}>
          Upload reference
        </button>
      </div>

      {notice && <p className="muted">{notice}</p>}

      <ul className="voices">
        {voices.map(v => (
          <li key={v.id}>
            {v.name} <span className="muted small">({Math.round(v.bytes / 1024)} KB)</span>
            <button onClick={() => onDelete(v.id)} disabled={busy}>
              remove
            </button>
          </li>
        ))}
        {voices.length === 0 && <li className="muted">none yet</li>}
      </ul>
    </div>
  );
}
