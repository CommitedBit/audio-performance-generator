import { useEffect, useRef, useState } from 'react';
import {
  type HealthInfo,
  type VoiceReference,
  deleteVoiceReference,
  fetchHealth,
  getApiKey,
  listVoiceReferences,
  setApiKey,
  uploadVoiceReference,
} from '../api/client';
import { useModels } from '../hooks/useModels';

/**
 * Server access, status, and voice-clone reference management.
 *
 * Model-provider credentials (ElevenLabs, HuggingFace) live in the server's
 * environment and never reach the browser. The one thing the browser holds is
 * this server's own access key, when the operator sets API_KEY.
 */
export default function SettingsPage() {
  const { models, loading, error, refresh } = useModels();
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [voices, setVoices] = useState<VoiceReference[]>([]);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const [keyInput, setKeyInput] = useState(getApiKey());
  const [keySaved, setKeySaved] = useState(getApiKey() !== '');

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

  const saveKey = (value: string) => {
    setApiKey(value);
    setKeyInput(value.trim());
    setKeySaved(value.trim() !== '');
    setNotice(value.trim() ? 'API key saved in this browser' : 'API key cleared');
    // Everything on this page may have failed with 401 before the key existed.
    refresh();
    void reloadVoices();
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
      <h2>Access key</h2>
      <p className="muted small">
        Only needed when the server sets <code>API_KEY</code>. Stored in this browser and sent
        with every request.
      </p>
      <div className="row">
        <input
          type="password"
          value={keyInput}
          onChange={e => setKeyInput(e.target.value)}
          placeholder={keySaved ? '' : 'not set'}
          autoComplete="off"
          aria-label="API key"
        />
        <button onClick={() => saveKey(keyInput)}>Save</button>
        {keySaved && <button onClick={() => saveKey('')}>Clear</button>}
      </div>

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
