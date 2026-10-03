import { useCallback, useEffect, useState } from 'react';
import { type Capability, type ModelsResponse, type ProviderInfo, fetchModels } from '../api/client';

interface ModelsState {
  models: ModelsResponse | null;
  loading: boolean;
  error: string | null;
  refresh: () => void;
}

/**
 * Discovers what the server can generate.
 *
 * The old UI hardcoded three ElevenLabs voice ids. Everything selectable is now
 * driven by this, so swapping a model server changes the UI with no code edit.
 */
export function useModels(): ModelsState {
  const [models, setModels] = useState<ModelsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    const ac = new AbortController();
    fetchModels(ac.signal)
      .then(res => {
        setModels(res);
        setError(null);
      })
      .catch((err: unknown) => {
        if (ac.signal.aborted) return;
        setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!ac.signal.aborted) setLoading(false);
      });
    return () => ac.abort();
  }, [nonce]);

  // Loading starts true for the first fetch; a refresh raises it here, in the
  // event, rather than in the effect, where setState costs an extra render.
  const refresh = useCallback(() => {
    setLoading(true);
    setNonce(n => n + 1);
  }, []);
  return { models, loading, error, refresh };
}

/** Providers that can actually serve this capability right now. */
export function availableFor(models: ModelsResponse | null, capability: Capability): ProviderInfo[] {
  if (!models) return [];
  return models.providers.filter(p => p.capability === capability && p.available);
}

/** Every provider for a capability, available or not — the UI explains the gaps. */
export function allFor(models: ModelsResponse | null, capability: Capability): ProviderInfo[] {
  if (!models) return [];
  return models.providers.filter(p => p.capability === capability);
}
