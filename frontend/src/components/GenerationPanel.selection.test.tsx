// @vitest-environment jsdom
//
// How the panel chooses its provider, voice and length. These used to be
// effects that copied derived values into state; they are now derived during
// render (and the length adjusted when the provider changes), so the rules
// themselves are pinned here.
import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { ModelsResponse, ProviderInfo } from '../api/client';
import GenerationPanel from './GenerationPanel';

vi.mock('idb-keyval', () => ({ get: vi.fn(), set: vi.fn(async () => {}) }));
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function provider(p: Partial<ProviderInfo> & Pick<ProviderInfo, 'id' | 'capability'>): ProviderInfo {
  return {
    name: p.id, license: 'MIT', requires_gpu: !p.remote, remote: false, description: '',
    available: true, unavailable_reason: '', loaded: false, voices: [], params: [], ...p,
  };
}

const seconds = (lo: number, hi: number, def: number) => [
  { name: 'seconds', type: 'float' as const, default: def, minimum: lo, maximum: hi, description: '' },
];

const MODELS: ModelsResponse = {
  device: 'cuda',
  providers: [
    provider({ id: 'chatterbox', capability: 'voice',
               voices: [{ id: 'default', name: 'Default', description: '', cloned: false },
                        { id: 'clone1', name: 'Narrator', description: '', cloned: true }] }),
    provider({ id: 'elevenlabs-voice', capability: 'voice', remote: true,
               voices: [{ id: 'rachel', name: 'Rachel', description: '', cloned: false }] }),
    provider({ id: 'acestep', capability: 'music', params: seconds(10, 600, 30) }),
    provider({ id: 'sa3-music', capability: 'music', params: seconds(1, 120, 45) }),
    provider({ id: 'elevenlabs-sfx', capability: 'sfx', remote: true, params: seconds(0.5, 22, 4) }),
  ],
  // sfx has only a cloud provider, so the server names no default for it.
  defaults: { voice: 'chatterbox', music: 'acestep', sfx: null },
};

afterEach(() => {
  vi.unstubAllGlobals();
  document.body.innerHTML = '';
});

/** Speech request bodies the panel sent, in order. */
let sent: Record<string, unknown>[] = [];

async function render() {
  sent = [];
  vi.stubGlobal('fetch', async (input: RequestInfo | URL, init: RequestInit = {}) => {
    if (String(input) === '/api/v1/models') {
      return new Response(JSON.stringify(MODELS), { headers: { 'Content-Type': 'application/json' } });
    }
    if (init.method === 'POST') sent.push(JSON.parse(String(init.body)));
    return new Response(JSON.stringify({ detail: 'not under test' }), { status: 500 });
  });
  const host = document.body.appendChild(document.createElement('div'));
  await act(async () => createRoot(host).render(<GenerationPanel />));
  await act(async () => { await new Promise(r => setTimeout(r, 0)); }); // the model list arrives
  return host;
}

const select = (host: HTMLElement, label: string) =>
  [...host.querySelectorAll('label')].find(l => l.textContent?.startsWith(label))!.querySelector('select')!;
const lengthField = (host: HTMLElement) => host.querySelector<HTMLInputElement>('input[aria-label="Length in seconds"]')!;

async function tab(host: HTMLElement, name: string) {
  const button = [...host.querySelectorAll('button')].find(b => b.textContent === name)!;
  await act(async () => button.click());
}

/** Change a React-controlled field: React only sees a value set through the prototype's setter. */
async function change(field: HTMLSelectElement | HTMLInputElement, value: string) {
  const proto = field instanceof HTMLSelectElement ? HTMLSelectElement.prototype : HTMLInputElement.prototype;
  await act(async () => {
    Object.getOwnPropertyDescriptor(proto, 'value')!.set!.call(field, value);
    field.dispatchEvent(new Event(field instanceof HTMLSelectElement ? 'change' : 'input', { bubbles: true }));
  });
}

/** Ask for speech and return the voice the request carried. */
async function voiceSent(host: HTMLElement): Promise<unknown> {
  const text = host.querySelector('textarea')!;
  await act(async () => {
    Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value')!.set!.call(text, 'hello');
    text.dispatchEvent(new Event('input', { bubbles: true }));
  });
  const generate = [...host.querySelectorAll('button')].find(b => b.textContent === 'Generate Voice')!;
  await act(async () => { generate.click(); await new Promise(r => setTimeout(r, 0)); });
  return sent.at(-1)?.voice_id;
}

describe('GenerationPanel selection', () => {
  it("follows each capability's server default and never picks a cloud provider on its own", async () => {
    const host = await render();
    expect(select(host, 'Model').value).toBe('chatterbox');

    await tab(host, 'Music');
    expect(select(host, 'Model').value).toBe('acestep');

    await tab(host, 'SFX');
    expect(select(host, 'Model').value).toBe('');                  // only cloud can serve: no pick
    expect(host.textContent).toContain('No local model can generate');
    await change(select(host, 'Model'), 'elevenlabs-sfx');          // an explicit choice is honoured
    expect(select(host, 'Model').value).toBe('elevenlabs-sfx');
  });

  it('keeps an explicit pick, and the voice follows the provider', async () => {
    const host = await render();
    expect(select(host, 'Voice').value).toBe('default');

    await change(select(host, 'Model'), 'elevenlabs-voice');
    expect(select(host, 'Model').value).toBe('elevenlabs-voice');
    // Asserted on the request, not the <select>: a select whose value matches
    // no option still DISPLAYS its first one, which would hide a missing fallback.
    expect(await voiceSent(host)).toBe('rachel');                   // that provider's first voice

    await change(select(host, 'Model'), 'chatterbox');
    await change(select(host, 'Voice'), 'clone1');
    expect(await voiceSent(host)).toBe('clone1');                   // the pick survives re-renders
    await tab(host, 'Music');
    await tab(host, 'Voice');
    expect(select(host, 'Model').value).toBe('chatterbox');
  });

  it('resets the length only when the new provider cannot take it', async () => {
    const host = await render();
    await tab(host, 'Music');
    expect(lengthField(host).value).toBe('30');                     // 8 is below ACE-Step's 10: its default

    await change(lengthField(host), '200');
    await change(select(host, 'Model'), 'sa3-music');               // max 120: 200 does not fit
    expect(lengthField(host).value).toBe('45');

    await change(lengthField(host), '100');
    await change(select(host, 'Model'), 'acestep');                 // 100 fits 10-600: kept
    expect(lengthField(host).value).toBe('100');
  });

  it('never clobbers a partly typed length', async () => {
    const host = await render();
    await tab(host, 'Music');
    await change(lengthField(host), '3');                           // on the way to "30"; below the minimum
    expect(lengthField(host).value).toBe('3');
  });
});
