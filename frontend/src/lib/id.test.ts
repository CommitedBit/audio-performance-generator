import { afterEach, describe, expect, it, vi } from 'vitest';
import { newId } from './id';

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;
const realCrypto = globalThis.crypto;

/** crypto as a page served over plain HTTP sees it: no randomUUID. */
function insecureContext(fill?: number) {
  vi.stubGlobal('crypto', {
    getRandomValues: <T extends ArrayBufferView>(array: T): T =>
      fill === undefined
        ? realCrypto.getRandomValues(array)
        : (new Uint8Array(array.buffer).fill(fill) as unknown as T),
  });
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('newId', () => {
  it('uses crypto.randomUUID where the browser has it', () => {
    const spy = vi.spyOn(realCrypto, 'randomUUID').mockReturnValue('0a0a0a0a-0a0a-4a0a-8a0a-0a0a0a0a0a0a');
    expect(newId()).toBe('0a0a0a0a-0a0a-4a0a-8a0a-0a0a0a0a0a0a');
    expect(spy).toHaveBeenCalledOnce();
    spy.mockRestore();
  });

  it('builds a v4 UUID itself in a non-secure context', () => {
    insecureContext();
    const ids = new Set(Array.from({ length: 1000 }, newId));
    expect(ids.size).toBe(1000);
    for (const id of ids) expect(id).toMatch(UUID_V4);
  });

  it('sets the version and variant bits whatever the random bytes are', () => {
    insecureContext(0x00);
    expect(newId()).toBe('00000000-0000-4000-8000-000000000000');
    insecureContext(0xff);
    expect(newId()).toBe('ffffffff-ffff-4fff-bfff-ffffffffffff');
  });
});
