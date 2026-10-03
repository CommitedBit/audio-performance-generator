/**
 * A random v4 UUID, for clip ids and cached-audio keys.
 *
 * crypto.randomUUID exists only in secure contexts (HTTPS or localhost). Open
 * the app over plain HTTP from another machine on the LAN and it is undefined,
 * so every generation failed with "crypto.randomUUID is not a function" after
 * the audio had already been made. crypto.getRandomValues has no such
 * restriction, so the same kind of id is built from it there.
 */
export function newId(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID();

  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80; // RFC 4122 variant
  const hex = Array.from(bytes, b => b.toString(16).padStart(2, '0')).join('');
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}
