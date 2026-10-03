import { describe, expect, it } from 'vitest';
import { formatElapsed } from './format';

describe('formatElapsed', () => {
  it.each([
    [0, '0s'],
    [7.9, '7s'],
    [59, '59s'],
    [60, '1m 00s'],
    [125, '2m 05s'],
    [3600, '1h 00m'],
    [3780, '1h 03m'],
    [-3, '0s'],
  ])('%s seconds -> %s', (seconds, expected) => {
    expect(formatElapsed(seconds)).toBe(expected);
  });
});
