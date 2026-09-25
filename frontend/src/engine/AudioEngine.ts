import type { Clip } from '../types/timeline';

export class AudioEngine {
  private ctx = new AudioContext();
  private buffers = new Map<string, AudioBuffer>(); // blobId -> decoded buffer
  private getBlob: (id: string) => Promise<Blob | undefined>;

  // NOTE: written as an explicit field rather than a constructor parameter
  // property because tsconfig sets `erasableSyntaxOnly`, which forbids the
  // `constructor(private getBlob: ...)` shorthand (it emits runtime code).
  constructor(getBlob: (id: string) => Promise<Blob | undefined>) {
    this.getBlob = getBlob;
  }

  async play(clips: Clip[], when = 0) {
    const t0 = this.ctx.currentTime + when;
    for (const c of clips) {
      const b = await this.loadBuffer(c.blobId);
      const src = this.ctx.createBufferSource();
      src.buffer = b;
      src.connect(this.ctx.destination);
      src.start(t0 + c.start, c.offset, c.duration);
    }
  }

  pause() { this.ctx.suspend(); }
  resume() { this.ctx.resume(); }

  private async loadBuffer(id: string) {
    if (!this.buffers.has(id)) {
      const blob = await this.getBlob(id);
      if (!blob) throw new Error('audio missing');
      this.buffers.set(
        id,
        await blob.arrayBuffer().then(ab => this.ctx.decodeAudioData(ab))
      );
    }
    return this.buffers.get(id)!;
  }
}
