/**
 * A voiceprint: what a cluster of call lines says about one voice. The live worker and the voices
 * evaluation (`scripts/eval/voices.ts`) build it the same way, so the evaluation measures what
 * akou ships.
 *
 * - `vec` is the mean of the raw embeddings, scaled to unit length.
 * - `secs` is the speech the embeddings came from.
 * - `cons` is how well the lines agree with their own direction: the length of the sum of each
 *   embedding scaled to unit length, divided by their count. One voice scores near 1; a cluster
 *   that holds two voices scores lower, since their directions partly cancel.
 *
 * `score` compares two voiceprints. Unlike `cosine` in speakers.ts, which compares over the
 * shorter of two vectors, it refuses vectors of different sizes, so two embedders' vectors are
 * never compared by accident.
 */

export interface Voiceprint {
  /** The mean embedding, unit length. */
  vec: Float32Array;
  /** Seconds of speech added. */
  secs: number;
  /** Mean agreement of the embeddings with their own direction, 0 to 1. */
  cons: number;
  /** Embeddings added. */
  n: number;
  /** Length of the mean raw embedding, before it is scaled to unit length. */
  norm: number;
}

/** The Euclidean length of `v`. */
export function norm(v: Float32Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) s += (v[i] as number) * (v[i] as number);
  return Math.sqrt(s);
}

/** `v` scaled to unit length; a zero vector comes back as zeros. */
export function unit(v: Float32Array): Float32Array {
  const n = norm(v);
  const out = new Float32Array(v.length);
  if (n === 0) return out;
  for (let i = 0; i < v.length; i++) out[i] = (v[i] as number) / n;
  return out;
}

/** The cosine of `a` and `b`, or null when their sizes differ or either is zero. */
export function score(a: Float32Array, b: Float32Array): number | null {
  if (a.length !== b.length || a.length === 0) return null;
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    const x = a[i] as number;
    const y = b[i] as number;
    dot += x * y;
    na += x * x;
    nb += y * y;
  }
  return na === 0 || nb === 0 ? null : dot / Math.sqrt(na * nb);
}

/** Adds embeddings one line at a time and gives the voiceprint of all of them. */
export class Accum {
  private sum: Float32Array | null = null;
  private nsum: Float32Array | null = null;
  private count = 0;
  private seconds = 0;

  /**
   * Adds one line's embedding and its length in seconds. A zero vector (an embedder that was not
   * ready) adds nothing and returns false. A vector of another size than the first throws.
   */
  add(vec: Float32Array, secs: number): boolean {
    const n = norm(vec);
    if (n === 0) return false;
    if (this.sum && this.sum.length !== vec.length)
      throw new Error(`embedding of size ${vec.length} added to voiceprint of ${this.sum.length}`);
    this.sum ??= new Float32Array(vec.length);
    this.nsum ??= new Float32Array(vec.length);
    for (let i = 0; i < vec.length; i++) {
      const x = vec[i] as number;
      this.sum[i] = (this.sum[i] as number) + x;
      this.nsum[i] = (this.nsum[i] as number) + x / n;
    }
    this.count++;
    this.seconds += secs;
    return true;
  }

  get n(): number {
    return this.count;
  }

  get secs(): number {
    return this.seconds;
  }

  /** The voiceprint so far, or null before any embedding was added. */
  result(): Voiceprint | null {
    if (!this.sum || !this.nsum || this.count === 0) return null;
    const mean = new Float32Array(this.sum.length);
    for (let i = 0; i < mean.length; i++) mean[i] = (this.sum[i] as number) / this.count;
    return {
      vec: unit(mean),
      secs: this.seconds,
      cons: Math.min(1, norm(this.nsum) / this.count),
      n: this.count,
      norm: norm(mean),
    };
  }
}
