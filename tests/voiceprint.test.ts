/**
 * The voiceprint of a cluster (`src/main/asr/voiceprint.ts`), shared by the live worker and the
 * voices evaluation (docs/TESTING.md TS-21b).
 */

import { describe, expect, test } from "bun:test";
import { Accum, norm, score, unit } from "../src/main/asr/voiceprint.ts";

const v = (...xs: number[]) => Float32Array.from(xs);

describe("Accum", () => {
  test("identical lines agree fully; the vector is the unit mean and secs add up", () => {
    const a = new Accum();
    a.add(v(3, 4, 0), 2);
    a.add(v(3, 4, 0), 1.5);
    const r = a.result();
    expect(r?.cons).toBeCloseTo(1, 6);
    expect(r?.secs).toBe(3.5);
    expect(r?.n).toBe(2);
    expect(r?.norm).toBeCloseTo(5, 6);
    const [x, y, z] = Array.from(r?.vec ?? []);
    expect(x).toBeCloseTo(0.6, 6);
    expect(y).toBeCloseTo(0.8, 6);
    expect(z).toBe(0);
  });

  test("a cluster of two orthogonal voices scores low, and below a pure one", () => {
    const mixed = new Accum();
    const pure = new Accum();
    for (let i = 0; i < 4; i++) {
      mixed.add(v(1, 0, 0), 2);
      mixed.add(v(0, 1, 0), 2);
      pure.add(v(1, 0.05 * i, 0), 2);
      pure.add(v(1, -0.05 * i, 0), 2);
    }
    const m = mixed.result()?.cons as number;
    const p = pure.result()?.cons as number;
    expect(m).toBeCloseTo(Math.SQRT1_2, 5);
    // The positive control: the same check on a pure cluster gives a higher value.
    expect(p).toBeGreaterThan(0.99);
    expect(m).toBeLessThan(p);
  });

  test("cons reads each line's direction, not its loudness", () => {
    const a = new Accum();
    a.add(v(10, 0), 1);
    a.add(v(0, 1), 1);
    // The raw mean leans to the loud line; the agreement does not.
    expect(a.result()?.cons).toBeCloseTo(Math.SQRT1_2, 5);
  });

  test("a zero vector adds nothing, a vector of another size throws, an empty one is null", () => {
    const a = new Accum();
    expect(a.result()).toBeNull();
    expect(a.add(v(0, 0, 0), 5)).toBe(false);
    expect(a.result()).toBeNull();
    expect(a.add(v(1, 0, 0), 1)).toBe(true);
    expect(() => a.add(v(1, 0), 1)).toThrow(/size 2/);
    expect(a.n).toBe(1);
    expect(a.secs).toBe(1);
  });
});

describe("Accum across parts and merges", () => {
  test("resume goes on from a written voiceprint as n lines that agreed cons", () => {
    const a = Accum.resume(v(2, 0, 0), 30, 0.8, 10);
    expect(a.n).toBe(10);
    expect(a.secs).toBe(30);
    expect(a.result()?.cons).toBeCloseTo(0.8, 6);
    a.add(v(1, 0, 0), 2);
    expect(a.secs).toBe(32);
    expect(a.result()?.cons).toBeCloseTo((8 + 1) / 11, 6);
    expect(Array.from(a.result()?.vec ?? [])).toEqual([1, 0, 0]);
    // A zero vector or no weight keeps only the seconds.
    const z = Accum.resume(v(0, 0), 4, 1, 10);
    expect([z.n, z.secs, z.result()]).toEqual([0, 4, null]);
  });

  test("merge adds another accumulator as if its lines were added here; unmerge takes it back", () => {
    const a = new Accum();
    a.add(v(1, 0), 2);
    const b = new Accum();
    b.add(v(0, 1), 3);
    a.merge(b);
    const both = new Accum();
    both.add(v(1, 0), 2);
    both.add(v(0, 1), 3);
    expect(a.result()).toEqual(both.result());
    a.unmerge(b);
    expect([a.n, a.secs, a.result()?.cons]).toEqual([1, 2, 1]);
    // Sizes must match, and a refused merge changes nothing.
    expect(() => a.merge(Accum.resume(v(1, 0, 0), 5, 1, 3))).toThrow(/size 3/);
    expect([a.n, a.secs]).toEqual([1, 2]);
  });
});

describe("score and unit", () => {
  test("score is null on a size mismatch, where a min-length cosine would say 1", () => {
    expect(score(v(1, 0), v(1, 0, 5))).toBeNull();
    expect(score(v(1, 0), v(2, 0))).toBeCloseTo(1, 6);
    expect(score(v(1, 0), v(0, 1))).toBeCloseTo(0, 6);
    expect(score(v(0, 0), v(1, 0))).toBeNull();
    expect(score(v(), v())).toBeNull();
  });

  test("unit scales to length 1 and leaves a zero vector zero", () => {
    expect(norm(unit(v(3, 4)))).toBeCloseTo(1, 6);
    expect(Array.from(unit(v(0, 0)))).toEqual([0, 0]);
  });
});
