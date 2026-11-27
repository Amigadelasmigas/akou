/**
 * Speaker labels (speakers.ts): live clusters, their persistence across parts, merge and unmerge,
 * and the final-to-live mapping.
 */

import { describe, expect, test } from "bun:test";
import {
  CENTROID_EVERY_MS,
  cosine,
  decodeVec,
  encodeVec,
  FIRST_MATCH_SEC,
  highestLabel,
  hungarianMax,
  LiveSpeakers,
  mapFinalToLive,
  type SpeakerEvent,
  StreamSpeakers,
  smoothTurns,
} from "../src/main/asr/speakers.ts";

const e = (...v: number[]) => Float32Array.from(v);

describe("live clusters", () => {
  test("a segment of 1 s or more joins the nearest cluster at 0.60, else starts c<N>", () => {
    const s = new LiveSpeakers();
    expect(s.assign(1, 0, 2, e(1, 0, 0))).toBe("c1");
    expect(s.assign(1, 3, 5, e(0.9, 0.1, 0))).toBe("c1");
    expect(s.assign(1, 6, 8, e(0, 1, 0))).toBe("c2");
    // Cosine 0.55 to c2: below the join threshold.
    expect(s.assign(1, 9, 11, e(0, 0.55, 0.835))).toBe("c3");
  });

  test("a short segment inherits the previous call speaker within 1 s, else c?", () => {
    const s = new LiveSpeakers();
    expect(s.assign(1, 0, 2, e(1, 0))).toBe("c1");
    expect(s.assign(1, 2.5, 2.9, null)).toBe("c1");
    expect(s.assign(1, 5, 5.5, null)).toBe("c?");
  });

  test("[T2.48, T3.10] Speaker numbers restart per part: centroids carry c2 into the next part", () => {
    const part1 = new LiveSpeakers();
    part1.assign(1, 0, 2, e(1, 0, 0));
    part1.assign(1, 3, 5, e(0, 1, 0));
    const written = part1.centroids(CENTROID_EVERY_MS, true);
    expect(written.map((w) => (w.type === "centroid" ? w.spk : ""))).toEqual(["c1", "c2"]);

    // A new app run restores from the log's speaker.centroid events.
    const part2 = new LiveSpeakers();
    part2.restore({
      centroids: written.flatMap((w) =>
        w.type === "centroid" ? [{ spk: w.spk, vec: w.vec }] : [],
      ),
    });
    expect(part2.assign(2, 0, 2, e(0.05, 1, 0))).toBe("c2");
    // A new voice continues the numbering, it does not restart at c1.
    expect(part2.assign(2, 3, 5, e(0, 0, 1))).toBe("c3");

    // Positive control: without the restore, the same voice is c1 again.
    const fresh = new LiveSpeakers();
    expect(fresh.assign(2, 0, 2, e(0.05, 1, 0))).toBe("c1");
  });

  test("centroids are written every five minutes, and all changed ones at a part end", () => {
    const s = new LiveSpeakers();
    s.assign(1, 0, 2, e(1, 0));
    expect(s.centroids(CENTROID_EVERY_MS)).toHaveLength(1);
    s.assign(1, 3, 5, e(1, 0.1));
    expect(s.centroids(CENTROID_EVERY_MS + 1000)).toEqual([]);
    expect(s.centroids(CENTROID_EVERY_MS + 2000, true)).toHaveLength(1);
    expect(s.centroids(CENTROID_EVERY_MS + 3000, true)).toEqual([]);
  });

  test("a centroid survives the base64 float32 round trip", () => {
    const v = e(0.25, -1.5, 3e-7);
    expect(decodeVec(encodeVec(v))).toEqual(v);
  });
});

describe("[judging] A wrong speaker merge with no way back", () => {
  test("converged clusters merge once; unmerge restores them and they are never merged again", () => {
    const s = new LiveSpeakers();
    s.assign(1, 0, 2, e(1, 0, 0));
    s.assign(1, 3, 5, e(0.45, 0.89, 0)); // cosine 0.45: its own cluster
    expect(s.merges()).toEqual([]);
    // c2 drifts toward c1 until the centroids are over 0.8 similar.
    for (let i = 0; i < 6; i++) s.assign(1, 6 + i * 2, 7.5 + i * 2, e(0.85, 0.53, 0));
    const m = s.merges();
    expect(m).toEqual([{ type: "merge", from: "c2", into: "c1" }]);
    // A merged-away cluster takes no segments.
    expect(s.assign(1, 30, 32, e(0.45, 0.89, 0))).toBe("c1");

    s.unmerge("c2", "c1");
    expect(s.assign(1, 33, 35, e(0.3, 0.95, 0))).toBe("c2");
    expect(s.merges()).toEqual([]);
  });

  test("an unmerged pair from the log is respected after a restore", () => {
    const s = new LiveSpeakers();
    s.restore({
      centroids: [
        { spk: "c1", vec: encodeVec(e(1, 0)) },
        { spk: "c2", vec: encodeVec(e(0.99, 0.1)) },
      ],
      unmerged: [{ from: "c2", into: "c1" }],
    });
    expect(s.merges()).toEqual([]);
    // Positive control: the same clusters with no unmerge history merge at once.
    const t = new LiveSpeakers();
    t.restore({
      centroids: [
        { spk: "c1", vec: encodeVec(e(1, 0)) },
        { spk: "c2", vec: encodeVec(e(0.99, 0.1)) },
      ],
    });
    expect(t.merges()).toHaveLength(1);
  });
});

describe("the speech and consistency of a cluster", () => {
  type Centroid = Extract<SpeakerEvent, { type: "centroid" }>;
  const only = (out: SpeakerEvent[]) => out.filter((c): c is Centroid => c.type === "centroid");

  test("secs sums the embedded segments' lengths on both paths; short lines add nothing", () => {
    const s = new LiveSpeakers();
    s.assign(1, 0, 2.5, e(1, 0, 0));
    s.assign(1, 3, 4.75, e(0.95, 0.05, 0));
    // Too short to embed: it inherits c1 but adds no speech.
    s.assign(1, 4.8, 5.4, null);
    s.addTo("c4", e(0, 0, 1), 3.25);
    s.addTo("c4", e(0, 0.1, 1), 1.5);
    const out = only(s.centroids(0, true));
    expect(out.map((c) => [c.spk, c.secs])).toEqual([
      ["c1", 4.3],
      ["c4", 4.8],
    ]);
  });

  test("cons is near 1 for one voice and drops for a cluster holding two", () => {
    const s = new LiveSpeakers();
    // Two directions 50 degrees apart, both within 0.60 of the running centroid: one cluster.
    s.addTo("c1", e(1, 0, 0), 2);
    s.addTo("c1", e(0.643, 0.766, 0), 2);
    s.addTo("c2", e(0, 0, 1), 2);
    s.addTo("c2", e(0, 0.05, 1), 2);
    const out = only(s.centroids(0, true));
    const cons = Object.fromEntries(out.map((c) => [c.spk, c.cons]));
    expect(cons.c2).toBeGreaterThan(0.99);
    expect(cons.c1).toBeCloseTo(Math.cos((25 * Math.PI) / 180), 2);
    expect(cons.c1 as number).toBeLessThan(cons.c2 as number);
  });

  test("the vector written and the labels are those of a cluster that keeps no voiceprint", () => {
    const s = new LiveSpeakers();
    const lines: [number, number, Float32Array][] = [
      [0, 2, e(1, 0, 0)],
      [3, 5, e(0.9, 0.1, 0)],
      [6, 8, e(0, 1, 0)],
      [9, 11, e(0.1, 0.9, 0.2)],
    ];
    const spks = lines.map(([a0, a1, v]) => s.assign(1, a0, a1, v));
    expect(spks).toEqual(["c1", "c1", "c2", "c2"]);
    const out = only(s.centroids(0, true));
    const near = (got: string | undefined, want: number[]) => {
      const vec = decodeVec(got ?? "");
      for (let i = 0; i < want.length; i++) expect(vec[i]).toBeCloseTo(want[i] as number, 6);
    };
    expect(out.map((c) => decodeVec(c.vec).length)).toEqual([3, 3]);
    near(out[0]?.vec, [0.95, 0.05, 0]);
    near(out[1]?.vec, [0.05, 0.95, 0.1]);
  });

  test("a cluster is written once, at once, when its speech first reaches FIRST_MATCH_SEC", () => {
    const s = new LiveSpeakers();
    let t = 1000;
    const step = (secs: number) => {
      s.addTo("c1", e(1, 0, 0), secs);
      return only(s.centroids(t++));
    };
    // A first write five minutes in, so the next ones are inside the interval.
    s.addTo("c2", e(0, 1, 0), 2);
    expect(only(s.centroids(CENTROID_EVERY_MS))).toHaveLength(1);
    t = CENTROID_EVERY_MS + 1;
    // The positive control: the step before the threshold writes nothing.
    expect(step(FIRST_MATCH_SEC - 0.5)).toEqual([]);
    const first = step(1);
    expect(first.map((c) => [c.spk, c.secs])).toEqual([["c1", FIRST_MATCH_SEC + 0.5]]);
    // Past the threshold, the cluster waits for the five-minute write again.
    expect(step(5)).toEqual([]);
    expect(step(5)).toEqual([]);
    expect(only(s.centroids(2 * CENTROID_EVERY_MS + 2)).map((c) => c.secs)).toEqual([
      FIRST_MATCH_SEC + 10.5,
    ]);
  });

  test("a restored cluster keeps secs and cons from its event, and counts only new lines without them", () => {
    const s = new LiveSpeakers();
    s.restore({
      centroids: [
        { spk: "c1", vec: encodeVec(e(1, 0, 0)), secs: 30.2, cons: 0.9 },
        { spk: "c2", vec: encodeVec(e(0, 1, 0)) },
      ],
    });
    // c1 already passed FIRST_MATCH_SEC in the earlier part: no early write for it.
    s.addTo("c1", e(1, 0, 0), 2);
    s.addTo("c2", e(0, 1, 0), 2);
    expect(only(s.centroids(1))).toEqual([]);
    const out = only(s.centroids(2, true));
    const by = Object.fromEntries(out.map((c) => [c.spk, c]));
    expect(by.c1?.secs).toBe(32.2);
    // Ten restored lines that agreed 0.9, and one that agrees fully.
    expect(by.c1?.cons).toBeCloseTo((0.9 * 10 + 1) / 11, 3);
    expect(by.c2?.secs).toBe(2);
    // No `cons` in its event: only the one line heard since, which agrees fully.
    expect(by.c2?.cons).toBeCloseTo(1, 3);
  });

  test("a merge adds the merged cluster's speech and an unmerge takes it back", () => {
    const s = new LiveSpeakers();
    s.assign(1, 0, 2, e(1, 0, 0));
    s.assign(1, 3, 5, e(0.45, 0.89, 0));
    for (let i = 0; i < 6; i++) s.assign(1, 6 + i * 2, 7.5 + i * 2, e(0.85, 0.53, 0));
    expect(s.merges()).toHaveLength(1);
    const merged = only(s.centroids(0, true)).find((c) => c.spk === "c1");
    expect(merged?.secs).toBe(2 + 2 + 6 * 1.5);
    s.unmerge("c2", "c1");
    s.assign(1, 30, 32, e(1, 0, 0));
    const back = only(s.centroids(1, true)).find((c) => c.spk === "c1");
    expect(back?.secs).toBe(4);
  });
});

describe("final clusters to live names", () => {
  test("the assignment is joint: each live cluster is used once, total overlap maximal", () => {
    // Greedy would give s0 -> c1 (10) and leave s1 with c2 (1); joint gives 9 + 9.
    expect(
      hungarianMax([
        [10, 9],
        [9, 1],
      ]),
    ).toEqual([1, 0]);
    expect(hungarianMax([[1, 2, 3]])).toEqual([2]);
    expect(hungarianMax([[5], [7]])).toEqual([-1, 0]);
  });

  test("60 % overlap or more is a map, below is a suggestion", () => {
    const m = mapFinalToLive(
      [
        { spk: "s0", t0: 0, t1: 10 },
        { spk: "s1", t0: 10, t1: 20 },
      ],
      [
        { spk: "c2", t0: 0, t1: 9 },
        { spk: "c1", t0: 10, t1: 15 },
        { spk: "c?", t0: 15, t1: 20 },
      ],
    );
    expect(m).toEqual([
      { final: "s0", live: "c2", overlap: 0.9, confirmed: true },
      { final: "s1", live: "c1", overlap: 0.5, confirmed: false },
    ]);
  });

  test("cosine of a zero vector is 0, never NaN", () => {
    expect(cosine(e(0, 0), e(1, 0))).toBe(0);
  });
});

describe("labels from a stream diarizer", () => {
  const R = 16000;
  test("a span is undecided until the model has decided all of it", () => {
    const s = new StreamSpeakers(R);
    s.add([{ speaker: 0, start: 0, end: 2 * R }], 1 * R);
    expect(s.speakerAt(0, 2 * R)).toBeNull();
    s.add([], 2 * R);
    expect(s.speakerAt(0, 2 * R)).toBe(0);
  });

  test("the speaker active longest inside the span wins; overlapping turns each count", () => {
    const s = new StreamSpeakers(R);
    s.add(
      [
        { speaker: 0, start: 0, end: 1 * R },
        { speaker: 1, start: 0.5 * R, end: 3 * R },
        { speaker: 0, start: 2.8 * R, end: 3 * R },
      ],
      4 * R,
    );
    expect(s.speakerAt(0, 3 * R)).toBe(1);
    // Positive control: a span inside speaker 0's turn alone is speaker 0.
    expect(s.speakerAt(0, 0.4 * R)).toBe(0);
  });

  test("a span nobody speaks in takes a turn within 1 s, else nobody (-1)", () => {
    const s = new StreamSpeakers(R);
    s.add([{ speaker: 2, start: 0, end: 1 * R }], 10 * R);
    expect(s.speakerAt(1.5 * R, 2 * R)).toBe(2);
    expect(s.speakerAt(5 * R, 6 * R)).toBe(-1);
  });

  test("pruned turns no longer label; kept ones do", () => {
    const s = new StreamSpeakers(R);
    s.add(
      [
        { speaker: 0, start: 0, end: 1 * R },
        { speaker: 1, start: 5 * R, end: 6 * R },
      ],
      10 * R,
    );
    s.prune(5 * R);
    expect(s.speakerAt(0, 1 * R)).toBe(-1);
    expect(s.speakerAt(5 * R, 6 * R)).toBe(1);
  });

  test("the highest c<N> is what a new stream numbers after", () => {
    expect(highestLabel(["you", "c?", "c2", "c10", "s4"])).toBe(10);
    expect(highestLabel([])).toBe(0);
  });

  test("addTo builds a label's centroid; nearest finds it at 0.60, never an excluded one", () => {
    const s = new LiveSpeakers();
    s.addTo("c1", e(1, 0, 0), 2);
    s.addTo("c2", e(0, 1, 0), 2);
    expect(s.nearest(e(0.9, 0.1, 0), new Set())).toBe("c1");
    expect(s.nearest(e(0.9, 0.1, 0), new Set(["c1"]))).toBeNull();
    // Positive control: below the join threshold nothing matches.
    expect(s.nearest(e(0.5, 0.5, 0.7), new Set())).toBeNull();
    const out = s.centroids(0, true);
    expect(out.map((c) => (c.type === "centroid" ? c.spk : ""))).toEqual(["c1", "c2"]);
    // A fresh cluster made by addTo keeps assign's numbering after it.
    expect(s.assign(1, 0, 2, e(0, 0, 1))).toBe("c3");
  });
});

describe("turns made fit to cut at", () => {
  test("one speaker's turns under 0.5 s apart join; turns under 0.2 s drop; overlaps stay", () => {
    const got = smoothTurns([
      { speaker: 0, start: 0, end: 1 },
      { speaker: 0, start: 1.3, end: 2 },
      { speaker: 1, start: 1.8, end: 3 },
      { speaker: 1, start: 4, end: 4.1 },
      { speaker: 0, start: 2.6, end: 3 },
    ]);
    expect(got).toEqual([
      { speaker: 0, start: 0, end: 2 },
      { speaker: 1, start: 1.8, end: 3 },
      { speaker: 0, start: 2.6, end: 3 },
    ]);
  });

  test("a blip that joins a neighbour survives; the same blip alone does not", () => {
    expect(
      smoothTurns([
        { speaker: 0, start: 0, end: 0.1 },
        { speaker: 0, start: 0.3, end: 0.4 },
      ]),
    ).toEqual([{ speaker: 0, start: 0, end: 0.4 }]);
    expect(smoothTurns([{ speaker: 0, start: 0, end: 0.1 }])).toEqual([]);
  });
});
