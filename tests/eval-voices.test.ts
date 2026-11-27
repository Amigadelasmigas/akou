/**
 * The voices evaluation (docs/TESTING.md TS-21b): `scripts/eval/voices.ts` on synthetic
 * voiceprints, and the labels `scripts/eval-local.ts --voices` takes from a call's log.
 */

import { describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  evaluateVoices,
  LabelBuilder,
  numbersJson,
  type Ranked,
  shuffled,
  simulate,
  type VoiceLabel,
} from "../scripts/eval/voices.ts";
import { namedLabels, voiceLabels } from "../scripts/eval-local.ts";
import { EVENTS_FILE } from "../src/core/log/writer.ts";
import type { Candidate } from "../src/main/voices/match.ts";
import type { VoiceThresholds } from "../src/main/voices/thresholds.ts";
import { LogBuilder, T0, tempDir } from "./helpers.ts";

const DIM = 32;

function rng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296 - 0.5;
  };
}

function voice(r: () => number): Float32Array {
  return Float32Array.from({ length: DIM }, () => r());
}

function blend(a: Float32Array, b: Float32Array, w: number): Float32Array {
  return a.map((x, i) => (1 - w) * x + w * (b[i] as number));
}

/** Lines of one voice: its base plus noise of `spread`, 20 lines of 6 s. */
function lines(base: Float32Array, spread: number, r: () => number) {
  return Array.from({ length: 20 }, () => ({
    vec: base.map((x) => x + spread * r()),
    secs: 6,
  }));
}

function label(
  call: string,
  person: string,
  ls: { vec: Float32Array; secs: number }[],
  at: number,
): VoiceLabel {
  const b = new LabelBuilder();
  for (const l of ls) b.add(l.vec, l.secs);
  return b.label(call, person, at) as VoiceLabel;
}

/** `people` voices, each in every one of `calls` calls. */
function corpus(o: {
  bases: Float32Array[];
  calls: number;
  spread: number;
  seed?: number;
  /** The person of each base; defaults to one person per base. */
  names?: string[];
  /** The calls' names start with it. */
  prefix?: string;
}): { learn: VoiceLabel[] } {
  const r = rng(o.seed ?? 7);
  const learn: VoiceLabel[] = [];
  for (let c = 0; c < o.calls; c++)
    o.bases.forEach((b, i) => {
      learn.push(label(`${o.prefix ?? "call"}${c}`, o.names?.[i] ?? `p${i}`, lines(b, o.spread, r), c));
    });
  return { learn };
}

const separable = () => {
  const r = rng(1);
  return corpus({ bases: [voice(r), voice(r), voice(r), voice(r)], calls: 5, spread: 0.15 });
};

function strings(v: unknown): string[] {
  if (typeof v === "string") return [v];
  if (v && typeof v === "object") return Object.values(v).flatMap(strings);
  return [];
}

const SCRIPT = join(import.meta.dir, "..", "scripts", "eval", "voices.ts");
const raw = (labels: VoiceLabel[]) =>
  labels.map((l) => ({
    ...l,
    vec: Array.from(l.vec),
    snaps: l.snaps?.map((s) => ({ ...s, vec: Array.from(s.vec) })),
  }));

describe("the pick", () => {
  test("separable people give a finite ACCEPT with zero wrong names", () => {
    const { learn } = separable();
    const { result, controlOk } = evaluateVoices({ learn });
    expect(result.pick.accept).toBeNumber();
    expect(result.pick.wrong_name_rate).toBe(0);
    expect(result.pick.correct_name_rate as number).toBeGreaterThanOrEqual(0.5);
    expect(result.pick_mean.accept).toBeNumber();
    expect(result.pairs.eer).toBe(0);
    expect(controlOk).toBe(true);
    expect(result.control).toEqual({ shuffled_passes: 0, shuffled_passes_mean: 0 });
    expect(result.open).toBeNull();
    // Every grid point, for both scores: 2 x 26 ACCEPT x 4 MARGIN x 4 x 4.
    expect(result.grid.length).toBe(2 * 26 * 4 * 4 * 4);
    expect(result.grid.filter((g) => g.mean === 1).length).toBe(26 * 4 * 4 * 4);
  });

  test("overlapping people give ACCEPT null (positive control of the one above)", () => {
    const r = rng(1);
    const one = voice(r);
    const { learn } = corpus({
      bases: [one, blend(one, voice(r), 0.02), blend(one, voice(r), 0.02), voice(r)],
      calls: 5,
      spread: 0.15,
    });
    const { result } = evaluateVoices({ learn, noGrid: true });
    expect(result.pick.accept).toBeNull();
    expect(result.pick.margin).toBeNull();
  });

  test("consistency gates nothing: labels that score low on it are still learnt and named", () => {
    const { learn } = separable();
    const low = learn.map((l) => ({
      ...l,
      cons: 0.05,
      snaps: l.snaps?.map((s) => ({ ...s, cons: 0.05 })),
    }));
    const a = evaluateVoices({ learn, noGrid: true }).result.pick;
    const b = evaluateVoices({ learn: low, noGrid: true }).result.pick;
    expect(b.accept).toBeNumber();
    expect(b).toEqual(a);
    expect(Object.keys(evaluateVoices({ learn, noGrid: true }).result)).not.toContain("min_cons");
  });
});

describe("leave one call out and unknown voices", () => {
  test("a call is never scored against its own exemplar", () => {
    const r = rng(3);
    const bases = [voice(r), voice(r), voice(r), voice(r)];
    // Each person in one call only: only their own call could name them.
    const alone = corpus({ bases, calls: 4, spread: 0.1, names: ["a", "b", "c", "d"] });
    alone.learn.forEach((l, i) => {
      l.person = `${l.person}${Math.floor(i / 4)}`;
    });
    const res = evaluateVoices({ learn: alone.learn });
    expect(res.result.pick.accept).toBeNull();
    expect(res.result.grid.every((g) => g.correct_name_rate === null)).toBe(true);
    // The positive control: the same voices with their names kept across calls are named.
    const kept = corpus({ bases, calls: 4, spread: 0.1 });
    expect(evaluateVoices({ ...kept, noGrid: true }).result.pick.accept).toBeNumber();
  });

  test("the unknown-voice pass counts a name as wrong", () => {
    const r = rng(5);
    const a = voice(r);
    const b = blend(a, voice(r), 0.35);
    const c = voice(r);
    const d = voice(r);
    // Two close voices that never share a call: each is named right while both are known, and as
    // the other once dropped from the store.
    const one = corpus({ bases: [a, c, d], names: ["a", "c", "d"], calls: 5, spread: 0.05 });
    const two = corpus({
      bases: [b, c, d],
      names: ["b", "c", "d"],
      calls: 5,
      spread: 0.05,
      seed: 8,
      prefix: "other",
    });
    const { result } = evaluateVoices({ learn: [...one.learn, ...two.learn] });
    const row = result.grid.find(
      (g) =>
        g.accept === 0.4 &&
        g.margin === 0 &&
        g.min_match_sec === 10 &&
        g.min_learn_sec === 10 &&
        g.mean === 0,
    );
    expect(row?.wrong_names_enrolled).toBe(0);
    expect(row?.wrong_names_unknown as number).toBeGreaterThan(0);
    expect(row?.wrong_names).toBe(row?.wrong_names_unknown as number);
  });
});

describe("akou's matcher in time order", () => {
  const T: VoiceThresholds = {
    accept: 0.6,
    offer: 0.5,
    margin: 0.15,
    minMatchSec: 10,
    minLearnSec: 10,
    score: "best",
  };
  const e = (i: number) => Float32Array.from({ length: 4 }, (_, k) => (k === i ? 1 : 0));
  const cand = (name: string, score: number, i: number): Candidate => ({
    name,
    score,
    best: e(i),
    exemplars: [e(i)],
  });
  const w = (t: number, ranked: Candidate[], secs = 30): Ranked => ({ t, secs, ranked });

  test("the label that writes first takes the person, not the one that scores higher", () => {
    // Label 0 writes at 10 s with Ana at 0.8; label 1 at 20 s with Ana at 0.9.
    const first = [w(10, [cand("Ana", 0.8, 0), cand("Bo", 0.1, 1)])];
    const later = [w(20, [cand("Ana", 0.9, 0), cand("Bo", 0.1, 1)])];
    const out = simulate([first, later], T);
    expect(out[0]?.final).toEqual({ name: "Ana", status: "named" });
    expect(out[1]?.final).toBeNull();
    // The positive control: with the times swapped label 1 takes Ana.
    const swapped = simulate([[w(30, first[0]?.ranked ?? [])], later], T);
    expect(swapped[1]?.final).toEqual({ name: "Ana", status: "named" });
    expect(swapped[0]?.final).toBeNull();
  });

  test("every write is a decision: a label that drifts is withdrawn and counted", () => {
    const writes = [
      w(10, [cand("Ana", 0.8, 0), cand("Bo", 0.1, 1)]),
      w(300, [cand("Bo", 0.85, 1), cand("Ana", 0.3, 0)]),
      w(600, [cand("Ana", 0.9, 0), cand("Bo", 0.2, 1)]),
    ];
    const [o] = simulate([writes], T);
    expect(o?.named).toEqual(new Set(["Ana", "Bo"]));
    expect(o?.withdrawn).toBe(1);
    // Ana was withdrawn from this label, so her return does not bring her back.
    expect(o?.final).toEqual({ name: "Bo", status: "named" });
    // The positive control: the first write alone keeps Ana.
    expect(simulate([[writes[0] as Ranked]], T)[0]).toMatchObject({ withdrawn: 0 });
  });

  test("a write under MIN_MATCH_SEC is no decision, and people outside the labels stay taken", () => {
    const out = simulate([[w(10, [cand("Ana", 0.9, 0)], 5)]], T);
    expect(out[0]).toMatchObject({ eligible: false, final: null });
    const fixed = simulate([[w(10, [cand("Ana", 0.9, 0)])]], T, new Set(["Ana"]));
    expect(fixed[0]?.final).toBeNull();
    // The positive control: without the fixed person, Ana is named.
    expect(simulate([[w(10, [cand("Ana", 0.9, 0)])]], T)[0]?.final?.name).toBe("Ana");
  });

  test("a label's writes: each forced speech, each 5-minute tick it spoke before, and the end", () => {
    const b = new LabelBuilder();
    const r = rng(2);
    const base = voice(r);
    for (let k = 1; k <= 12; k++) b.add(base.map((x) => x + 0.1 * r()), 6, k * 60);
    const snaps = (b.label("c", "p") as VoiceLabel).snaps ?? [];
    expect(snaps.map((s) => [s.t, Math.round(s.secs), s.min ?? 0])).toEqual([
      [120, 12, 10],
      [240, 24, 20],
      [300, 24, 0],
      [300, 30, 30],
      [600, 54, 0],
      [600, 60, 60],
      [720, 72, 0],
    ]);
  });
});

describe("the open set", () => {
  test("labels nobody named are matched for counts only, with the call's named people left out", () => {
    const r = rng(21);
    const bases = [voice(r), voice(r), voice(r)];
    const { learn } = corpus({ bases, calls: 4, spread: 0.1 });
    const rr = rng(22);
    // In call0: p0's voice nobody named (p0 is named there too, so it is taken), p1's voice in a
    // call where nobody named p1, and a voice no one knows.
    const open = [
      label("call0", "", lines(bases[0] as Float32Array, 0.1, rr), 0),
      label("call9", "", lines(bases[1] as Float32Array, 0.1, rr), 9),
      label("call9", "", lines(voice(rr), 0.1, rr), 9),
    ];
    const { result } = evaluateVoices({ learn, open, noGrid: true });
    expect(result.pick.accept).toBeNumber();
    expect(result.open).toMatchObject({ clusters: 3, eligible: 3, named: 1 });
    const h = result.open?.histogram ?? [];
    expect(h.length).toBe(20);
    expect(h.reduce((a, b) => a + b, 0)).toBe(3);
    // The positive control: with p0 not named in call0, its voice is named there as well.
    const fewer = learn.filter((l) => !(l.call === "call0" && l.person === "p0"));
    expect(evaluateVoices({ learn: fewer, open, noGrid: true }).result.open?.named).toBe(2);
  });
});

describe("the gate", () => {
  test("shuffled labels that still pass make the script exit non-zero and write nothing", () => {
    const t = tempDir("akou-eval-voices-");
    try {
      const run = (learn: VoiceLabel[], name: string) => {
        const file = join(t.dir, `${name}.json`);
        const out = join(t.dir, `${name}-out.json`);
        writeFileSync(file, JSON.stringify({ learn: raw(learn) }));
        const p = Bun.spawnSync([process.execPath, SCRIPT, file, "--out", out], {
          stdout: "pipe",
          stderr: "pipe",
        });
        return { code: p.exitCode, out };
      };
      // One person in every call: shuffling changes nothing, so the gate passes shuffled too.
      const r = rng(9);
      const one = corpus({ bases: [voice(r)], calls: 4, spread: 0.1 });
      const bad = run(one.learn, "one");
      expect(bad.code).toBe(1);
      expect(existsSync(bad.out)).toBe(false);
      // The positive control: people that differ pass, and the file is written.
      const good = run(separable().learn, "good");
      expect(good.code).toBe(0);
      const text = readFileSync(good.out, "utf8");
      expect(JSON.parse(text).pick.accept).toBeNumber();
      for (const w of ["call0", "p0", "p3"]) expect(text).not.toContain(`"${w}"`);
    } finally {
      t.cleanup();
    }
  });

  test("two people in four calls pass the control: a shuffle that only renames people is refused", () => {
    for (let seed = 1; seed <= 6; seed++) {
      const r = rng(100 + seed);
      const data = corpus({ bases: [voice(r), voice(r)], calls: 4, spread: 0.1, seed });
      const { result, controlOk } = evaluateVoices({ ...data, noGrid: true });
      expect(result.pick.accept).toBeNumber();
      expect(controlOk).toBe(true);
    }
    // The shuffle itself never keeps who is with whom when another grouping exists.
    const r = rng(11);
    const { learn } = corpus({ bases: [voice(r), voice(r)], calls: 4, spread: 0.1 });
    for (let seed = 1; seed <= 50; seed++) {
      const sh = shuffled(learn, learn, seed).learn;
      const same = learn.every((a, i) =>
        learn.every((b, j) => (a.person === b.person) === (sh[i]?.person === sh[j]?.person)),
      );
      expect(same).toBe(false);
    }
  });

  test("the output holds no string values, and the writer refuses one", () => {
    const { result } = evaluateVoices({ ...separable(), noGrid: true });
    expect(strings(result)).toEqual([]);
    expect(strings(JSON.parse(numbersJson(result)))).toEqual([]);
    expect(() => numbersJson({ a: 1, b: { c: "Ben" } })).toThrow(/numbers only/);
    expect(() => numbersJson({ a: [1, "x"] })).toThrow(/numbers only/);
    expect(() => numbersJson({ a: Number.POSITIVE_INFINITY })).toThrow(/finite/);
  });
});

describe("labels from a call's log", () => {
  function call(): LogBuilder {
    const b = new LogBuilder();
    b.created();
    b.partStarted(1, T0);
    const seg = (id: string, spk: string, a0: number, a1: number) =>
      b.seg({ id, spk, a0, a1, text: "words" });
    seg("l000001", "c1", 0, 4);
    seg("l000002", "c2", 4, 30);
    seg("l000003", "c3", 30, 34);
    seg("l000004", "c4", 34, 38);
    seg("l000005", "c5", 38, 42);
    seg("l000006", "c6", 42, 42.5);
    seg("l000007", "c1", 42.5, 46);
    b.seg({ id: "l000008", ch: "mic", spk: "you", a0: 0, a1: 5, text: "mine" });
    const name = (spk: string, n: string, by: string) =>
      b.add({ type: "speaker.name", spk, name: n, by });
    name("c1", "Ana", "user");
    name("c2", " ana ", "agent:cli");
    name("c3", "Lee", "user");
    name("c3", "Kim", "user");
    name("c4", "Max", "agent:claude");
    name("c5", "Bo", "app");
    name("c6", "Sol", "user");
    name("c?", "Who", "user");
    return b;
  }

  test("only user and CLI names teach; a renamed label and a name on two labels are dropped", () => {
    const b = call();
    expect([...namedLabels(b.events)]).toEqual([["c6", "sol"]]);
    // The positive control: without the second label named Ana, c1 stands.
    const only = b.events.filter((e) => !(e.type === "speaker.name" && e.spk === "c2"));
    expect(new Map(namedLabels(only)).get("c1")).toBe("ana");
  });

  test("a named label merged into another named label is dropped: the merge renamed it", () => {
    const b = new LogBuilder();
    b.created();
    b.partStarted(1, T0);
    b.seg({ id: "l000001", spk: "c1", a0: 0, a1: 4, text: "words" });
    b.seg({ id: "l000002", spk: "c2", a0: 4, a1: 8, text: "words" });
    b.add({ type: "speaker.name", spk: "c1", name: "Bob", by: "user" });
    b.add({ type: "speaker.name", spk: "c2", name: "Ana", by: "user" });
    // The positive control: before the merge both labels stand.
    expect(new Map(namedLabels(b.events))).toEqual(
      new Map([
        ["c1", "bob"],
        ["c2", "ana"],
      ]),
    );
    b.add({ type: "speaker.merge", from: "c2", into: "c1" });
    expect([...namedLabels(b.events)]).toEqual([["c1", "bob"]]);
    // Undone, the merge no longer renames c2.
    b.add({ type: "speaker.unmerge", from: "c2", into: "c1" });
    expect(new Map(namedLabels(b.events)).get("c2")).toBe("ana");
  });

  test("a label's lines of 1 s or more are embedded from the call channel, cut at 12 s", async () => {
    const t = tempDir("akou-eval-voices-log-");
    try {
      const dir = join(t.dir, "work", "a");
      mkdirSync(dir, { recursive: true });
      const write = (drop: string) =>
        writeFileSync(
          join(dir, EVENTS_FILE),
          call()
            .events.filter((e) => !(e.type === "speaker.name" && e.spk === drop))
            .map((e) => `${JSON.stringify(e)}\n`)
            .join(""),
        );
      const cuts: number[] = [];
      const io = {
        read: () => new Float32Array(16000 * 60).fill(0.1),
        embed: (s: Float32Array) => {
          cuts.push(s.length / 16000);
          return Float32Array.from([1, 0, 0]);
        },
      };
      // c1 alone is Ana: its two lines (4 s and 3.5 s); c6's half-second line is too short. c2,
      // now unnamed, is the open set: its 26 s line is cut at 12 s, as the live worker cuts one.
      write("c2");
      const one = await voiceLabels(t.dir, io);
      expect(one.labels.map((l) => l.person)).toEqual(["ana"]);
      expect(cuts).toEqual([4, 12, 3.5]);
      expect(one.labels[0]?.secs).toBe(7.5);
      expect(one.open.map((l) => l.secs)).toEqual([12]);
      // Each line is a write at its end on the call's clock.
      expect(one.labels[0]?.snaps?.at(-1)?.t).toBe(46);
      // c2 alone is Ana: c1 is the open set now.
      write("c1");
      cuts.length = 0;
      const two = await voiceLabels(t.dir, io);
      expect(cuts).toEqual([4, 12, 3.5]);
      expect(two.labels.map((l) => [l.person, l.secs])).toEqual([["ana", 12]]);
      expect(two.open.map((l) => l.secs)).toEqual([7.5]);
      // A part with no audio is counted, and gives no label.
      const none = await voiceLabels(t.dir, { ...io, read: () => null });
      expect(none).toMatchObject({ labels: [], open: [], calls: 1, skipped: 1 });
    } finally {
      t.cleanup();
    }
  });
});
