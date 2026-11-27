/**
 * The voices evaluation (docs/TESTING.md TS-21b): `scripts/eval/voices.ts` on synthetic
 * voiceprints, and the labels `scripts/eval-local.ts --voices` takes from a call's log.
 */

import { describe, expect, test } from "bun:test";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import {
  decide,
  evaluateVoices,
  LabelBuilder,
  type MixedLabel,
  numbersJson,
  shuffled,
  type VoiceLabel,
} from "../scripts/eval/voices.ts";
import { namedLabels, voiceLabels } from "../scripts/eval-local.ts";
import { EVENTS_FILE } from "../src/core/log/writer.ts";
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

/** `people` voices, each in every one of `calls` calls, plus the two-voice clusters of each call. */
function corpus(o: {
  bases: Float32Array[];
  calls: number;
  spread: number;
  seed?: number;
  /** The person of each base; defaults to one person per base. */
  names?: string[];
  /** The calls' names start with it. */
  prefix?: string;
}): { learn: VoiceLabel[]; mixed: MixedLabel[] } {
  const r = rng(o.seed ?? 7);
  const learn: VoiceLabel[] = [];
  const mixed: MixedLabel[] = [];
  for (let c = 0; c < o.calls; c++) {
    const perCall = o.bases.map((b) => lines(b, o.spread, r));
    perCall.forEach((ls, i) => {
      learn.push(label(`${o.prefix ?? "call"}${c}`, o.names?.[i] ?? `p${i}`, ls, c));
    });
    const b = new LabelBuilder();
    for (let k = 0; k < 20; k++) {
      const x = perCall[0]?.[k];
      const y = perCall[1]?.[k];
      if (x) b.add(x.vec, x.secs);
      if (y) b.add(y.vec, y.secs);
    }
    const m = b.label("", "");
    if (m) mixed.push({ cons: m.cons, snaps: m.snaps });
  }
  return { learn, mixed };
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
    const { learn, mixed } = separable();
    const { result, controlOk } = evaluateVoices({ learn, mixed });
    expect(result.pick.accept).toBeNumber();
    expect(result.pick.wrong_name_rate).toBe(0);
    expect(result.pick.correct_name_rate as number).toBeGreaterThanOrEqual(0.5);
    expect(result.pick.min_cons).toBeNumber();
    expect(result.mixed_labels).toBe(5);
    expect(result.pairs.eer).toBe(0);
    expect(controlOk).toBe(true);
    expect(result.control.shuffled_passes).toBe(0);
    // Every grid point is there: 26 ACCEPT x 4 MARGIN x 4 x 4 x 10 MIN_CONS.
    expect(result.grid.length).toBe(26 * 4 * 4 * 4 * 10);
  });

  test("overlapping people give ACCEPT null (positive control of the one above)", () => {
    const r = rng(1);
    const one = voice(r);
    const { learn, mixed } = corpus({
      bases: [one, blend(one, voice(r), 0.02), blend(one, voice(r), 0.02), voice(r)],
      calls: 5,
      spread: 0.15,
    });
    // Their two-voice clusters look like one voice, so the clusters of separable people stand in:
    // MIN_CONS must not be what makes ACCEPT null.
    expect(mixed.length).toBe(5);
    const { result } = evaluateVoices({ learn, mixed: separable().mixed, noGrid: true });
    expect(result.pick.min_cons).toBeNumber();
    expect(result.pick.accept).toBeNull();
    expect(result.pick.margin).toBeNull();
  });

  test("a two-voice cluster sets MIN_CONS above its consistency", () => {
    const { learn, mixed } = separable();
    const { result } = evaluateVoices({ learn, mixed, noGrid: true });
    const top = Math.max(...mixed.flatMap((m) => [m.cons, ...(m.snaps ?? []).map((s) => s.cons)]));
    expect(result.pick.min_cons as number).toBeGreaterThan(top);
  });

  test("with no two-voice cluster nothing is picked, since no MIN_CONS was tested", () => {
    const { learn, mixed } = separable();
    const none = evaluateVoices({ learn, mixed: [], noGrid: true });
    expect(none.result.pick).toMatchObject({ min_cons: null, accept: null, offer: null });
    expect(none.result.mixed_labels).toBe(0);
    expect(none.controlOk).toBe(true);
    // The positive control: the same labels with their clusters pick every value.
    expect(evaluateVoices({ learn, mixed, noGrid: true }).result.pick.min_cons).toBeNumber();
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
    const res = evaluateVoices({ learn: alone.learn, mixed: alone.mixed });
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
    const learn = [...one.learn, ...two.learn];
    const mixed = [...one.mixed, ...two.mixed];
    const { result } = evaluateVoices({ learn, mixed });
    const row = result.grid.find(
      (g) =>
        g.accept === 0.4 &&
        g.margin === 0 &&
        g.min_match_sec === 10 &&
        g.min_learn_sec === 10 &&
        g.min_cons === 0.5,
    );
    expect(row?.wrong_names_enrolled).toBe(0);
    expect(row?.wrong_names_unknown as number).toBeGreaterThan(0);
    expect(row?.wrong_names).toBe(row?.wrong_names_unknown as number);
  });

  test("decide gives a person to one label per call, by descending top score", () => {
    const out = decide(
      [
        [
          { person: 0, s: 0.8 },
          { person: 1, s: 0.3 },
        ],
        [
          { person: 0, s: 0.9 },
          { person: 1, s: 0.7 },
        ],
      ],
      { accept: 0.6, offer: 0.5, margin: 0.15 },
    );
    expect(out[1]).toMatchObject({ status: "named", person: 0 });
    // Person 0 is taken; person 1 at 0.3 is below OFFER.
    expect(out[0]).toMatchObject({ status: "none", person: 1 });
    const close = decide(
      [
        [
          { person: 0, s: 0.8 },
          { person: 1, s: 0.7 },
        ],
      ],
      {
        accept: 0.6,
        offer: 0.5,
        margin: 0.15,
      },
    );
    expect(close[0]?.status).toBe("offered");
  });
});

describe("the gate", () => {
  test("shuffled labels that still pass make the script exit non-zero and write nothing", () => {
    const t = tempDir("akou-eval-voices-");
    try {
      const run = (learn: VoiceLabel[], mixed: MixedLabel[], name: string) => {
        const file = join(t.dir, `${name}.json`);
        const out = join(t.dir, `${name}-out.json`);
        writeFileSync(file, JSON.stringify({ learn: raw(learn), mixed }));
        const p = Bun.spawnSync([process.execPath, SCRIPT, file, "--out", out], {
          stdout: "pipe",
          stderr: "pipe",
        });
        return { code: p.exitCode, out };
      };
      // One person in every call: shuffling changes nothing, so the gate passes shuffled too.
      const r = rng(9);
      const one = corpus({ bases: [voice(r)], calls: 4, spread: 0.1 });
      const bad = run(one.learn, [{ cons: 0.3 }], "one");
      expect(bad.code).toBe(1);
      expect(existsSync(bad.out)).toBe(false);
      // The positive control: people that differ pass, and the file is written.
      const { learn, mixed } = separable();
      const good = run(learn, mixed, "good");
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
    const { learn, mixed } = separable();
    const { result } = evaluateVoices({ learn, mixed, noGrid: true });
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
      // c1 alone is Ana: its two lines (4 s and 3.5 s); c6's half-second line is too short.
      write("c2");
      const one = await voiceLabels(t.dir, io);
      expect(one.labels.map((l) => l.person)).toEqual(["ana"]);
      expect(cuts).toEqual([4, 3.5]);
      expect(one.labels[0]?.secs).toBe(7.5);
      // c2 alone is Ana: its 26 s line is cut at 12 s, as the live worker cuts a line.
      write("c1");
      cuts.length = 0;
      const two = await voiceLabels(t.dir, io);
      expect(cuts).toEqual([12]);
      expect(two.labels[0]?.secs).toBe(12);
      // A part with no audio is counted, and gives no label.
      const none = await voiceLabels(t.dir, { ...io, read: () => null });
      expect(none).toMatchObject({ labels: [], calls: 1, skipped: 1 });
    } finally {
      t.cleanup();
    }
  });
});
