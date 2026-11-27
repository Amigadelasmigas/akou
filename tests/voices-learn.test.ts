/**
 * What a call teaches the voices file (`src/main/voices/learn.ts`) and how the Host learns and
 * matches from a call's events (`src/main/voices/host.ts`), on a real fold and a real voices file
 * in a temporary folder. The thresholds are passed in, so no test depends on the measured values.
 */

import { afterEach, describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { type Ranked, simulate } from "../scripts/eval/voices.ts";
import type { EventDraft, LogEvent } from "../src/core/log/events.ts";
import { CallView } from "../src/core/log/fold.ts";
import { encodeVec } from "../src/main/asr/speakers.ts";
import { LIVE_EMBEDDER, VoiceHost } from "../src/main/voices/host.ts";
import { lessonFor } from "../src/main/voices/learn.ts";
import { rankPeople } from "../src/main/voices/match.ts";
import { type Person, VoiceStore } from "../src/main/voices/store.ts";
import type { VoiceThresholds } from "../src/main/voices/thresholds.ts";
import { LogBuilder, T0, tempDir } from "./helpers.ts";

const MODEL = LIVE_EMBEDDER;
const CALL = "01J8Z6Q4M2VX0K7B3D4E5F6G7H";

const T: VoiceThresholds = {
  accept: 0.8,
  offer: 0.6,
  margin: 0.1,
  minMatchSec: 10,
  minLearnSec: 10,
  score: "best",
};

/** A unit vector `deg` degrees from the first axis toward the second. */
function at(deg: number): Float32Array {
  const r = (deg * Math.PI) / 180;
  return Float32Array.from([Math.cos(r), Math.sin(r), 0, 0]);
}

/** A unit vector `deg` degrees from the first axis toward the third. */
function off(deg: number): Float32Array {
  const r = (deg * Math.PI) / 180;
  return Float32Array.from([Math.cos(r), 0, Math.sin(r), 0]);
}

function centroid(spk: string, vec: Float32Array, secs: number, cons = 0.8): EventDraft {
  return { type: "speaker.centroid", spk, vec: encodeVec(vec), secs, cons };
}

/** A call in memory: a fold the Host reads, a log it writes, and `app.watch` as a direct call. */
class Rig {
  readonly b = new LogBuilder();
  readonly view = new CallView();
  readonly written: LogEvent[] = [];
  readonly store: VoiceStore;
  readonly host: VoiceHost;
  enabled = true;
  closed = false;

  constructor(
    readonly dir: string,
    thresholds: VoiceThresholds = T,
  ) {
    this.store = new VoiceStore(join(dir, "voices.yaml"), { now: () => T0, log: () => {} });
    this.host = new VoiceHost({
      store: this.store,
      enabled: () => this.enabled,
      thresholds: () => thresholds,
      call: (id) =>
        id === CALL
          ? { view: this.view, append: (d) => (this.closed ? null : this.add(d)) }
          : undefined,
      log: () => {},
    });
    this.add(this.b.created() as unknown as EventDraft, true);
    this.add(this.b.partStarted(1, T0) as unknown as EventDraft, true);
  }

  /** Appends one event and hands it to the Host, as the app's watchers are. */
  add(draft: EventDraft, built = false): LogEvent {
    const e = built ? (draft as unknown as LogEvent) : this.b.add(draft);
    this.view.apply(e);
    if (!built) this.written.push(e);
    this.host.onEvent(CALL, e);
    return e;
  }

  /** A line for each label, so the roster lists it. */
  speakers(...spks: string[]): void {
    for (const spk of spks)
      this.add({
        type: "seg",
        id: `l${String(this.b.events.length + 1).padStart(6, "0")}`,
        rev: 1,
        layer: "live",
        part: 1,
        ch: spk === "you" ? "mic" : "call",
        spk,
        a0: 0,
        a1: 1,
        w0: T0,
        w1: T0 + 1000,
        text: "hello",
      });
  }

  async settle(): Promise<void> {
    await this.host.idle();
  }

  voices(): Extract<LogEvent, { type: "speaker.voice" }>[] {
    return this.written.filter(
      (e): e is Extract<LogEvent, { type: "speaker.voice" }> => e.type === "speaker.voice",
    );
  }

  async names(): Promise<Record<string, string[]>> {
    const out: Record<string, string[]> = {};
    for (const p of (await this.store.load(MODEL, 4)).people)
      out[p.name] = p.exemplars.map((e) => `${e.call}/${e.spk}`);
    return out;
  }
}

let dirs: { cleanup: () => void }[] = [];
afterEach(() => {
  for (const d of dirs) d.cleanup();
  dirs = [];
});

function rig(thresholds: VoiceThresholds = T): Rig {
  const d = tempDir("akou-voices-learn-");
  dirs.push(d);
  return new Rig(d.dir, thresholds);
}

async function seed(r: Rig, people: Person[]): Promise<void> {
  for (const p of people) for (const e of p.exemplars) await r.store.upsert(p.name, e);
}

function person(name: string, vecs: Float32Array[]): Person {
  return {
    name,
    exemplars: vecs.map((vec, i) => ({
      model: MODEL,
      dim: vec.length,
      vec,
      secs: 60,
      cons: 0.8,
      call: `earlier-${name}-${i}`,
      spk: "c1",
      at: "2026-09-01T10:00:00.000Z",
    })),
  };
}

describe("the learn rule", () => {
  function view(...drafts: EventDraft[]): CallView {
    const b = new LogBuilder();
    b.created();
    b.partStarted(1, T0);
    b.seg({ id: "l000001", spk: "c2", text: "hello" });
    b.seg({ id: "l000002", spk: "c3", text: "hello" });
    b.seg({ id: "l000003", ch: "mic", spk: "you", text: "hello" });
    for (const d of drafts) b.add(d);
    const v = new CallView();
    for (const e of b.events) v.apply(e);
    return v;
  }
  const c = { vec: at(0), secs: 30, cons: 0.7 };
  const lesson = (v: CallView, spk = "c2", over: Partial<Parameters<typeof lessonFor>[0]> = {}) =>
    lessonFor({
      view: v,
      call: CALL,
      spk,
      model: MODEL,
      centroid: c,
      minLearnSec: 10,
      at: "2026-11-27T10:00:00.000Z",
      ...over,
    });
  const name = (spk: string, n: string, by: string): EventDraft => ({
    type: "speaker.name",
    spk,
    name: n,
    by,
  });

  test("[VO-5] only names by user and agent:cli teach", () => {
    // Positive control: a name the user typed teaches, and so does one from the CLI.
    expect(lesson(view(name("c2", "Ben", "user")))).toMatchObject({
      name: "Ben",
      exemplar: { model: MODEL, dim: 4, secs: 30, cons: 0.7, call: CALL, spk: "c2" },
    });
    expect(lesson(view(name("c2", "Ben", "agent:cli")))?.name).toBe("Ben");
    // akou's own names (an import), an MCP agent's, a voice match and an offer never teach.
    expect(lesson(view(name("c2", "Ben", "app")))).toBeNull();
    expect(lesson(view(name("c2", "Ben", "agent:claude-code")))).toBeNull();
    expect(lesson(view(name("c2", "Ben", "agent:mcp")))).toBeNull();
    const voice = (status: "named" | "offered"): EventDraft => ({
      type: "speaker.voice",
      spk: "c2",
      name: "Ben",
      status,
      score: 0.9,
      by: "app",
    });
    expect(lesson(view(voice("named")))).toBeNull();
    expect(lesson(view(voice("offered")))).toBeNull();
  });

  test("only a c<N> merge root with a full centroid of MIN_LEARN_SEC teaches", () => {
    const named = view(name("c2", "Ben", "user"), name("you", "Ana", "user"));
    expect(lesson(named)).not.toBeNull();
    expect(lesson(named, "you")).toBeNull();
    expect(lesson(view(name("s1", "Ben", "user")), "s1")).toBeNull();
    expect(lesson(view(name("c?", "Ben", "user")), "c?")).toBeNull();
    // Merged into another label: its centroid is no longer the group's voice.
    expect(
      lesson(view(name("c2", "Ben", "user"), { type: "speaker.merge", from: "c2", into: "c3" })),
    ).toBeNull();
    expect(lesson(named, "c2", { centroid: undefined })).toBeNull();
    expect(lesson(named, "c2", { centroid: { vec: at(0), cons: 0.7 } })).toBeNull();
    expect(lesson(named, "c2", { centroid: { vec: at(0), secs: 30 } })).toBeNull();
    expect(lesson(named, "c2", { centroid: { ...c, secs: 9.9 } })).toBeNull();
    expect(lesson(named, "c2", { minLearnSec: null })).toBeNull();
    // A low consistency gates nothing.
    expect(lesson(named, "c2", { centroid: { ...c, cons: 0.1 } })).not.toBeNull();
  });
});

describe("the Host learns", () => {
  test("[VO-5] a typed name teaches the label's centroid; an MCP name and an app name do not", async () => {
    const r = rig();
    r.speakers("c1", "c2", "c3");
    r.add(centroid("c1", at(0), 30));
    r.add(centroid("c2", at(90), 30));
    r.add(centroid("c3", off(90), 30));
    r.add({ type: "speaker.name", spk: "c1", name: "Ben", by: "user" });
    r.add({ type: "speaker.name", spk: "c2", name: "Cy", by: "agent:claude-code" });
    r.add({ type: "speaker.name", spk: "c3", name: "Di", by: "app" });
    await r.settle();
    expect(await r.names()).toEqual({ Ben: [`${CALL}/c1`] });
    // A later centroid of the named label is kept in place of the first.
    r.add(centroid("c1", at(3), 60));
    await r.settle();
    const ben = (await r.store.load(MODEL, 4)).people.find((p) => p.name === "Ben");
    expect(ben?.exemplars).toHaveLength(1);
    expect(ben?.exemplars[0]?.secs).toBe(60);
  });

  test("a label under MIN_LEARN_SEC teaches once it has the speech", async () => {
    const r = rig();
    r.speakers("c1");
    r.add(centroid("c1", at(0), 5));
    r.add({ type: "speaker.name", spk: "c1", name: "Ben", by: "user" });
    await r.settle();
    expect(existsSync(r.store.path)).toBe(false);
    r.add(centroid("c1", at(0), 12));
    await r.settle();
    expect(await r.names()).toEqual({ Ben: [`${CALL}/c1`] });
  });

  test("a rename in the call moves the exemplar to the new name", async () => {
    const r = rig();
    r.speakers("c1");
    r.add(centroid("c1", at(0), 30));
    r.add({ type: "speaker.name", spk: "c1", name: "Ben", by: "user" });
    await r.settle();
    expect(await r.names()).toEqual({ Ben: [`${CALL}/c1`] });
    r.add({ type: "speaker.name", spk: "c1", name: "Ben Ortiz", by: "user" });
    await r.settle();
    expect(await r.names()).toEqual({ "Ben Ortiz": [`${CALL}/c1`] });
  });

  test("a typed name after a voice name wins and teaches", async () => {
    const r = rig();
    await seed(r, [person("Ben", [at(0)])]);
    r.speakers("c1");
    r.add(centroid("c1", at(2), 30));
    await r.settle();
    expect(r.voices()).toMatchObject([{ spk: "c1", name: "Ben", status: "named", by: "app" }]);
    // Named by voice: nothing taught from this call yet.
    expect(await r.names()).toEqual({ Ben: ["earlier-Ben-0/c1"] });
    r.add({ type: "speaker.name", spk: "c1", name: "Benjamin", by: "user" });
    await r.settle();
    expect(r.view.speakerLabel("c1")).toBe("Benjamin");
    expect(await r.names()).toEqual({ Ben: ["earlier-Ben-0/c1"], Benjamin: [`${CALL}/c1`] });
    // Typed: the label is left alone at its next centroid, even though Ben still matches.
    const before = r.voices().length;
    r.add(centroid("c1", at(1), 60));
    await r.settle();
    expect(r.voices()).toHaveLength(before);
  });

  test("voices.enabled off stops learning and matching and keeps the file", async () => {
    for (const on of [true, false]) {
      const r = rig();
      await seed(r, [person("Ben", [at(0)])]);
      const file = readFileSync(r.store.path, "utf8");
      r.enabled = on;
      r.speakers("c1", "c2");
      r.add(centroid("c1", at(2), 30));
      r.add(centroid("c2", at(90), 30));
      r.add({ type: "speaker.name", spk: "c2", name: "Cy", by: "user" });
      await r.settle();
      if (on) {
        // The positive control: on, c1 is named and Cy is learnt.
        expect(r.voices()).toMatchObject([{ spk: "c1", name: "Ben", status: "named" }]);
        expect(Object.keys(await r.names()).sort()).toEqual(["Ben", "Cy"]);
      } else {
        expect(r.voices()).toEqual([]);
        expect(readFileSync(r.store.path, "utf8")).toBe(file);
      }
    }
  });
});

describe("the Host matches", () => {
  test("a voice is named with the score to 2 places, offered, or left alone", async () => {
    const r = rig();
    await seed(r, [person("Ben", [at(0)]), person("Cy", [off(90)])]);
    r.speakers("c1", "c2", "c3");
    r.add(centroid("c1", at(2), 30)); // cos 0.9994: named
    r.add(centroid("c2", at(140), 30)); // Ben -0.77, Cy 0: nobody
    r.add(centroid("c3", off(40), 30)); // Ben taken; Cy cos 50 = 0.64: offered
    await r.settle();
    expect(r.voices()).toEqual([
      expect.objectContaining({ spk: "c1", name: "Ben", status: "named", score: 1, by: "app" }),
      expect.objectContaining({ spk: "c3", name: "Cy", status: "offered", score: 0.64, by: "app" }),
    ]);
  });

  test("a label under MIN_MATCH_SEC, or with thresholds that decide nothing, is left alone", async () => {
    const none: VoiceThresholds = { ...T, accept: null, offer: null };
    for (const [th, secs, want] of [
      [T, 9, 0],
      [T, 10, 1],
      [none, 30, 0],
      [{ ...T, minMatchSec: null }, 30, 0],
    ] as const) {
      const r = rig(th);
      await seed(r, [person("Ben", [at(0)])]);
      r.speakers("c1");
      r.add(centroid("c1", at(2), secs));
      await r.settle();
      expect(r.voices()).toHaveLength(want);
    }
  });

  test("a call's own exemplars never name its other labels", async () => {
    const r = rig();
    r.speakers("c1", "c2");
    r.add(centroid("c1", at(0), 30));
    r.add({ type: "speaker.name", spk: "c1", name: "Ben", by: "user" });
    await r.settle();
    // c1 is renamed: Ben is no longer taken in this call, and only this call taught Ben.
    r.add({ type: "speaker.name", spk: "c1", name: "Bo", by: "agent:claude-code" });
    r.add(centroid("c2", at(1), 30));
    await r.settle();
    expect(r.voices()).toEqual([]);
    // The positive control: Ben taught by an earlier call names c2.
    await seed(r, [person("Ben", [at(0)])]);
    r.add(centroid("c2", at(1), 40));
    await r.settle();
    expect(r.voices()).toMatchObject([{ spk: "c2", name: "Ben", status: "named" }]);
  });
});

describe("the Host and the voices evaluation make the same decisions", () => {
  // Ana and Bo are two voices in one plane, Cy is out of it.
  const people = [person("Ana", [at(0), at(4)]), person("Bo", [at(90)]), person("Cy", [off(70)])];
  /** One recorded call: every centroid write, with its call time in seconds. */
  const writes: { spk: string; t: number; secs: number; vec: Float32Array }[] = [
    { spk: "c3", t: 5, secs: 5, vec: at(3) }, // under MIN_MATCH_SEC
    { spk: "c1", t: 10, secs: 12, vec: at(5) }, // Ana named
    { spk: "c2", t: 20, secs: 15, vec: at(40) }, // Ana taken: Bo offered
    { spk: "c1", t: 30, secs: 40, vec: at(80) }, // drifts off Ana: withdrawn, Bo taken
    { spk: "c2", t: 40, secs: 30, vec: at(85) }, // Bo raised to a name
    { spk: "c3", t: 50, secs: 20, vec: at(3) }, // Ana free on c3: named
  ];
  const labels = ["c1", "c2", "c3"];

  /** What the evaluation's simulation says, per label. */
  function evaluation() {
    const ranked: Ranked[][] = labels.map((spk) =>
      writes
        .filter((w) => w.spk === spk)
        .map((w) => ({ t: w.t, secs: w.secs, ranked: rankPeople(people, MODEL, 4, w.vec) })),
    );
    return simulate(ranked, T).map((o) => ({
      named: [...o.named].sort(),
      offered: [...o.offered].sort(),
      final: o.final,
      withdrawn: o.withdrawn,
    }));
  }

  /** What the Host writes when the same writes land in `order`. */
  async function host(order: typeof writes) {
    const r = rig();
    await seed(r, people);
    r.speakers(...labels);
    for (const w of order) {
      r.add(centroid(w.spk, w.vec, w.secs));
      await r.settle();
    }
    const v = r.voices();
    return labels.map((spk) => {
      const mine = v.filter((e) => e.spk === spk);
      const final = r.view.voiceFor(spk);
      return {
        named: [...new Set(mine.filter((e) => e.status === "named").map((e) => e.name))].sort(),
        offered: [...new Set(mine.filter((e) => e.status === "offered").map((e) => e.name))].sort(),
        final: final ? { name: final.name, status: final.status } : null,
        withdrawn: mine.filter((e) => e.status === "refused" && e.by === "app").length,
      };
    });
  }

  test("one recorded call replayed through both gives the same speaker.voice decisions", async () => {
    const want = evaluation();
    // The sequence exercises a name, an offer, a withdrawal and a raise.
    expect(want).toEqual([
      { named: ["Ana"], offered: [], final: null, withdrawn: 1 },
      { named: ["Bo"], offered: ["Bo"], final: { name: "Bo", status: "named" }, withdrawn: 0 },
      { named: ["Ana"], offered: [], final: { name: "Ana", status: "named" }, withdrawn: 0 },
    ]);
    expect(await host(writes)).toEqual(want);
    // The positive control: the same writes in another order decide otherwise, so the comparison
    // tells the order apart.
    const swapped = [...writes];
    [swapped[3], swapped[5]] = [
      swapped[5] as (typeof writes)[number],
      swapped[3] as (typeof writes)[number],
    ];
    expect(await host(swapped)).not.toEqual(want);
  });
});
