#!/usr/bin/env bun

/**
 * Whether a voice is recognisable across calls, measured before akou names anyone by voice
 * (docs/TESTING.md TS-21b). The input is a list of labels: one live speaker of one call, the
 * person it is, and its voiceprint (`src/main/asr/voiceprint.ts`). The output is numbers only.
 *
 *   bun scripts/eval/voices.ts <labels.json> [--out <file>] [--max-wrong 0]
 *
 * `scripts/eval-local.ts --voices` builds the labels from your own calls and
 * `scripts/eval/voices-ami.ts` from the AMI Meeting Corpus; both call `evaluateVoices` here. The
 * labels file holds `{learn, match?, mixed?}`, each label `{call, person, vec: number[], secs,
 * cons, at?, snaps?}`; `match` defaults to `learn`.
 *
 * - **The learn rule.** A label becomes an exemplar of its person when `secs >= MIN_LEARN_SEC` and
 *   `cons >= MIN_CONS`. Each person keeps the `KEEP` newest (by `at`).
 * - **The match rule.** A label is eligible once it has `MIN_MATCH_SEC` of speech, judged on its
 *   voiceprint at that moment (`snaps`), with `cons >= MIN_CONS`. A person's score is their best
 *   exemplar. Labels are taken by descending top score, and a person goes to at most one label per
 *   call. With p1 the top score and p2 the runner-up: named when p1 >= ACCEPT and
 *   p1 - p2 >= MARGIN, else offered when p1 >= OFFER.
 * - **Leave one call out.** A call is scored against a store learnt from every other call.
 * - **Unknown voices.** A second pass drops each person of the call from the store in turn; any
 *   name given to their label is wrong.
 * - **The grid** sweeps ACCEPT, MARGIN, MIN_MATCH_SEC, MIN_LEARN_SEC and MIN_CONS with offers off;
 *   the OFFER sweep runs at the picked point.
 * - **The pick.** MIN_CONS is the lowest value that refuses every synthetic two-voice cluster
 *   (`mixed`) and keeps at least 90 % of the single-voice labels. Then, at that MIN_CONS, the
 *   first point by MIN_MATCH_SEC, MIN_LEARN_SEC, ACCEPT and MARGIN, each ascending, whose
 *   wrong-name rate is at most `--max-wrong` and which names at least half the eligible known
 *   voices. With no such point ACCEPT is null: akou only offers. OFFER is the lowest value with at
 *   most 10 % wrong offers.
 * - **The positive control.** The same run with the person labels shuffled must fail the gate.
 *   When it passes, the data cannot tell people apart, and the script exits 1 and writes nothing.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { defaultModelsDir, modelFile } from "../../src/main/asr/models.ts";
import { Accum, score } from "../../src/main/asr/voiceprint.ts";
import { percentile } from "./score.ts";

function range(from: number, to: number, step: number): number[] {
  const out: number[] = [];
  for (let i = 0; from + i * step <= to + 1e-9; i++)
    out.push(Math.round((from + i * step) * 100) / 100);
  return out;
}

export const GRID = {
  accept: range(0.4, 0.9, 0.02),
  margin: [0, 0.05, 0.1, 0.15],
  minMatchSec: [10, 20, 30, 60],
  minLearnSec: [10, 20, 30, 60],
  minCons: range(0.5, 0.95, 0.05),
  offer: range(0.3, 0.9, 0.02),
};

/** Exemplars kept per person. */
export const KEEP = 8;
/** The share of eligible known voices that must be named, or ACCEPT is null. */
export const MIN_NAMED = 0.5;
/** The most wrong offers among offers. */
export const MAX_WRONG_OFFERS = 0.1;
/** The share of single-voice labels MIN_CONS must keep. */
export const MIN_SINGLE_KEPT = 0.9;

/** A label's voiceprint when it first reached `min` seconds (a MIN_MATCH_SEC grid value). */
export interface Snap {
  min: number;
  vec: Float32Array;
  cons: number;
}

export interface VoiceLabel {
  call: string;
  person: string;
  /** The voiceprint of all its lines: unit length. */
  vec: Float32Array;
  secs: number;
  cons: number;
  /** Newer is larger; defaults to the label's place in the list. */
  at?: number;
  /** Length of the mean raw embedding. */
  norm?: number;
  /** Absent: the full voiceprint stands in at every MIN_MATCH_SEC it reaches. */
  snaps?: Snap[];
}

/** A synthetic cluster of two voices: only its consistency is read. */
export interface MixedLabel {
  cons: number;
  snaps?: Snap[];
}

export interface Thresholds {
  accept: number | null;
  offer: number | null;
  margin: number;
  minMatchSec: number;
  minLearnSec: number;
  minCons: number;
}

/** Builds one label from its lines, recording a snapshot as it reaches each grid MIN_MATCH_SEC. */
export class LabelBuilder {
  private readonly acc = new Accum();
  private readonly snaps: Snap[] = [];

  add(vec: Float32Array, secs: number): void {
    if (!this.acc.add(vec, secs)) return;
    for (const min of GRID.minMatchSec) {
      if (this.acc.secs < min || this.snaps.some((s) => s.min === min)) continue;
      const r = this.acc.result();
      if (r) this.snaps.push({ min, vec: r.vec, cons: r.cons });
    }
  }

  get secs(): number {
    return this.acc.secs;
  }

  label(call: string, person: string, at?: number): VoiceLabel | null {
    const r = this.acc.result();
    if (!r) return null;
    return {
      call,
      person,
      vec: r.vec,
      secs: r.secs,
      cons: r.cons,
      norm: r.norm,
      at,
      snaps: this.snaps.map((s) => ({ ...s })),
    };
  }
}

// ---------------------------------------------------------------------------
// The rules

interface Lab {
  call: string;
  person: number;
  vec: Float32Array;
  secs: number;
  cons: number;
  at: number;
  snaps?: Snap[];
}

/** The voiceprint a label is matched on at `minMatch`, or null while it has less speech. */
function matchView(l: Lab, minMatch: number): { vec: Float32Array; cons: number } | null {
  if (l.snaps) return l.snaps.find((s) => s.min === minMatch) ?? null;
  return l.secs >= minMatch ? l : null;
}

/** Each person's exemplars under the learn rule: the `KEEP` newest eligible labels. */
function learn(labels: readonly Lab[], minLearn: number, minCons: number) {
  const by = new Map<number, Lab[]>();
  for (const l of labels) {
    if (l.secs < minLearn || l.cons < minCons) continue;
    const list = by.get(l.person) ?? [];
    list.push(l);
    by.set(l.person, list);
  }
  const store = new Map<number, Float32Array[]>();
  for (const [p, list] of by)
    store.set(
      p,
      list
        .sort((a, b) => b.at - a.at)
        .slice(0, KEEP)
        .map((l) => l.vec),
    );
  return store;
}

interface Ranked {
  person: number;
  s: number;
}

/** Every person's best exemplar score, highest first; exemplars of another size are skipped. */
function rank(vec: Float32Array, store: Map<number, Float32Array[]>): Ranked[] {
  const out: Ranked[] = [];
  for (const [person, ex] of store) {
    let best: number | null = null;
    for (const e of ex) {
      const s = score(vec, e);
      if (s !== null && (best === null || s > best)) best = s;
    }
    if (best !== null) out.push({ person, s: best });
  }
  return out.sort((a, b) => b.s - a.s || a.person - b.person);
}

export type Status = "named" | "offered" | "none";

/**
 * The greedy one-to-one match of one call's eligible labels: the label with the highest top score
 * among people not yet given out goes first.
 */
export function decide(
  ranked: readonly (readonly Ranked[])[],
  t: { accept: number | null; offer: number | null; margin: number },
): { status: Status; person: number; s: number }[] {
  const out = ranked.map(() => ({ status: "none" as Status, person: -1, s: 0 }));
  const taken = new Set<number>();
  const done = new Set<number>();
  while (done.size < ranked.length) {
    let pick = -1;
    let p1: Ranked | undefined;
    let p2: Ranked | undefined;
    for (let i = 0; i < ranked.length; i++) {
      if (done.has(i)) continue;
      const free = (ranked[i] as readonly Ranked[]).filter((r) => !taken.has(r.person));
      const top = free[0];
      if (!top) continue;
      if (!p1 || top.s > p1.s) {
        pick = i;
        p1 = top;
        p2 = free[1];
      }
    }
    if (pick < 0 || !p1) break;
    done.add(pick);
    const gap = p1.s - (p2?.s ?? -1);
    const status: Status =
      t.accept !== null && p1.s >= t.accept && gap >= t.margin
        ? "named"
        : t.offer !== null && p1.s >= t.offer
          ? "offered"
          : "none";
    if (status !== "none") taken.add(p1.person);
    out[pick] = { status, person: p1.person, s: p1.s };
  }
  return out;
}

// ---------------------------------------------------------------------------
// Leave one call out, and the unknown-voice pass

interface Case {
  call: string;
  /** The person dropped from the store: the unknown-voice pass. -1 for the known pass. */
  drop: number;
}

interface Cand {
  person: number;
  target: boolean;
  /** The label's person is in the store. */
  known: boolean;
  ranked: Ranked[];
}

interface Prepared {
  learn: Lab[];
  match: Lab[];
  calls: string[];
  cases: Case[];
}

function prepare(learnIn: readonly VoiceLabel[], matchIn: readonly VoiceLabel[]): Prepared {
  const ids = new Map<string, number>();
  const id = (p: string) => {
    let n = ids.get(p);
    if (n === undefined) {
      n = ids.size;
      ids.set(p, n);
    }
    return n;
  };
  const conv = (l: VoiceLabel, i: number): Lab => ({
    call: l.call,
    person: id(l.person),
    vec: l.vec,
    secs: l.secs,
    cons: l.cons,
    at: l.at ?? i,
    snaps: l.snaps,
  });
  const learn = learnIn.map(conv);
  const match = learnIn === matchIn ? learn : matchIn.map(conv);
  const calls = [...new Set(match.map((l) => l.call))].sort();
  const cases: Case[] = [];
  for (const call of calls) {
    cases.push({ call, drop: -1 });
    for (const p of new Set(match.filter((l) => l.call === call).map((l) => l.person)))
      cases.push({ call, drop: p });
  }
  return { learn, match, calls, cases };
}

/** Each case's eligible labels, ranked against its own store, at one setting. */
function rankCases(d: Prepared, minLearn: number, minCons: number, minMatch: number): Cand[][] {
  return d.cases.map((c) => {
    const store = learn(
      d.learn.filter((l) => l.call !== c.call && l.person !== c.drop),
      minLearn,
      minCons,
    );
    const out: Cand[] = [];
    for (const l of d.match) {
      if (l.call !== c.call) continue;
      const v = matchView(l, minMatch);
      if (!v || v.cons < minCons) continue;
      out.push({
        person: l.person,
        target: c.drop < 0 || l.person === c.drop,
        known: store.has(l.person),
        ranked: rank(v.vec, store),
      });
    }
    return out;
  });
}

interface Tally {
  /** Known pass: eligible labels whose person is in the store, and of them, named right. */
  eligibleKnown: number;
  correct: number;
  /** Known pass: every eligible label, and the wrong names among them. */
  eligible: number;
  wrongKnown: number;
  /** Wrong names between two people both in the store. */
  wrongEnrolled: number;
  /** Unknown pass: eligible labels of the dropped person, and the names they got. */
  eligibleUnknown: number;
  wrongUnknown: number;
  offers: number;
  wrongOffers: number;
}

function tally(
  d: Prepared,
  ranked: readonly Cand[][],
  t: { accept: number | null; offer: number | null; margin: number },
): Tally {
  const n: Tally = {
    eligibleKnown: 0,
    correct: 0,
    eligible: 0,
    wrongKnown: 0,
    wrongEnrolled: 0,
    eligibleUnknown: 0,
    wrongUnknown: 0,
    offers: 0,
    wrongOffers: 0,
  };
  d.cases.forEach((c, k) => {
    const cands = ranked[k] as Cand[];
    const res = decide(
      cands.map((x) => x.ranked),
      t,
    );
    cands.forEach((x, i) => {
      if (!x.target) return;
      const r = res[i] as { status: Status; person: number };
      const right = r.person === x.person;
      if (c.drop < 0) {
        n.eligible++;
        if (x.known) n.eligibleKnown++;
        if (r.status === "named" && right) n.correct++;
        if (r.status === "named" && !right) {
          n.wrongKnown++;
          if (x.known) n.wrongEnrolled++;
        }
      } else {
        n.eligibleUnknown++;
        if (r.status === "named") n.wrongUnknown++;
      }
      if (r.status === "offered") {
        n.offers++;
        if (c.drop >= 0 || !right) n.wrongOffers++;
      }
    });
  });
  return n;
}

const rate = (a: number, b: number) => (b > 0 ? Math.round((a / b) * 10000) / 10000 : null);

function wrongRate(n: Tally): number {
  return rate(n.wrongKnown + n.wrongUnknown, n.eligible + n.eligibleUnknown) ?? 0;
}

function meets(n: Tally, maxWrong: number): boolean {
  return (
    n.eligibleKnown > 0 && wrongRate(n) <= maxWrong && n.correct / n.eligibleKnown >= MIN_NAMED
  );
}

// ---------------------------------------------------------------------------
// The result

export interface GridRow {
  accept: number;
  margin: number;
  min_match_sec: number;
  min_learn_sec: number;
  min_cons: number;
  eligible_share: number | null;
  correct_name_rate: number | null;
  wrong_name_rate: number | null;
  wrong_names: number;
  wrong_names_enrolled: number;
  wrong_names_unknown: number;
}

export interface VoicesResult {
  labels: number;
  people: number;
  calls: number;
  mixed_labels: number;
  mean_raw_norm: number | null;
  pairs: {
    same: number;
    different: number;
    same_quantiles: Record<string, number> | null;
    different_quantiles: Record<string, number> | null;
    eer: number | null;
  };
  min_cons: { min_cons: number; single_kept: number | null; mixed_refused: number }[];
  pick: {
    accept: number | null;
    offer: number | null;
    margin: number | null;
    min_match_sec: number | null;
    min_learn_sec: number | null;
    min_cons: number | null;
    correct_name_rate: number | null;
    wrong_name_rate: number | null;
    offer_rate: number | null;
    wrong_offer_rate: number | null;
  };
  offers: {
    offer: number;
    offers: number;
    wrong_offers: number;
    offer_rate: number | null;
    wrong_offer_rate: number | null;
  }[];
  control: { shuffled_passes: number };
  grid: GridRow[];
}

const QUANTILES = [5, 25, 50, 75, 95];

function quantiles(xs: readonly number[]): Record<string, number> | null {
  if (xs.length === 0) return null;
  return Object.fromEntries(
    QUANTILES.map((p) => [`p${p}`, Math.round(percentile(xs, p) * 10000) / 10000]),
  );
}

/** The equal error rate of same-person and different-person scores, or null without both. */
export function eer(same: readonly number[], diff: readonly number[]): number | null {
  if (same.length === 0 || diff.length === 0) return null;
  const s = [...same].sort((a, b) => a - b);
  const d = [...diff].sort((a, b) => a - b);
  const below = (xs: number[], t: number) => {
    let lo = 0;
    let hi = xs.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if ((xs[mid] as number) < t) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
  let best = Number.POSITIVE_INFINITY;
  let out = 1;
  for (const t of [...s, ...d, Number.POSITIVE_INFINITY]) {
    const frr = below(s, t) / s.length;
    const far = (d.length - below(d, t)) / d.length;
    if (Math.abs(far - frr) < best) {
      best = Math.abs(far - frr);
      out = (far + frr) / 2;
    }
  }
  return Math.round(out * 10000) / 10000;
}

function pairs(d: Prepared) {
  const same: number[] = [];
  const diff: number[] = [];
  const one = d.learn === d.match;
  d.match.forEach((a, i) => {
    d.learn.forEach((b, j) => {
      if ((one && j <= i) || a.call === b.call) return;
      const s = score(a.vec, b.vec);
      if (s === null) return;
      (a.person === b.person ? same : diff).push(s);
    });
  });
  return { same, diff };
}

function minConsTable(match: readonly Lab[], mixed: readonly MixedLabel[]) {
  return GRID.minCons.map((v) => ({
    min_cons: v,
    single_kept: rate(match.filter((l) => l.cons >= v).length, match.length),
    mixed_refused: mixed.filter((m) => m.cons < v && (m.snaps ?? []).every((s) => s.cons < v))
      .length,
  }));
}

function pickPoint(d: Prepared, minCons: number, maxWrong: number) {
  for (const minMatch of GRID.minMatchSec)
    for (const minLearn of GRID.minLearnSec) {
      const ranked = rankCases(d, minLearn, minCons, minMatch);
      for (const accept of GRID.accept)
        for (const margin of GRID.margin) {
          const n = tally(d, ranked, { accept, offer: null, margin });
          if (meets(n, maxWrong)) return { accept, margin, minMatch, minLearn, n };
        }
    }
  return null;
}

/** People shuffled among the labels: one person per (call, person) key, in both sets. */
export function shuffled(
  learn: readonly VoiceLabel[],
  match: readonly VoiceLabel[],
  seed = 1,
): { learn: VoiceLabel[]; match: VoiceLabel[] } {
  const key = (l: VoiceLabel) => `${l.call}\u0000${l.person}`;
  const keys = [...new Set([...learn, ...match].map(key))].sort();
  const people = keys.map((k) => k.split("\u0000")[1] as string);
  let s = seed >>> 0;
  const rnd = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  for (let i = people.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [people[i], people[j]] = [people[j] as string, people[i] as string];
  }
  const to = new Map(keys.map((k, i) => [k, people[i] as string]));
  const swap = (l: VoiceLabel) => ({ ...l, person: to.get(key(l)) as string });
  const ls = learn.map(swap);
  return { learn: ls, match: learn === match ? ls : match.map(swap) };
}

export interface VoicesInput {
  learn: readonly VoiceLabel[];
  /** The labels matched; defaults to `learn` (another channel for the far-field run). */
  match?: readonly VoiceLabel[];
  mixed?: readonly MixedLabel[];
  /** The highest wrong-name rate the gate allows: 0 on your own calls, 0.005 on AMI. */
  maxWrong?: number;
  /** Skip the full grid in the result (the pick and the control still run). */
  noGrid?: boolean;
}

/** The pick alone, for the positive control. */
function pickOnly(
  learnIn: readonly VoiceLabel[],
  matchIn: readonly VoiceLabel[],
  mixed: readonly MixedLabel[],
  maxWrong: number,
) {
  const d = prepare(learnIn, matchIn);
  const minCons = minConsTable(d.match, mixed).find(
    (r) => r.mixed_refused === mixed.length && (r.single_kept ?? 0) >= MIN_SINGLE_KEPT,
  )?.min_cons;
  if (minCons === undefined) return { d, minCons: null, point: null };
  return { d, minCons, point: pickPoint(d, minCons, maxWrong) };
}

/** The whole measurement, and whether its positive control failed as it must. */
export function evaluateVoices(input: VoicesInput): { result: VoicesResult; controlOk: boolean } {
  const learnIn = input.learn;
  const matchIn = input.match ?? input.learn;
  const mixed = input.mixed ?? [];
  const maxWrong = input.maxWrong ?? 0;
  const { d, minCons, point } = pickOnly(learnIn, matchIn, mixed, maxWrong);
  const p = pairs(d);
  const norms = [...learnIn, ...(matchIn === learnIn ? [] : matchIn)]
    .map((l) => l.norm)
    .filter((x): x is number => typeof x === "number");

  const grid: GridRow[] = [];
  if (!input.noGrid)
    for (const minLearn of GRID.minLearnSec)
      for (const mc of GRID.minCons)
        for (const minMatch of GRID.minMatchSec) {
          const ranked = rankCases(d, minLearn, mc, minMatch);
          const total = d.match.length;
          for (const margin of GRID.margin)
            for (const accept of GRID.accept) {
              const n = tally(d, ranked, { accept, offer: null, margin });
              grid.push({
                accept,
                margin,
                min_match_sec: minMatch,
                min_learn_sec: minLearn,
                min_cons: mc,
                eligible_share: rate(n.eligible, total),
                correct_name_rate: rate(n.correct, n.eligibleKnown),
                wrong_name_rate: rate(
                  n.wrongKnown + n.wrongUnknown,
                  n.eligible + n.eligibleUnknown,
                ),
                wrong_names: n.wrongKnown + n.wrongUnknown,
                wrong_names_enrolled: n.wrongEnrolled,
                wrong_names_unknown: n.wrongUnknown,
              });
            }
        }

  const offers: VoicesResult["offers"] = [];
  let offer: number | null = null;
  let chosen: Tally | null = point?.n ?? null;
  if (minCons !== null) {
    const minMatch = point?.minMatch ?? (GRID.minMatchSec[0] as number);
    const minLearn = point?.minLearn ?? (GRID.minLearnSec[0] as number);
    const ranked = rankCases(d, minLearn, minCons, minMatch);
    for (const o of GRID.offer) {
      if (point && o >= point.accept) break;
      const n = tally(d, ranked, {
        accept: point?.accept ?? null,
        offer: o,
        margin: point?.margin ?? 0,
      });
      offers.push({
        offer: o,
        offers: n.offers,
        wrong_offers: n.wrongOffers,
        offer_rate: rate(n.offers, n.eligible + n.eligibleUnknown),
        wrong_offer_rate: rate(n.wrongOffers, n.offers),
      });
      if (offer === null && n.offers > 0 && n.wrongOffers / n.offers <= MAX_WRONG_OFFERS) {
        offer = o;
        if (point) chosen = n;
      }
    }
  }

  // The positive control: shuffled people at the picked point, or the whole pick when none.
  const sh = shuffled(learnIn, matchIn);
  const shMatch = matchIn === learnIn ? sh.learn : sh.match;
  let shuffledPasses: boolean;
  if (point && minCons !== null) {
    const ds = prepare(sh.learn, shMatch);
    const n = tally(ds, rankCases(ds, point.minLearn, minCons, point.minMatch), {
      accept: point.accept,
      offer: null,
      margin: point.margin,
    });
    shuffledPasses = meets(n, maxWrong);
  } else shuffledPasses = pickOnly(sh.learn, shMatch, mixed, maxWrong).point !== null;

  const offerRow = offers.find((r) => r.offer === offer);
  const result: VoicesResult = {
    labels: d.match.length,
    people: new Set(d.match.map((l) => l.person)).size,
    calls: d.calls.length,
    mixed_labels: mixed.length,
    mean_raw_norm:
      norms.length > 0
        ? Math.round((norms.reduce((a, b) => a + b, 0) / norms.length) * 10000) / 10000
        : null,
    pairs: {
      same: p.same.length,
      different: p.diff.length,
      same_quantiles: quantiles(p.same),
      different_quantiles: quantiles(p.diff),
      eer: eer(p.same, p.diff),
    },
    min_cons: minConsTable(d.match, mixed),
    pick: {
      accept: point?.accept ?? null,
      offer,
      margin: point?.margin ?? null,
      min_match_sec: point?.minMatch ?? null,
      min_learn_sec: point?.minLearn ?? null,
      min_cons: minCons,
      correct_name_rate: chosen ? rate(chosen.correct, chosen.eligibleKnown) : null,
      wrong_name_rate: chosen ? wrongRate(chosen) : null,
      offer_rate: offerRow?.offer_rate ?? null,
      wrong_offer_rate: offerRow?.wrong_offer_rate ?? null,
    },
    offers,
    control: { shuffled_passes: shuffledPasses ? 1 : 0 },
    grid,
  };
  return { result, controlOk: !shuffledPasses };
}

// ---------------------------------------------------------------------------
// Writing it, and the embedder

/**
 * The JSON to write, or an error when any value is not a number, a boolean or null: keys are the
 * only strings, so no name or word of a call can reach the file.
 */
export function numbersJson(result: object): string {
  const walk = (v: unknown, at: string): void => {
    if (v === null || typeof v === "boolean") return;
    if (typeof v === "number") {
      if (!Number.isFinite(v)) throw new Error(`${at} is not a finite number`);
      return;
    }
    if (typeof v === "object") {
      for (const [k, x] of Object.entries(v as Record<string, unknown>)) walk(x, `${at}.${k}`);
      return;
    }
    throw new Error(`${at} is a ${typeof v}: the voices evaluation writes numbers only`);
  };
  walk(result, "result");
  return `${JSON.stringify(result)}\n`;
}

/**
 * Writes the result of `evaluateVoices`, or refuses: exit 1 when the shuffled control passed,
 * 65 when a value is not a number. Returns the exit code.
 */
export function finish(
  run: { result: object; controlOk: boolean },
  out: string | undefined,
  who: string,
): number {
  if (!run.controlOk) {
    console.error(
      `${who}: the positive control failed: with people shuffled the gate still passes, so these labels cannot tell people apart; nothing written`,
    );
    return 1;
  }
  let text: string;
  try {
    text = numbersJson(run.result);
  } catch (err) {
    console.error(`${who}: refusing to write: ${(err as Error).message}`);
    return 65;
  }
  if (out) {
    writeFileSync(out, text);
    console.error(`${who}: wrote ${out}`);
  } else process.stdout.write(text);
  return 0;
}

/** Speaker models the evaluation can run, by id: the folder and file under the models folder. */
export const VOICE_MODELS: Record<string, { file: string }> = {
  "titanet-small": { file: "nemo_en_titanet_small.onnx" },
  // https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx
  "campplus-zh-en-advanced": {
    file: "3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx",
  },
};

/** The ONNX file of `model`: an id of `VOICE_MODELS` under `dir`, or a path ending in `.onnx`. */
export function voiceModelPath(model = "titanet-small", dir = defaultModelsDir()): string {
  if (model.endsWith(".onnx")) return model;
  const m = VOICE_MODELS[model];
  if (!m) throw new Error(`no speaker model is called ${model}; pass an id or an .onnx file`);
  return modelFile(dir, model, m.file);
}

/** A speaker embedder over one ONNX file, through sherpa-onnx (16 kHz mono samples). */
export function openEmbedder(path: string, threads = 2): (samples: Float32Array) => Float32Array {
  const sherpa = createRequire(import.meta.url)("sherpa-onnx-node");
  const ex = new sherpa.SpeakerEmbeddingExtractor({ model: path, numThreads: threads, debug: 0 });
  return (samples) => {
    const s = ex.createStream();
    s.acceptWaveform({ samples, sampleRate: 16000 });
    s.inputFinished();
    if (!ex.isReady(s)) return new Float32Array(ex.dim);
    return Float32Array.from(ex.compute(s, false) as Float32Array);
  };
}

/** The longest stretch embedded at once, seconds: the live worker's `asr.segmentWindow`. */
export const MAX_CHUNK_SECONDS = 12;
/** The shortest line the live worker embeds, seconds (`MIN_EMBED_SECONDS`). */
export const MIN_CHUNK_SECONDS = 1;

// ---------------------------------------------------------------------------

interface RawLabel {
  call: string;
  person: string;
  vec: number[];
  secs: number;
  cons: number;
  at?: number;
  norm?: number;
  snaps?: { min: number; vec: number[]; cons: number }[];
}

function fromRaw(l: RawLabel): VoiceLabel {
  return {
    ...l,
    vec: Float32Array.from(l.vec),
    snaps: l.snaps?.map((s) => ({ ...s, vec: Float32Array.from(s.vec) })),
  };
}

async function main(argv: string[]): Promise<number> {
  const flag = (n: string) => {
    const i = argv.indexOf(n);
    return i === -1 ? undefined : argv[i + 1];
  };
  const file = argv[0];
  if (!file || file.startsWith("--")) {
    console.error("usage: bun scripts/eval/voices.ts <labels.json> [--out <file>] [--max-wrong 0]");
    return 64;
  }
  const raw = JSON.parse(readFileSync(file, "utf8")) as {
    learn: RawLabel[];
    match?: RawLabel[];
    mixed?: MixedLabel[];
  };
  const learn = raw.learn.map(fromRaw);
  const run = evaluateVoices({
    learn,
    match: raw.match?.map(fromRaw),
    mixed: raw.mixed?.map((m) => ({
      cons: m.cons,
      snaps: m.snaps?.map((s) => ({ ...s, vec: new Float32Array(0) })),
    })),
    maxWrong: Number(flag("--max-wrong") ?? 0),
  });
  return finish(run, flag("--out"), "voices");
}

if (import.meta.main) process.exit(await main(process.argv.slice(2)));
