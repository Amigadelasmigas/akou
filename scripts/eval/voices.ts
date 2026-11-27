#!/usr/bin/env bun

/**
 * Whether a voice is recognisable across calls, measured before akou names anyone by voice
 * (docs/TESTING.md TS-21b). The input is a list of labels: one live speaker of one call, the
 * person it is, and its voiceprint (`src/main/asr/voiceprint.ts`) at each centroid write. The
 * output is numbers only.
 *
 *   bun scripts/eval/voices.ts <labels.json> [--out <file>] [--max-wrong 0]
 *
 * `scripts/eval-local.ts --voices` builds the labels from your own calls and
 * `scripts/eval/voices-ami.ts` from the AMI Meeting Corpus; both call `evaluateVoices` here. The
 * labels file holds `{learn, match?, open?}`, each label `{call, person, vec: number[], secs,
 * cons, at?, snaps?}`, each snap `{t, secs, vec: number[], cons, min?}`; `match` defaults to
 * `learn`.
 *
 * - **The learn rule.** A label becomes an exemplar of its person when `secs >= MIN_LEARN_SEC`.
 *   Each person keeps the `KEEP` newest (by `at`). `cons` is recorded and never gates anything.
 * - **The writes.** A label's voiceprint is matched at each centroid write the live worker makes:
 *   when its speech first reaches MIN_MATCH_SEC (and `FIRST_MATCH_SEC`), at each 5-minute tick of
 *   the call clock it spoke in since the last one, and at the end. Labels without snaps have one
 *   write, their full voiceprint.
 * - **The match rule** is akou's own, `src/main/voices/match.ts`: every write of a call, in time
 *   order, is one decision for its label, with the people other labels show left out and a name
 *   withdrawn when the voice moves away from it. A label counts as wrongly named when any write
 *   named it wrongly, and as named right when it ends named right and was never named wrongly.
 * - **Leave one call out.** A call is scored against a store learnt from every other call.
 * - **Unknown voices.** A second pass drops each person of the call from the store in turn; any
 *   name given to their label is wrong.
 * - **The grid** sweeps ACCEPT, MARGIN, MIN_MATCH_SEC and MIN_LEARN_SEC with offers off, once
 *   with a person scored by their best exemplar and once by the mean of their exemplars
 *   (`pick`/`pick_mean`). The OFFER sweep runs at each picked point.
 * - **The pick.** The first point by MIN_MATCH_SEC, MIN_LEARN_SEC, ACCEPT and MARGIN, each
 *   ascending, whose wrong-name rate is at most `--max-wrong` and which names at least half the
 *   eligible known voices. With no such point ACCEPT is null: akou only offers. OFFER is the
 *   lowest value with at most 10 % wrong offers.
 * - **The open set** (`open`): labels nobody named (no truth exists for them) are matched against
 *   the store learnt from every named label, at the picked point, with the people named in their
 *   call left out. Only counts and a histogram of their top scores are written.
 * - **The positive control.** The same run with the person labels shuffled must fail the gate at
 *   the picked point. A shuffle that only renames people across calls is refused and another
 *   drawn. When it passes, the data cannot tell people apart, and the script exits 1 and writes
 *   nothing.
 */

import { readFileSync, writeFileSync } from "node:fs";
import { createRequire } from "node:module";
import { defaultModelsDir, modelFile } from "../../src/main/asr/models.ts";
import { CENTROID_EVERY_MS, FIRST_MATCH_SEC } from "../../src/main/asr/speakers.ts";
import { Accum, score } from "../../src/main/asr/voiceprint.ts";
import { type Candidate, decideRanked, rankPeople } from "../../src/main/voices/match.ts";
import type { Person } from "../../src/main/voices/store.ts";
import type { ScoreMode, VoiceThresholds } from "../../src/main/voices/thresholds.ts";
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
  offer: range(0.3, 0.9, 0.02),
};

/** Exemplars kept per person. */
export const KEEP = 8;
/** The share of eligible known voices that must be named, or ACCEPT is null. */
export const MIN_NAMED = 0.5;
/** The most wrong offers among offers. */
export const MAX_WRONG_OFFERS = 0.1;
/** The live worker's periodic centroid write, seconds of call clock. */
export const WRITE_EVERY_SEC = CENTROID_EVERY_MS / 1000;
/** The speech at which a label gets a write of its own: every grid MIN_MATCH_SEC and the worker's. */
const FORCED = [...new Set([...GRID.minMatchSec, FIRST_MATCH_SEC])].sort((a, b) => a - b);
/** Bins of the open set's top-score histogram, over 0 to 1. */
export const HISTOGRAM_BINS = 20;

/** A label's voiceprint at one centroid write. */
export interface Snap {
  /** Seconds into the call. */
  t: number;
  /** Seconds of speech so far. */
  secs: number;
  vec: Float32Array;
  cons: number;
  /** Set on a write forced when the speech first reached this many seconds. */
  min?: number;
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
  /** Its writes in time order. Absent: one write, the full voiceprint. */
  snaps?: Snap[];
}

/**
 * Builds one label from its lines in time order, recording a write where the live worker makes
 * one: when the speech first reaches each forced value, at each periodic tick it spoke before, and
 * at the end.
 */
export class LabelBuilder {
  private readonly acc = new Accum();
  private readonly snaps: Snap[] = [];
  private readonly forced = new Set<number>();
  private last = 0;
  private dirty = false;

  /** One line's embedding, its seconds of speech, and its end in seconds into the call. */
  add(vec: Float32Array, secs: number, t?: number): void {
    const time = t ?? this.acc.secs + secs;
    if (this.dirty && Math.floor(time / WRITE_EVERY_SEC) > Math.floor(this.last / WRITE_EVERY_SEC))
      this.snap((Math.floor(this.last / WRITE_EVERY_SEC) + 1) * WRITE_EVERY_SEC);
    if (!this.acc.add(vec, secs)) return;
    this.last = time;
    this.dirty = true;
    for (const min of FORCED) {
      if (this.acc.secs < min || this.forced.has(min)) continue;
      this.forced.add(min);
      this.snap(time, min);
    }
  }

  private snap(t: number, min?: number): void {
    const r = this.acc.result();
    if (!r) return;
    this.snaps.push({ t, secs: r.secs, vec: r.vec, cons: r.cons, ...(min ? { min } : {}) });
    if (min === undefined) this.dirty = false;
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
      snaps: [
        ...this.snaps.map((s) => ({ ...s })),
        { t: this.last, secs: r.secs, vec: r.vec, cons: r.cons },
      ],
    };
  }
}

// ---------------------------------------------------------------------------
// The store and the writes

interface Lab {
  call: string;
  person: number;
  vec: Float32Array;
  secs: number;
  cons: number;
  at: number;
  snaps?: Snap[];
}

/** The model id the evaluation's store is tagged with: one embedder per run. */
const EVAL_MODEL = "eval";

/** Each person's exemplars under the learn rule: the `KEEP` newest eligible labels. */
function learn(labels: readonly Lab[], minLearn: number): Person[] {
  const by = new Map<number, Lab[]>();
  for (const l of labels) {
    if (l.secs < minLearn) continue;
    const list = by.get(l.person) ?? [];
    list.push(l);
    by.set(l.person, list);
  }
  return [...by].map(([p, list]) => ({
    name: String(p),
    exemplars: list
      .sort((a, b) => b.at - a.at)
      .slice(0, KEEP)
      .map((l) => ({
        model: EVAL_MODEL,
        dim: l.vec.length,
        vec: l.vec,
        secs: l.secs,
        cons: l.cons,
        call: l.call,
        spk: "c1",
        at: "",
      })),
  }));
}

/** A write ranked against one store: its time, its speech and every person scored. */
export interface Ranked {
  t: number;
  secs: number;
  min?: number;
  ranked: Candidate[];
}

/** The writes of a label at `minMatch`: its forced one at that speech, the worker's, the rest. */
function writesAt(snaps: readonly Ranked[], minMatch: number): Ranked[] {
  return snaps.filter(
    (s) =>
      s.secs >= minMatch &&
      (s.min === undefined || s.min === minMatch || s.min === FIRST_MATCH_SEC),
  );
}

function rankLabel(l: Lab, people: readonly Person[], mode: ScoreMode): Ranked[] {
  const snaps = l.snaps ?? [{ t: 0, secs: l.secs, vec: l.vec, cons: l.cons }];
  return snaps.map((s) => ({
    t: s.t,
    secs: s.secs,
    min: s.min,
    ranked: rankPeople(people, EVAL_MODEL, s.vec.length, s.vec, mode),
  }));
}

/** What became of one label over a call. */
interface Outcome {
  /** Every person it was named as, and offered as. */
  named: Set<string>;
  offered: Set<string>;
  /** The name it showed at the end. */
  final: { name: string; status: "named" | "offered" } | null;
  withdrawn: number;
  /** It had a write at or over MIN_MATCH_SEC. */
  eligible: boolean;
  /** Its top score at its first write. */
  first: number | null;
}

/**
 * One call's writes, in time order, each decided by akou's matcher for its label. `fixed` are
 * people shown on labels outside `labels` (the named labels, for the open set).
 */
export function simulate(
  labels: readonly (readonly Ranked[])[],
  t: VoiceThresholds,
  fixed: ReadonlySet<string> = new Set(),
): Outcome[] {
  const m = t.minMatchSec ?? Number.POSITIVE_INFINITY;
  const writes: { i: number; w: Ranked }[] = [];
  const out: Outcome[] = labels.map((snaps, i) => {
    const ws = writesAt(snaps, m);
    for (const w of ws) writes.push({ i, w });
    return {
      named: new Set<string>(),
      offered: new Set<string>(),
      final: null,
      withdrawn: 0,
      eligible: ws.length > 0,
      first: ws[0]?.ranked[0]?.score ?? null,
    };
  });
  writes.sort((a, b) => a.w.t - b.w.t || a.i - b.i);
  const refused = labels.map(() => new Set<string>());
  for (const { i, w } of writes) {
    const o = out[i] as Outcome;
    const taken = new Set(fixed);
    out.forEach((x, j) => {
      if (j !== i && x.final) taken.add(x.final.name);
    });
    let d = decideRanked(
      w.ranked,
      {
        spk: "c1",
        secs: w.secs,
        typed: false,
        voice: o.final,
        taken,
        refused: refused[i] as Set<string>,
      },
      t,
    );
    if (d.kind === "withdraw") {
      (refused[i] as Set<string>).add(d.name);
      o.final = null;
      o.withdrawn++;
      d = d.next;
    }
    if (d.kind === "name") {
      o.final = { name: d.name, status: "named" };
      o.named.add(d.name);
    } else if (d.kind === "offer") {
      o.final = { name: d.name, status: "offered" };
      o.offered.add(d.name);
    }
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

interface Prepared {
  learn: Lab[];
  match: Lab[];
  open: Lab[];
  calls: string[];
  cases: Case[];
  /** The ranked writes of each case's labels, by MIN_LEARN_SEC and score mode. */
  cache: Map<string, { labels: Lab[]; known: boolean[]; ranked: Ranked[][] }[]>;
}

function prepare(
  learnIn: readonly VoiceLabel[],
  matchIn: readonly VoiceLabel[],
  openIn: readonly VoiceLabel[] = [],
): Prepared {
  const ids = new Map<string, number>();
  const id = (p: string) => {
    let n = ids.get(p);
    if (n === undefined) {
      n = ids.size;
      ids.set(p, n);
    }
    return n;
  };
  const seen = new Map<VoiceLabel, Lab>();
  const conv = (l: VoiceLabel, i: number): Lab => {
    const had = seen.get(l);
    if (had) return had;
    const lab = {
      call: l.call,
      person: id(l.person),
      vec: l.vec,
      secs: l.secs,
      cons: l.cons,
      at: l.at ?? i,
      snaps: l.snaps,
    };
    seen.set(l, lab);
    return lab;
  };
  const learn = learnIn.map(conv);
  const match = matchIn.map(conv);
  const open = openIn.map((l, i) => ({ ...conv(l, i), person: -1 }));
  const calls = [...new Set(match.map((l) => l.call))].sort();
  const cases: Case[] = [];
  for (const call of calls) {
    cases.push({ call, drop: -1 });
    for (const p of new Set(match.filter((l) => l.call === call).map((l) => l.person)))
      cases.push({ call, drop: p });
  }
  return { learn, match, open, calls, cases, cache: new Map() };
}

/** Each case's labels ranked against its own store (`rankCases` is cached per setting). */
function rankCases(d: Prepared, minLearn: number, mode: ScoreMode) {
  const key = `${minLearn}/${mode}`;
  const had = d.cache.get(key);
  if (had) return had;
  const out = d.cases.map((c) => {
    const store = learn(
      d.learn.filter((l) => l.call !== c.call && l.person !== c.drop),
      minLearn,
    );
    const names = new Set(store.map((p) => p.name));
    const labels = d.match.filter((l) => l.call === c.call);
    return {
      labels,
      known: labels.map((l) => names.has(String(l.person))),
      ranked: labels.map((l) => rankLabel(l, store, mode)),
    };
  });
  d.cache.set(key, out);
  return out;
}

interface Tally {
  /** Known pass: eligible labels whose person is in the store, and of them, named right. */
  eligibleKnown: number;
  correct: number;
  /** Known pass: every eligible label, and the wrongly named among them. */
  eligible: number;
  wrongKnown: number;
  /** Wrong names between two people both in the store. */
  wrongEnrolled: number;
  /** Unknown pass: eligible labels of the dropped person, and the named among them. */
  eligibleUnknown: number;
  wrongUnknown: number;
  offers: number;
  wrongOffers: number;
  /** Labels that had a name withdrawn. */
  withdrawals: number;
}

interface Setting {
  accept: number | null;
  offer: number | null;
  margin: number;
  minMatch: number;
  minLearn: number;
  mode: ScoreMode;
}

const thresholdsOf = (s: Setting): VoiceThresholds => ({
  accept: s.accept,
  offer: s.offer,
  margin: s.margin,
  minMatchSec: s.minMatch,
  minLearnSec: s.minLearn,
  score: s.mode,
});

function tally(d: Prepared, s: Setting): Tally {
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
    withdrawals: 0,
  };
  const t = thresholdsOf(s);
  const cases = rankCases(d, s.minLearn, s.mode);
  d.cases.forEach((c, k) => {
    const cc = cases[k] as (typeof cases)[number];
    const res = simulate(cc.ranked, t);
    res.forEach((o, i) => {
      const lab = cc.labels[i] as Lab;
      if (!o.eligible || (c.drop >= 0 && lab.person !== c.drop)) return;
      const me = String(lab.person);
      const wrongName = [...o.named].some((p) => p !== me);
      if (o.withdrawn > 0) n.withdrawals++;
      if (c.drop < 0) {
        n.eligible++;
        if (cc.known[i]) n.eligibleKnown++;
        if (!wrongName && o.final?.status === "named" && o.final.name === me) n.correct++;
        if (wrongName) {
          n.wrongKnown++;
          if (cc.known[i]) n.wrongEnrolled++;
        }
      } else {
        n.eligibleUnknown++;
        if (o.named.size > 0) n.wrongUnknown++;
      }
      if (o.offered.size > 0) {
        n.offers++;
        if (c.drop >= 0 || [...o.offered].some((p) => p !== me)) n.wrongOffers++;
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
  /** 1 when a person scored the mean of their exemplars, 0 for the best one. */
  mean: number;
  eligible_share: number | null;
  correct_name_rate: number | null;
  wrong_name_rate: number | null;
  wrong_names: number;
  wrong_names_enrolled: number;
  wrong_names_unknown: number;
  withdrawals: number;
}

export interface Pick {
  accept: number | null;
  offer: number | null;
  margin: number | null;
  min_match_sec: number | null;
  min_learn_sec: number | null;
  correct_name_rate: number | null;
  wrong_name_rate: number | null;
  offer_rate: number | null;
  wrong_offer_rate: number | null;
}

export interface OfferRow {
  offer: number;
  offers: number;
  wrong_offers: number;
  offer_rate: number | null;
  wrong_offer_rate: number | null;
}

export interface OpenResult {
  clusters: number;
  eligible: number;
  named: number;
  offered: number;
  withdrawn: number;
  /** Eligible clusters by top score at their first write, `HISTOGRAM_BINS` bins over 0 to 1. */
  histogram: number[];
}

export interface VoicesResult {
  labels: number;
  people: number;
  calls: number;
  mean_raw_norm: number | null;
  pairs: {
    same: number;
    different: number;
    same_quantiles: Record<string, number> | null;
    different_quantiles: Record<string, number> | null;
    eer: number | null;
  };
  pick: Pick;
  pick_mean: Pick;
  offers: OfferRow[];
  offers_mean: OfferRow[];
  /** Null when no open labels were given. */
  open: OpenResult | null;
  open_mean: OpenResult | null;
  control: { shuffled_passes: number; shuffled_passes_mean: number };
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

/** Scores of every pair of labels from two calls, each pair once. */
function pairs(d: Prepared) {
  const same: number[] = [];
  const diff: number[] = [];
  const inMatch = new Map(d.match.map((l, i) => [l, i]));
  d.match.forEach((a, i) => {
    for (const b of d.learn) {
      const j = inMatch.get(b);
      if ((j !== undefined && j <= i) || a.call === b.call) continue;
      const s = score(a.vec, b.vec);
      if (s === null) continue;
      (a.person === b.person ? same : diff).push(s);
    }
  });
  return { same, diff };
}

interface Point {
  accept: number;
  margin: number;
  minMatch: number;
  minLearn: number;
  n: Tally;
}

function pickPoint(d: Prepared, mode: ScoreMode, maxWrong: number): Point | null {
  for (const minMatch of GRID.minMatchSec)
    for (const minLearn of GRID.minLearnSec)
      for (const accept of GRID.accept)
        for (const margin of GRID.margin) {
          const n = tally(d, { accept, offer: null, margin, minMatch, minLearn, mode });
          if (meets(n, maxWrong)) return { accept, margin, minMatch, minLearn, n };
        }
  return null;
}

/** The pick for one score mode, with its OFFER sweep. */
function pickFor(d: Prepared, mode: ScoreMode, maxWrong: number) {
  const point = pickPoint(d, mode, maxWrong);
  const offers: OfferRow[] = [];
  let offer: number | null = null;
  let chosen: Tally | null = point?.n ?? null;
  const minMatch = point?.minMatch ?? (GRID.minMatchSec[0] as number);
  const minLearn = point?.minLearn ?? (GRID.minLearnSec[0] as number);
  for (const o of GRID.offer) {
    if (point && o >= point.accept) break;
    const n = tally(d, {
      accept: point?.accept ?? null,
      offer: o,
      margin: point?.margin ?? 0,
      minMatch,
      minLearn,
      mode,
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
  const row = offers.find((r) => r.offer === offer);
  const pick: Pick = {
    accept: point?.accept ?? null,
    offer,
    margin: point?.margin ?? null,
    min_match_sec: point?.minMatch ?? null,
    min_learn_sec: point?.minLearn ?? null,
    correct_name_rate: chosen ? rate(chosen.correct, chosen.eligibleKnown) : null,
    wrong_name_rate: chosen ? wrongRate(chosen) : null,
    offer_rate: row?.offer_rate ?? null,
    wrong_offer_rate: row?.wrong_offer_rate ?? null,
  };
  const setting: Setting = {
    accept: point?.accept ?? null,
    offer,
    margin: point?.margin ?? 0,
    minMatch,
    minLearn,
    mode,
  };
  return { point, pick, offers, setting };
}

/**
 * The open set at `s`: every open label matched against the store of every named label, call by
 * call in time order, with the people named in its call left out.
 */
function openSet(d: Prepared, s: Setting): OpenResult {
  const store = learn(d.learn, s.minLearn);
  const t = thresholdsOf(s);
  const out: OpenResult = {
    clusters: d.open.length,
    eligible: 0,
    named: 0,
    offered: 0,
    withdrawn: 0,
    histogram: Array.from({ length: HISTOGRAM_BINS }, () => 0),
  };
  for (const call of [...new Set(d.open.map((l) => l.call))].sort()) {
    const labels = d.open.filter((l) => l.call === call);
    const fixed = new Set(d.match.filter((l) => l.call === call).map((l) => String(l.person)));
    const res = simulate(
      labels.map((l) => rankLabel(l, store, s.mode)),
      t,
      fixed,
    );
    for (const o of res) {
      if (!o.eligible) continue;
      out.eligible++;
      if (o.named.size > 0) out.named++;
      if (o.offered.size > 0) out.offered++;
      if (o.withdrawn > 0) out.withdrawn++;
      if (o.first !== null) {
        const bin = Math.min(HISTOGRAM_BINS - 1, Math.max(0, Math.floor(o.first * HISTOGRAM_BINS)));
        out.histogram[bin] = (out.histogram[bin] as number) + 1;
      }
    }
  }
  return out;
}

/** Each key's person as the index of that person's first key: equal when two labelings group alike. */
function grouping(people: readonly string[]): number[] {
  const first = new Map<string, number>();
  return people.map((p, i) => {
    const n = first.get(p);
    if (n !== undefined) return n;
    first.set(p, i);
    return i;
  });
}

/** Seeds tried for a shuffle that groups the labels differently. */
const SHUFFLE_TRIES = 1000;

/**
 * People shuffled among the labels: one person per (call, person) key, in both sets. A shuffle
 * that only renames people (every key still grouped with the same keys) keeps who is who, so it is
 * refused and the next seed drawn; when no shuffle can group them differently (one person, or one
 * key per person), the last one tried stands.
 */
export function shuffled(
  learn: readonly VoiceLabel[],
  match: readonly VoiceLabel[],
  seed = 1,
): { learn: VoiceLabel[]; match: VoiceLabel[] } {
  const key = (l: VoiceLabel) => `${l.call}\u0000${l.person}`;
  const keys = [...new Set([...learn, ...match].map(key))].sort();
  const original = keys.map((k) => k.split("\u0000")[1] as string);
  const was = grouping(original).join(",");
  let people = original;
  for (let tri = 0; tri < SHUFFLE_TRIES; tri++) {
    people = [...original];
    let s = (seed + tri) >>> 0;
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
    if (grouping(people).join(",") !== was) break;
  }
  const to = new Map(keys.map((k, i) => [k, people[i] as string]));
  const swapped = new Map<VoiceLabel, VoiceLabel>();
  const swap = (l: VoiceLabel) => {
    const had = swapped.get(l);
    if (had) return had;
    const out = { ...l, person: to.get(key(l)) as string };
    swapped.set(l, out);
    return out;
  };
  return { learn: learn.map(swap), match: match.map(swap) };
}

export interface VoicesInput {
  learn: readonly VoiceLabel[];
  /** The labels matched; defaults to `learn` (another channel for the far-field run). */
  match?: readonly VoiceLabel[];
  /** Labels nobody named, matched for counts only; their `person` is not read. */
  open?: readonly VoiceLabel[];
  /** The highest wrong-name rate the gate allows: 0 on your own calls, 0.005 on AMI. */
  maxWrong?: number;
  /** Skip the full grid in the result (the picks and the control still run). */
  noGrid?: boolean;
}

const MODES: readonly ScoreMode[] = ["best", "mean"];

/** The whole measurement, and whether its positive control failed as it must. */
export function evaluateVoices(input: VoicesInput): { result: VoicesResult; controlOk: boolean } {
  const learnIn = input.learn;
  const matchIn = input.match ?? input.learn;
  const maxWrong = input.maxWrong ?? 0;
  const d = prepare(learnIn, matchIn, input.open);
  const picks = MODES.map((mode) => pickFor(d, mode, maxWrong));
  const p = pairs(d);
  const norms = [...new Set([...learnIn, ...matchIn])]
    .map((l) => l.norm)
    .filter((x): x is number => typeof x === "number");

  const grid: GridRow[] = [];
  if (!input.noGrid)
    for (const mode of MODES)
      for (const minLearn of GRID.minLearnSec)
        for (const minMatch of GRID.minMatchSec)
          for (const margin of GRID.margin)
            for (const accept of GRID.accept) {
              const n = tally(d, { accept, offer: null, margin, minMatch, minLearn, mode });
              grid.push({
                accept,
                margin,
                min_match_sec: minMatch,
                min_learn_sec: minLearn,
                mean: mode === "mean" ? 1 : 0,
                eligible_share: rate(n.eligible, d.match.length),
                correct_name_rate: rate(n.correct, n.eligibleKnown),
                wrong_name_rate: rate(
                  n.wrongKnown + n.wrongUnknown,
                  n.eligible + n.eligibleUnknown,
                ),
                wrong_names: n.wrongKnown + n.wrongUnknown,
                wrong_names_enrolled: n.wrongEnrolled,
                wrong_names_unknown: n.wrongUnknown,
                withdrawals: n.withdrawals,
              });
            }

  // The positive control: shuffled people at each picked point, or the whole pick when none.
  const sh = shuffled(learnIn, matchIn);
  const ds = prepare(sh.learn, input.match ? sh.match : sh.learn);
  const passes = picks.map(({ point }, k) => {
    const mode = MODES[k] as ScoreMode;
    if (!point) return pickPoint(ds, mode, maxWrong) !== null;
    return meets(tally(ds, { ...point, offer: null, mode }), maxWrong);
  });

  const [best, mean] = picks as [(typeof picks)[number], (typeof picks)[number]];
  const hasOpen = (input.open?.length ?? 0) > 0;
  const result: VoicesResult = {
    labels: d.match.length,
    people: new Set(d.match.map((l) => l.person)).size,
    calls: d.calls.length,
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
    pick: best.pick,
    pick_mean: mean.pick,
    offers: best.offers,
    offers_mean: mean.offers,
    open: hasOpen ? openSet(d, best.setting) : null,
    open_mean: hasOpen ? openSet(d, mean.setting) : null,
    control: { shuffled_passes: passes[0] ? 1 : 0, shuffled_passes_mean: passes[1] ? 1 : 0 },
    grid,
  };
  return { result, controlOk: !passes.some(Boolean) };
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
  snaps?: { t: number; secs: number; vec: number[]; cons: number; min?: number }[];
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
    open?: RawLabel[];
  };
  const learn = raw.learn.map(fromRaw);
  const run = evaluateVoices({
    learn,
    match: raw.match?.map(fromRaw),
    open: raw.open?.map(fromRaw),
    maxWrong: Number(flag("--max-wrong") ?? 0),
  });
  return finish(run, flag("--out"), "voices");
}

if (import.meta.main) process.exit(await main(process.argv.slice(2)));
