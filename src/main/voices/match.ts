/**
 * Who a call speaker is by voice: one decision for one label at one `speaker.centroid` write. A
 * pure function with no IO; the voices evaluation (`scripts/eval/voices.ts`) drives the same code
 * in the same order, so its numbers describe what akou does.
 *
 * - **Order.** Each centroid write of a label is one decision for that label, made when it lands.
 *   People already named or offered on other labels of the call are left out, and so are people
 *   this label refused or had withdrawn. Labels are never ranked against each other by score.
 * - **Scores.** A person scores the best cosine over their exemplars (`score: "best"`), or the
 *   cosine to the mean of their exemplars (`"mean"`). Only exemplars of the running embedder's
 *   model and size count [VO-3]; a person the store no longer holds has no score [VO-2].
 * - **The runner-up.** With p1 the top person, a person whose best exemplar is itself at least
 *   ACCEPT-similar to one of p1's exemplars is the same voice stored under a second name, and is
 *   passed over; the next person down is the runner-up [VO-11].
 * - **Named:** p1 >= ACCEPT and p1 - runner-up >= MARGIN. **Offered:** p1 >= OFFER.
 * - **Withdrawn.** A label showing a name from a match loses it when that person falls below OFFER
 *   (ACCEPT when no offers), or when another person is now named, one that is not the shown
 *   name's own voice under a second name. The label then decides again without the withdrawn
 *   person [VO-10] [VO-11]. An offer can still be raised to a name.
 * - **Skipped:** a label with a typed name, a label that is not `c<N>`, a label under
 *   MIN_MATCH_SEC, and every label while the thresholds leave nothing to decide.
 *
 * Scores are returned unrounded; the caller rounds what it writes.
 */

import { score as cosine, unit } from "../asr/voiceprint.ts";
import type { Person } from "./store.ts";
import type { ScoreMode, VoiceThresholds } from "./thresholds.ts";

/** One person's standing against a label's voiceprint. */
export interface Candidate {
  name: string;
  score: number;
  /** The exemplar closest to the label. */
  best: Float32Array;
  /** Every exemplar of the running embedder, unit length. */
  exemplars: readonly Float32Array[];
}

/**
 * Every person scored against `vec`, highest first (ties by name). Exemplars of another model or
 * size are skipped, and a person with none left is not listed. A `vec` of another size than `dim`
 * scores no one.
 */
export function rankPeople(
  people: readonly Person[],
  model: string,
  dim: number,
  vec: Float32Array,
  mode: ScoreMode = "best",
): Candidate[] {
  if (vec.length !== dim) return [];
  const v = unit(vec);
  const out: Candidate[] = [];
  for (const p of people) {
    const ex = p.exemplars
      .filter((e) => e.model === model && e.dim === dim && e.vec.length === dim)
      .map((e) => unit(e.vec));
    let best: Float32Array | null = null;
    let top: number | null = null;
    for (const e of ex) {
      const s = cosine(v, e);
      if (s !== null && (top === null || s > top)) {
        top = s;
        best = e;
      }
    }
    if (top === null || !best) continue;
    let s = top;
    if (mode === "mean") {
      const sum = new Float32Array(dim);
      for (const e of ex)
        for (let i = 0; i < dim; i++) sum[i] = (sum[i] as number) + (e[i] as number);
      const m = cosine(v, sum);
      if (m === null) continue;
      s = m;
    }
    out.push({ name: p.name, score: s, best, exemplars: ex });
  }
  return out.sort((a, b) => b.score - a.score || (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
}

/** One label of one call, as it stands at one centroid write. */
export interface LabelState {
  /** The live label, `c<N>`; any other is skipped. */
  spk: string;
  /** Seconds of speech in its centroid. */
  secs: number;
  /** It has a typed name (`speaker.name`): akou leaves it alone. */
  typed: boolean;
  /** The name it shows from a match, if any. */
  voice: { name: string; status: "named" | "offered" } | null;
  /** People named or offered on other labels of this call. */
  taken: ReadonlySet<string>;
  /** People refused on this label, by a person or withdrawn by akou. */
  refused: ReadonlySet<string>;
}

/** What a decision from scratch says. */
export type Fresh =
  | { kind: "none" }
  | { kind: "name"; name: string; score: number }
  | { kind: "offer"; name: string; score: number };

/** `withdraw`: write `refused` (by the app) for `name`, then act on `next`. */
export type Decision = Fresh | { kind: "withdraw"; name: string; next: Fresh };

const NONE: Fresh = { kind: "none" };

/** `q` is `top`'s voice under another name: its best exemplar is ACCEPT-similar to one of top's. */
function sameVoice(top: Candidate, q: Candidate, accept: number): boolean {
  return top.exemplars.some((e) => (cosine(q.best, e) ?? -1) >= accept);
}

/** The person after `top` who is not `top`'s voice under another name, or undefined. */
function runnerUp(open: readonly Candidate[], accept: number): Candidate | undefined {
  const top = open[0] as Candidate;
  for (let i = 1; i < open.length; i++) {
    const q = open[i] as Candidate;
    if (!sameVoice(top, q, accept)) return q;
  }
  return undefined;
}

function fresh(open: readonly Candidate[], t: VoiceThresholds): Fresh {
  const p1 = open[0];
  if (!p1) return NONE;
  if (t.accept !== null && t.margin !== null && p1.score >= t.accept) {
    const p2 = runnerUp(open, t.accept);
    if (p1.score - (p2?.score ?? -1) >= t.margin)
      return { kind: "name", name: p1.name, score: p1.score };
  }
  if (t.offer !== null && p1.score >= t.offer)
    return { kind: "offer", name: p1.name, score: p1.score };
  return NONE;
}

/** The decision for one label from `ranked` (`rankPeople` of its centroid). */
export function decideRanked(
  ranked: readonly Candidate[],
  label: LabelState,
  t: VoiceThresholds,
): Decision {
  if (label.typed || !/^c\d+$/.test(label.spk)) return NONE;
  if (t.minMatchSec === null || label.secs < t.minMatchSec) return NONE;
  const floor = t.offer ?? t.accept;
  if (floor === null) return NONE;
  const open = ranked.filter((c) => !label.taken.has(c.name) && !label.refused.has(c.name));
  const now = fresh(open, t);
  const cur = label.voice;
  if (!cur) return now;
  const mine = open.find((c) => c.name === cur.name);
  // Another name now on top moves the label away, unless it is the shown name's own voice [VO-11].
  const moved =
    now.kind === "name" &&
    now.name !== cur.name &&
    !!mine &&
    !sameVoice(open[0] as Candidate, mine, t.accept as number);
  if (!mine || mine.score < floor || moved)
    return {
      kind: "withdraw",
      name: cur.name,
      next: fresh(
        open.filter((c) => c.name !== cur.name),
        t,
      ),
    };
  if (cur.status === "offered" && now.kind === "name" && now.name === cur.name) return now;
  return NONE;
}

/** The decision for one label at one centroid write, against the store's people. */
export function matchVoice(input: {
  people: readonly Person[];
  /** The running embedder. */
  model: string;
  dim: number;
  /** The label's centroid; normalised here. */
  vec: Float32Array;
  label: LabelState;
  thresholds: VoiceThresholds;
}): Decision {
  const t = input.thresholds;
  if (t.minMatchSec === null || (t.accept === null && t.offer === null)) return NONE;
  return decideRanked(
    rankPeople(input.people, input.model, input.dim, input.vec, t.score),
    input.label,
    t,
  );
}
