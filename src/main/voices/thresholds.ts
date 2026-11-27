/**
 * The numbers that decide when akou names a voice, per speaker embedder. They come from
 * `docs/gates/voices.json`, which the voices evaluation fills (docs/TESTING.md TS-21b) and which
 * is read into the build here: no value is written in the code.
 *
 * ```json
 * { "titanet-small": { "accept": 0.7, "offer": 0.6, "margin": 0.1, "minMatchSec": 20,
 *                      "minLearnSec": 30, "score": "best" } }
 * ```
 *
 * - `accept` and `margin`: a voice is named when its best person scores `accept` or more and leads
 *   the next person by `margin` or more. Either null: akou names no one by voice.
 * - `offer`: a voice is offered as a guess from this score. Null: akou offers no one.
 * - `minMatchSec`: the speech a label needs before it is matched. Null: akou does nothing by voice.
 * - `minLearnSec`: the speech a named label needs before it teaches the store.
 * - `score`: how a person is scored, `best` (the best cosine over their exemplars) or `mean` (the
 *   cosine to the mean of their exemplars). Anything else reads as `best`.
 *
 * An embedder with no entry gets every value null, so a model nobody measured is never matched.
 */

import gate from "../../../docs/gates/voices.json" with { type: "json" };

export type ScoreMode = "best" | "mean";

export interface VoiceThresholds {
  accept: number | null;
  offer: number | null;
  margin: number | null;
  minMatchSec: number | null;
  minLearnSec: number | null;
  score: ScoreMode;
}

/** Every value null: akou does nothing by voice. */
export const NO_THRESHOLDS: Readonly<VoiceThresholds> = Object.freeze({
  accept: null,
  offer: null,
  margin: null,
  minMatchSec: null,
  minLearnSec: null,
  score: "best",
});

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

/** The thresholds for the embedder `model`, from `table` (the gate file unless a test passes one). */
export function thresholdsFor(model: string, table: unknown = gate): VoiceThresholds {
  if (!table || typeof table !== "object" || !Object.hasOwn(table, model))
    return { ...NO_THRESHOLDS };
  const e = (table as Record<string, unknown>)[model];
  if (!e || typeof e !== "object") return { ...NO_THRESHOLDS };
  const r = e as Record<string, unknown>;
  return {
    accept: num(r.accept),
    offer: num(r.offer),
    margin: num(r.margin),
    minMatchSec: num(r.minMatchSec),
    minLearnSec: num(r.minLearnSec),
    score: r.score === "mean" ? "mean" : "best",
  };
}
