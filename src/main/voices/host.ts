/**
 * Voices on the Host: the app feeds it every event of every call (`app.watch`), and it teaches the
 * voices file and names the voices it knows. App mode only; server mode never makes one [VO-8].
 *
 * - **Learn** (`learn.ts`) on each `speaker.name` and `speaker.centroid`: a label a person named
 *   keeps its latest centroid in `voices.yaml` under that name [VO-5].
 * - **Match** (`match.ts`) on each `speaker.centroid` only: each centroid write of a label is one
 *   decision for that label, in the order the writes land, exactly as the voices evaluation
 *   (`scripts/eval/voices.ts`) drives the same function. People shown on other labels of the call
 *   are left out, and so are people refused on this label or withdrawn from it. A decision writes
 *   `speaker.voice` by `app`: `refused` for a withdrawn name, then `named` or `offered` with the
 *   score to 2 places. A name typed later wins in the fold [VO-1] and teaches.
 * - **Purge**: a call deleted for good loses its exemplars; trash and restore keep them [VO-4].
 *
 * Every step runs on one queue, so decisions keep the order of the events. A decision reads the
 * fold and writes in the same turn, with nothing awaited in between, so it never acts on a fold a
 * later event changed. Exemplars taught by the call being matched are left out, as the evaluation
 * leaves the call out of its store. With `voices.enabled` off nothing learns or matches, and the
 * file stays as it is. The log gets counts only, never a name or a vector.
 */

import { stat } from "node:fs/promises";
import type { EventDraft, LogEvent } from "../../core/log/events.ts";
import type { CallView } from "../../core/log/fold.ts";
import { decodeVec } from "../asr/speakers.ts";
import { type LabelCentroid, lessonFor } from "./learn.ts";
import { type LabelState, matchVoice } from "./match.ts";
import type { Person, VoiceStore } from "./store.ts";
import type { VoiceThresholds } from "./thresholds.ts";

/** The live speaker embedder: every `speaker.centroid` is TitaNet-small's. */
export const LIVE_EMBEDDER = "titanet-small";

/** One call as the Host reads and writes it. */
export interface VoiceCall {
  view: CallView;
  /** Appends one event at once; null when the log is closed or refuses it. */
  append(draft: EventDraft): LogEvent | null;
}

export interface VoiceHostOptions {
  store: VoiceStore;
  /** `voices.enabled`, read at each event. */
  enabled: () => boolean;
  thresholds: (model: string) => VoiceThresholds;
  /** The call, while it is loaded. */
  call: (id: string) => VoiceCall | undefined;
  log: (level: "info" | "warn", msg: string) => void;
  /** The embedder the centroids come from; `LIVE_EMBEDDER` unless a test passes another. */
  model?: string;
}

/** The matcher's view of one label from the fold, or null when it is merged into another. */
export function labelState(view: CallView, spk: string, secs: number): LabelState | null {
  if (view.resolveSpeaker(spk) !== spk) return null;
  const roster = view.roster();
  const me = roster.find((r) => r.spk === spk);
  const typed = !!me?.name && !me.byVoice;
  const v = typed ? undefined : view.voiceFor(spk);
  const taken = new Set<string>();
  for (const r of roster) {
    if ((r.mergedInto ?? r.spk) === spk) continue;
    if (r.name && !r.byVoice) taken.add(r.name);
    else if (r.voice) taken.add(r.voice.name);
  }
  return {
    spk,
    secs,
    typed,
    voice: v ? { name: v.name, status: v.status } : null,
    taken,
    refused: view.refusedVoices(spk),
  };
}

const round2 = (x: number): number => Math.round(x * 100) / 100;

export class VoiceHost {
  readonly model: string;
  private queue: Promise<void> = Promise.resolve();
  /** The people of the file as last read, by the file's identity and the embedding size. */
  private cache: { key: string; dim: number; people: Person[] } | null = null;
  private readFailed: string | null = null;

  constructor(private readonly o: VoiceHostOptions) {
    this.model = o.model ?? LIVE_EMBEDDER;
  }

  /** Every event of every call; only names and centroids do anything. */
  onEvent(call: string, e: LogEvent): void {
    if (e.type !== "speaker.centroid" && e.type !== "speaker.name") return;
    this.run(() => this.handle(call, e));
  }

  /** A call deleted for good: its exemplars leave the file, whatever `voices.enabled` says. */
  purged(call: string): void {
    this.run(async () => {
      const n = await this.o.store.forgetCall(call);
      if (n > 0) this.o.log("info", `voices: dropped ${n} voiceprint(s) of a deleted call`);
    });
  }

  /** Resolves when every step queued so far is done. */
  idle(): Promise<void> {
    return this.queue;
  }

  private run(step: () => Promise<void>): void {
    // The store's message may quote a name, and the log takes none.
    this.queue = this.queue.then(step).catch(() => {
      this.o.log(
        "warn",
        "voices: voices.yaml could not be read or written; fix or delete it by hand",
      );
    });
  }

  private async handle(call: string, e: LogEvent): Promise<void> {
    if (!this.o.enabled()) return;
    if (e.type !== "speaker.centroid" && e.type !== "speaker.name") return;
    const c0 = this.o.call(call);
    if (!c0) return;
    const t = this.o.thresholds(this.model);
    const spk = e.spk;
    let centroid: LabelCentroid | undefined;
    if (e.type === "speaker.centroid") {
      centroid = { vec: decodeVec(e.vec), secs: e.secs, cons: e.cons };
    } else {
      const s = c0.view.speakerState().centroids.find((x) => x.spk === spk);
      centroid = s && { vec: decodeVec(s.vec), secs: s.secs, cons: s.cons };
    }
    const lesson = lessonFor({
      view: c0.view,
      call,
      spk,
      model: this.model,
      centroid,
      minLearnSec: t.minLearnSec,
      at: new Date(e.t).toISOString(),
    });
    if (lesson) {
      try {
        await this.o.store.upsert(lesson.name, lesson.exemplar);
      } catch {
        this.o.log("warn", "voices: a voiceprint was not kept in voices.yaml");
      }
    }
    if (e.type !== "speaker.centroid" || !centroid) return;
    const secs = centroid.secs;
    if (!/^c\d+$/.test(spk) || secs === undefined || centroid.cons === undefined) return;
    if (t.minMatchSec === null || secs < t.minMatchSec) return;
    if (t.accept === null && t.offer === null) return;
    const all = await this.people(centroid.vec.length);
    if (!all) return;
    // Leave this call out, as the evaluation scores a call against what every other call taught.
    const people = all
      .map((p) => ({ name: p.name, exemplars: p.exemplars.filter((x) => x.call !== call) }))
      .filter((p) => p.exemplars.length > 0);
    // From here to the writes nothing is awaited: each decision reads the fold as it is now.
    const c = this.o.call(call);
    if (!c || !this.o.enabled()) return;
    // A withdraw writes `refused`, then the label decides again without that person.
    for (let i = 0; i < 4; i++) {
      const label = labelState(c.view, spk, secs);
      if (!label) return;
      const d = matchVoice({
        people,
        model: this.model,
        dim: centroid.vec.length,
        vec: centroid.vec,
        label,
        thresholds: t,
      });
      if (d.kind === "none") return;
      if (d.kind === "withdraw") {
        if (!c.append({ type: "speaker.voice", spk, name: d.name, status: "refused", by: "app" }))
          return;
        continue;
      }
      c.append({
        type: "speaker.voice",
        spk,
        name: d.name,
        status: d.kind === "name" ? "named" : "offered",
        score: round2(d.score),
        by: "app",
      });
      return;
    }
  }

  /** The people of the running embedder, read again only when the file changed. */
  private async people(dim: number): Promise<Person[] | null> {
    let key = "none";
    try {
      const s = await stat(this.o.store.path);
      key = `${s.ino}:${s.size}:${s.mtimeMs}`;
    } catch {}
    if (this.cache?.key === key && this.cache.dim === dim) return this.cache.people;
    try {
      const { people } = await this.o.store.load(this.model, dim);
      this.cache = { key, dim, people };
      this.readFailed = null;
      return people;
    } catch {
      // Logged once per version of the file; the store's message may quote a name.
      if (this.readFailed !== key)
        this.o.log("warn", "voices: voices.yaml cannot be read; fix or delete it by hand");
      this.readFailed = key;
      return null;
    }
  }
}
