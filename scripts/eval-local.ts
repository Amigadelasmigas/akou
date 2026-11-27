#!/usr/bin/env bun

/**
 * The evaluation on your own calls (docs/TESTING.md TS-21): the nightly's scoring (scripts/eval/
 * score.ts) run on the calls on this machine, with your own fixes as the hand reference. Only
 * numbers leave it; no word of a call is ever written.
 *
 *   bun scripts/eval-local.ts [--root ~/Recordings/akou] [--rttm <dir>] [--out docs/gates/eval-local-<date>.json]
 *   bun scripts/eval-local.ts --voices <calls dir> [--model titanet-small|<file.onnx>] [--models <dir>] [--out <file>]
 *
 * - **WER against hand corrections.** A fix is a correction a person or an agent made once the
 *   transcript had begun: a `vocab.add` not written by the app, after the call's first line, and a
 *   line's text rewritten by hand (a `seg` revision with `by`). The reference is each line as it
 *   reads with those fixes, the hypothesis the same line without them. `wer_fixed_lines` scores the
 *   lines a fix changed; `wer_lower_bound` spreads the same errors over every line of the calls that
 *   have a fix, the floor of the whole-call WER, since a line nobody fixed may still be wrong.
 * - **Recall.** Of the times a vocabulary term (every `vocab.add` term of the call) appears in the
 *   reference, the share the recognizer already wrote right.
 * - **DER**, with `--rttm`: `<dir>/<call folder>.rttm` holds the speakers you labelled by hand, in
 *   seconds of the call's audio. The hypothesis is akou's speaker per line. Calls with more than one
 *   part are skipped and counted, because an RTTM file has one timeline.
 *
 * - **Voices**, with `--voices` (TS-21b): whether a named voice is recognised in your other calls.
 *   A label is a live `c<N>` speaker that you named, or the CLI did (`user`, `agent:cli`). A name
 *   on two labels of one call, and a label whose name changed later, are dropped as likely merges
 *   or splits. A person is their name, trimmed and in lower case; a name of fewer words that
 *   starts the only longer name it starts (`ana` and `ana ruiz`) is that person too. Each label's
 *   call lines of 1 s or more are embedded again from the part audio (the call channel, cut at
 *   12 s as the live worker cuts them, each placed at its end on the call's clock) with the
 *   speaker model, TitaNet small unless `--model` names another, and scored by
 *   `scripts/eval/voices.ts`. Every live `c<N>` speaker nobody named is embedded the same way and
 *   matched as the open set: only how many were named or offered is written. Needs ffmpeg.
 *
 * The result is checked before it is written: every value must be a number (or null where nothing
 * was scored), so a string, and with it any text of a call, makes the script refuse and write
 * nothing. Keep the RTTM files outside the repository: they are yours, not the project's.
 */

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { basename, dirname, join } from "node:path";
import type { LogEvent } from "../src/core/log/events.ts";
import { fold } from "../src/core/log/fold.ts";
import { readLog } from "../src/core/log/reader.ts";
import { EVENTS_FILE } from "../src/core/log/writer.ts";
import { ffmpegCommand } from "../src/main/asr/decode.ts";
import { ASR_RATE } from "../src/main/asr/engine.ts";
import { partFile } from "../src/main/call/folder.ts";
import { listCallDirs } from "../src/main/call/recovery.ts";
import { der, normalizeText, parseRttm, type Turn, wordErrors } from "./eval/score.ts";
import {
  evaluateVoices,
  finish,
  LabelBuilder,
  MAX_CHUNK_SECONDS,
  MIN_CHUNK_SECONDS,
  openEmbedder,
  type VoiceLabel,
  voiceModelPath,
} from "./eval/voices.ts";

export interface LocalResult {
  calls: number;
  calls_with_fixes: number;
  lines: number;
  lines_fixed: number;
  ref_words_fixed_lines: number;
  ref_words_all_lines: number;
  word_errors: number;
  /** Percent; null when no line was fixed. */
  wer_fixed_lines: number | null;
  wer_lower_bound: number | null;
  term_occurrences: number;
  term_recall: number | null;
  der_calls: number;
  der_skipped_multipart: number;
  der: number | null;
}

/** The events with every hand fix taken out: what the recognizer and the app wrote. */
export function withoutFixes(events: readonly LogEvent[]): LogEvent[] {
  const firstSeg = events.find((e) => e.type === "seg")?.seq ?? Number.POSITIVE_INFINITY;
  const fixIds = new Set(
    events
      .filter((e) => e.type === "vocab.add" && e.rev === 1 && e.by !== "app" && e.seq > firstSeg)
      .map((e) => (e as Extract<LogEvent, { type: "vocab.add" }>).id),
  );
  return events.filter((e) => {
    if (e.type === "vocab.add") return !fixIds.has(e.id);
    if (e.type === "seg") return !(e.rev > 1 && e.by !== undefined);
    return true;
  });
}

/** How many times `term` occurs in `words`, as a run of normalized words. */
function count(words: readonly string[], term: readonly string[]): number {
  if (term.length === 0) return 0;
  let n = 0;
  for (let i = 0; i + term.length <= words.length; i++)
    if (term.every((w, k) => words[i + k] === w)) n++;
  return n;
}

export interface CallScore {
  lines: number;
  linesFixed: number;
  refWordsFixed: number;
  refWordsAll: number;
  errors: number;
  termOccurrences: number;
  termHits: number;
  parts: number;
  /** akou's speaker per line, part 1's audio seconds. */
  turns: Turn[];
}

/** One call's numbers. */
export function scoreCall(events: readonly LogEvent[]): CallScore {
  const ref = fold(events);
  const hyp = fold(withoutFixes(events));
  const terms = [
    ...new Set(
      events
        .filter((e): e is Extract<LogEvent, { type: "vocab.add" }> => e.type === "vocab.add")
        .map((e) => e.term)
        .filter((t): t is string => typeof t === "string"),
    ),
  ]
    .map(normalizeText)
    .filter((t) => t.length > 0);
  const s: CallScore = {
    lines: 0,
    linesFixed: 0,
    refWordsFixed: 0,
    refWordsAll: 0,
    errors: 0,
    termOccurrences: 0,
    termHits: 0,
    parts: ref.parts().length,
    turns: [],
  };
  for (const line of ref.lines("best")) {
    const r = normalizeText(line.text);
    const h = normalizeText(hyp.resolve(line.id)?.text ?? "");
    s.lines++;
    s.refWordsAll += r.length;
    s.turns.push({ start: line.a0, end: line.a1, speaker: line.spk });
    for (const t of terms) {
      const inRef = count(r, t);
      s.termOccurrences += inRef;
      s.termHits += Math.min(inRef, count(h, t));
    }
    if (r.join(" ") === h.join(" ")) continue;
    s.linesFixed++;
    s.refWordsFixed += r.length;
    s.errors += wordErrors(r, h);
  }
  return s;
}

const pct = (a: number, b: number) => (b > 0 ? Math.round((10000 * a) / b) / 100 : null);

/** Every call under `root`, scored; DER for the calls with an RTTM file in `rttmDir`. */
export async function evaluate(root: string, rttmDir?: string): Promise<LocalResult> {
  const out: LocalResult = {
    calls: 0,
    calls_with_fixes: 0,
    lines: 0,
    lines_fixed: 0,
    ref_words_fixed_lines: 0,
    ref_words_all_lines: 0,
    word_errors: 0,
    wer_fixed_lines: null,
    wer_lower_bound: null,
    term_occurrences: 0,
    term_recall: null,
    der_calls: 0,
    der_skipped_multipart: 0,
    der: null,
  };
  let termHits = 0;
  let derSpeech = 0;
  let derErrors = 0;
  for (const { dir } of listCallDirs(root)) {
    const { events } = await readLog(join(dir, EVENTS_FILE));
    const s = scoreCall(events);
    out.calls++;
    out.lines += s.lines;
    out.term_occurrences += s.termOccurrences;
    termHits += s.termHits;
    if (s.linesFixed > 0) {
      out.calls_with_fixes++;
      out.lines_fixed += s.linesFixed;
      out.ref_words_fixed_lines += s.refWordsFixed;
      out.ref_words_all_lines += s.refWordsAll;
      out.word_errors += s.errors;
    }
    const rttm = rttmDir ? join(rttmDir, `${basename(dir)}.rttm`) : null;
    if (rttm && existsSync(rttm)) {
      if (s.parts > 1) {
        out.der_skipped_multipart++;
        continue;
      }
      const d = der(parseRttm(readFileSync(rttm, "utf8")), s.turns);
      out.der_calls++;
      derSpeech += d.speech;
      derErrors += d.missed + d.falseAlarm + d.confusion;
    }
  }
  out.wer_fixed_lines = pct(out.word_errors, out.ref_words_fixed_lines);
  out.wer_lower_bound = pct(out.word_errors, out.ref_words_all_lines);
  out.term_recall = pct(termHits, out.term_occurrences);
  out.der = pct(derErrors, derSpeech);
  return out;
}

/**
 * The JSON to write, or an error when any value is not a number, a boolean or null: the one gate
 * between a call's text and the repository.
 */
export function numbersOnly(result: object): string {
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
    throw new Error(`${at} is a ${typeof v}: eval-local writes numbers only, never text`);
  };
  walk(result, "result");
  return `${JSON.stringify(result, null, 2)}\n`;
}

/** The authors whose names teach akou a voice: the user, in the window or through the CLI. */
const TEACHERS = new Set(["user", "agent:cli"]);

/**
 * The live call labels of one call that stand for a person: `c<N>` labels named by `user` or
 * `agent:cli`, less a label whose name changed later, a name given to two labels and a label merged
 * into another: its lines show under the other's name, so the merge renamed it. Label to the
 * person's key (the name, trimmed and lower case).
 */
export function namedLabels(events: readonly LogEvent[]): Map<string, string> {
  const view = fold(events);
  const names = new Map<string, { names: Set<string>; last: string; by: string }>();
  for (const e of events) {
    if (e.type !== "speaker.name" || !/^c\d+$/.test(e.spk)) continue;
    const key = e.name.trim().toLowerCase();
    const n = names.get(e.spk) ?? { names: new Set<string>(), last: key, by: e.by };
    n.names.add(key);
    n.last = key;
    n.by = e.by;
    names.set(e.spk, n);
  }
  const kept = [...names].filter(
    ([spk, n]) => n.names.size === 1 && TEACHERS.has(n.by) && view.resolveSpeaker(spk) === spk,
  );
  const count = new Map<string, number>();
  for (const [, n] of kept) count.set(n.last, (count.get(n.last) ?? 0) + 1);
  return new Map(
    kept
      .filter(([, n]) => n.last !== "" && count.get(n.last) === 1)
      .map(([spk, n]) => [spk, n.last]),
  );
}

/** The call channel (right) of a part's audio, 16 kHz mono, through ffmpeg; null when missing. */
export function readCallChannel(dir: string, part: number): Float32Array | null {
  const opus = join(dir, partFile(part));
  const wav = opus.replace(/\.opus$/, ".wav");
  const path = existsSync(wav) ? wav : existsSync(opus) ? opus : null;
  if (!path) return null;
  const run = spawnSync(
    ffmpegCommand(),
    [
      "-nostdin",
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      path,
      "-map",
      "0:a:0",
      "-af",
      "pan=mono|c0=c1",
      "-ar",
      String(ASR_RATE),
      "-f",
      "f32le",
      "pipe:1",
    ],
    { maxBuffer: 2 ** 31 },
  );
  if (run.status !== 0 || !run.stdout) return null;
  const b = run.stdout;
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + (b.length - (b.length % 4))));
}

export interface VoiceReaders {
  /** One line's embedding, from 16 kHz mono samples. */
  embed(samples: Float32Array): Float32Array;
  /** A part's call channel, 16 kHz mono; null when it cannot be read. */
  read(dir: string, part: number): Float32Array | null;
}

/**
 * The person each key stands for when a shorter name is the start of exactly one longer name: a
 * first name used in some calls and the full name in others are one person. Keys with no such
 * longer name are left out.
 */
export function nameAliases(keys: Iterable<string>): Map<string, string> {
  const all = [...new Set(keys)];
  const words = (k: string) => k.split(/\s+/).filter(Boolean);
  const out = new Map<string, string>();
  for (const k of all) {
    const w = words(k);
    const longer = all.filter((o) => {
      const x = words(o);
      return x.length > w.length && w.every((t, i) => t === x[i]);
    });
    if (longer.length === 1) out.set(k, longer[0] as string);
  }
  return out;
}

/** `named` with each key replaced by its alias, less any person now on two labels of the call. */
export function aliased(
  named: ReadonlyMap<string, string>,
  alias: ReadonlyMap<string, string>,
): Map<string, string> {
  const to = new Map([...named].map(([spk, k]) => [spk, alias.get(k) ?? k]));
  const count = new Map<string, number>();
  for (const k of to.values()) count.set(k, (count.get(k) ?? 0) + 1);
  return new Map([...to].filter(([, k]) => count.get(k) === 1));
}

/** The live `c<N>` speakers of a call that nobody named and that no merge folded into another. */
export function unnamedLabels(events: readonly LogEvent[]): Set<string> {
  const view = fold(events);
  const named = new Set(
    events.filter((e) => e.type === "speaker.name").map((e) => (e as { spk: string }).spk),
  );
  const out = new Set<string>();
  for (const e of events)
    if (
      e.type === "seg" &&
      /^c\d+$/.test(e.spk) &&
      !named.has(e.spk) &&
      view.resolveSpeaker(e.spk) === e.spk
    )
      out.add(e.spk);
  return out;
}

/** The named labels and the open set (labels nobody named) of every call under `root`. */
export async function voiceLabels(
  root: string,
  io: VoiceReaders,
): Promise<{ labels: VoiceLabel[]; open: VoiceLabel[]; calls: number; skipped: number }> {
  const labels: VoiceLabel[] = [];
  const open: VoiceLabel[] = [];
  let calls = 0;
  let skipped = 0;
  const logs: { dir: string; events: LogEvent[] }[] = [];
  for (const { dir } of listCallDirs(root))
    logs.push({ dir, events: (await readLog(join(dir, EVENTS_FILE))).events });
  const alias = nameAliases(logs.flatMap(({ events }) => [...namedLabels(events).values()]));
  for (const { dir, events } of logs) {
    const named = aliased(namedLabels(events), alias);
    const unnamed = unnamedLabels(events);
    if (named.size === 0 && unnamed.size === 0) continue;
    if (named.size > 0) calls++;
    const view = fold(events);
    const parts = view.parts();
    const t0 = parts[0]?.wallStart ?? 0;
    const offset = new Map(parts.map((p) => [p.part, (p.wallStart - t0) / 1000]));
    const ids = [...new Set(events.filter((e) => e.type === "seg").map((e) => e.id))];
    const audio = new Map<number, Float32Array | null>();
    const lines = new Map<string, { vec: Float32Array; secs: number; t: number }[]>();
    let missing = false;
    for (const id of ids) {
      const s = view.segment(id);
      if (s?.layer !== "live" || s.ch !== "call" || s.text === null || s.echo) continue;
      if (!(named.has(s.spk) || unnamed.has(s.spk)) || s.a1 - s.a0 < MIN_CHUNK_SECONDS) continue;
      if (!audio.has(s.part)) audio.set(s.part, io.read(dir, s.part));
      const samples = audio.get(s.part);
      if (!samples) {
        missing = true;
        continue;
      }
      const end = Math.min(s.a1, s.a0 + MAX_CHUNK_SECONDS);
      const cut = samples.subarray(Math.round(s.a0 * ASR_RATE), Math.round(end * ASR_RATE));
      if (cut.length < MIN_CHUNK_SECONDS * ASR_RATE) continue;
      const list = lines.get(s.spk) ?? [];
      list.push({ vec: io.embed(cut), secs: end - s.a0, t: (offset.get(s.part) ?? 0) + end });
      lines.set(s.spk, list);
    }
    if (missing) skipped++;
    const at = parts[0]?.wallStart ?? calls;
    for (const spk of [...lines.keys()].sort()) {
      const b = new LabelBuilder();
      for (const l of (lines.get(spk) ?? []).sort((x, y) => x.t - y.t)) b.add(l.vec, l.secs, l.t);
      const person = named.get(spk);
      const label = b.label(dir, person ?? "", at);
      if (label) (person === undefined ? open : labels).push(label);
    }
  }
  return { labels, open, calls, skipped };
}

async function voicesMain(root: string, flag: (n: string) => string | undefined): Promise<number> {
  const embed = openEmbedder(voiceModelPath(flag("--model"), flag("--models")));
  const { labels, open, calls, skipped } = await voiceLabels(root, {
    embed,
    read: readCallChannel,
  });
  if (labels.length === 0) {
    console.error(`eval-local: no named call speaker under ${root}`);
    return 66;
  }
  const day = new Date().toISOString().slice(0, 10);
  const out =
    flag("--out") ?? join(import.meta.dir, "..", "docs", "gates", `voices-local-${day}.json`);
  mkdirSync(dirname(out), { recursive: true });
  const run = evaluateVoices({ learn: labels, open, maxWrong: 0 });
  return finish(
    { ...run, result: { ...run.result, calls_read: calls, calls_without_audio: skipped } },
    out,
    "eval-local",
  );
}

async function main(argv: string[]): Promise<number> {
  const flag = (n: string) => {
    const i = argv.indexOf(n);
    return i === -1 ? undefined : argv[i + 1];
  };
  const voices = flag("--voices");
  if (voices) return voicesMain(voices, flag);
  const root = flag("--root") ?? join(homedir(), "Recordings", "akou");
  const day = new Date().toISOString().slice(0, 10);
  const out =
    flag("--out") ?? join(import.meta.dir, "..", "docs", "gates", `eval-local-${day}.json`);
  const result = await evaluate(root, flag("--rttm"));
  if (result.calls === 0) {
    console.error(`eval-local: no call under ${root}; pass --root`);
    return 66;
  }
  let text: string;
  try {
    text = numbersOnly(result);
  } catch (err) {
    console.error(`eval-local: refusing to write ${out}: ${(err as Error).message}`);
    return 65;
  }
  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, text);
  console.log(text.trimEnd());
  console.error(`eval-local: wrote ${out}`);
  return 0;
}

if (import.meta.main) process.exit(await main(process.argv.slice(2)));
