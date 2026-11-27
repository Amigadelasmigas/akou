#!/usr/bin/env bun

/**
 * The voices evaluation (docs/TESTING.md TS-21b) on the AMI Meeting Corpus, audio and annotations
 * CC-BY-4.0. Each scenario series is the same four people in four meetings, so a voice named in
 * one meeting can be looked for in the other three.
 *
 *   bun scripts/eval/voices-ami.ts <dir> [--far] [--model titanet-small|<file.onnx>] [--models <dir>] [--out <file>]
 *
 * - The series: ES2004a-d and IS1009a-d (test), ES2002a-d (train) and ES2011a-d (dev). The audio
 *   and the `only_words` references of the AMI diarization setup are fetched into `<dir>` once.
 * - A label is one person in one meeting, by the reference's speaker id. Their turns are cut the
 *   way the live worker cuts lines: turns under 1 s are left out, longer ones go in pieces of at
 *   most 12 s.
 * - Every pair of people of one meeting, their pieces interleaved, is a synthetic two-voice cluster
 *   for MIN_CONS.
 * - `--far` learns on the headset mix and matches on the first microphone of the array, the case
 *   of a room microphone against a headset.
 * - The gate allows 0.5 % wrong names here. Only numbers are written.
 */

import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { ASR_RATE } from "../../src/main/asr/engine.ts";
import { AMI, readWav } from "./nightly.ts";
import { parseRttm, type Turn } from "./score.ts";
import {
  evaluateVoices,
  finish,
  LabelBuilder,
  MAX_CHUNK_SECONDS,
  MIN_CHUNK_SECONDS,
  type MixedLabel,
  openEmbedder,
  type VoiceLabel,
  voiceModelPath,
} from "./voices.ts";

const RTTMS =
  "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/67c2d539286e89f68952d5dcf83912bd9f01dfae/only_words/rttms";

/** Each series and the split of the diarization setup its references sit in. */
export const SERIES = [
  { id: "ES2004", split: "test" },
  { id: "IS1009", split: "test" },
  { id: "ES2002", split: "train" },
  { id: "ES2011", split: "dev" },
] as const;

/** The highest wrong-name rate the gate allows on AMI. */
export const AMI_MAX_WRONG = 0.005;

/** Pieces of a turn as the live worker would cut it: 12 s at most, none under 1 s. */
export function pieces(t: Turn): { start: number; end: number }[] {
  const out: { start: number; end: number }[] = [];
  for (let s = t.start; t.end - s >= MIN_CHUNK_SECONDS; s += MAX_CHUNK_SECONDS)
    out.push({ start: s, end: Math.min(t.end, s + MAX_CHUNK_SECONDS) });
  return out;
}

async function fetchOnce(url: string, path: string): Promise<void> {
  if (existsSync(path)) return;
  const r = await fetch(url, { redirect: "follow" });
  if (!r.ok) throw new Error(`GET ${url}: HTTP ${r.status}`);
  writeFileSync(`${path}.part`, new Uint8Array(await r.arrayBuffer()));
  renameSync(`${path}.part`, path);
}

type Embed = (samples: Float32Array) => Float32Array;

/** One meeting's lines per person, embedded from `samples`. */
function embedTurns(samples: Float32Array, turns: readonly Turn[], embed: Embed) {
  const by = new Map<string, { vec: Float32Array; secs: number }[]>();
  for (const t of turns)
    for (const p of pieces(t)) {
      const cut = samples.subarray(Math.round(p.start * ASR_RATE), Math.round(p.end * ASR_RATE));
      if (cut.length < MIN_CHUNK_SECONDS * ASR_RATE) continue;
      const list = by.get(t.speaker) ?? [];
      list.push({ vec: embed(cut), secs: p.end - p.start });
      by.set(t.speaker, list);
    }
  return by;
}

function toLabels(
  by: Map<string, { vec: Float32Array; secs: number }[]>,
  call: string,
  at: number,
): VoiceLabel[] {
  const out: VoiceLabel[] = [];
  for (const [person, lines] of [...by].sort(([a], [b]) => a.localeCompare(b))) {
    const b = new LabelBuilder();
    for (const l of lines) b.add(l.vec, l.secs);
    const label = b.label(call, person, at);
    if (label) out.push(label);
  }
  return out;
}

function toMixed(by: Map<string, { vec: Float32Array; secs: number }[]>): MixedLabel[] {
  const people = [...by.keys()].sort();
  const out: MixedLabel[] = [];
  for (let i = 0; i < people.length; i++)
    for (let j = i + 1; j < people.length; j++) {
      const a = by.get(people[i] as string) ?? [];
      const c = by.get(people[j] as string) ?? [];
      const b = new LabelBuilder();
      for (let k = 0; k < Math.max(a.length, c.length); k++) {
        const x = a[k];
        const y = c[k];
        if (x) b.add(x.vec, x.secs);
        if (y) b.add(y.vec, y.secs);
      }
      const m = b.label("", "");
      if (m) out.push({ cons: m.cons, snaps: m.snaps });
    }
  return out;
}

async function main(argv: string[]): Promise<number> {
  const flag = (n: string) => {
    const i = argv.indexOf(n);
    return i === -1 ? undefined : argv[i + 1];
  };
  const dir = argv[0];
  if (!dir || dir.startsWith("--")) {
    console.error("usage: bun scripts/eval/voices-ami.ts <dir> [--far] [--model <id|file.onnx>]");
    return 64;
  }
  const far = argv.includes("--far");
  mkdirSync(dir, { recursive: true });
  const embed = openEmbedder(voiceModelPath(flag("--model"), flag("--models")));
  const learn: VoiceLabel[] = [];
  const match: VoiceLabel[] = [];
  const mixed: MixedLabel[] = [];
  let at = 0;
  for (const s of SERIES)
    for (const letter of ["a", "b", "c", "d"]) {
      const id = `${s.id}${letter}`;
      const rttm = join(dir, `${id}.rttm`);
      await fetchOnce(`${RTTMS}/${s.split}/${id}.rttm`, rttm);
      const turns = parseRttm(readFileSync(rttm, "utf8"));
      const read = async (mic: string) => {
        const wav = join(dir, `${id}.${mic}.wav`);
        await fetchOnce(`${AMI.audio}/${id}/audio/${id}.${mic}.wav`, wav);
        return readWav(new Uint8Array(readFileSync(wav)));
      };
      const head = embedTurns(await read("Mix-Headset"), turns, embed);
      learn.push(...toLabels(head, id, at));
      mixed.push(...toMixed(head));
      if (far) match.push(...toLabels(embedTurns(await read("Array1-01"), turns, embed), id, at));
      at++;
      console.error(`voices-ami: ${id} embedded`);
    }
  const run = evaluateVoices({
    learn,
    match: far ? match : undefined,
    mixed,
    maxWrong: AMI_MAX_WRONG,
  });
  return finish(
    { ...run, result: { ...run.result, far: far ? 1 : 0 } },
    flag("--out"),
    "voices-ami",
  );
}

if (import.meta.main) process.exit(await main(process.argv.slice(2)));
