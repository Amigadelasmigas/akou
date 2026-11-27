#!/usr/bin/env bun

/**
 * The voices evaluation (docs/TESTING.md TS-21b) on the AMI Meeting Corpus, audio and annotations
 * CC-BY-4.0. Each scenario series is the same four people in four meetings, so a voice named in
 * one meeting can be looked for in the other three.
 *
 *   bun scripts/eval/voices-ami.ts <dir> [--far] [--opus] [--gallery N] [--model titanet-small|<file.onnx>] [--models <dir>] [--out <file>]
 *
 * - The series: ES2004a-d and IS1009a-d (test), ES2002a-d (train) and ES2011a-d (dev). The audio
 *   and the `only_words` references of the AMI diarization setup are fetched into `<dir>` once; a
 *   download answered with HTTP 503 is tried again.
 * - A label is one person in one meeting, by the reference's speaker id. Their turns are cut the
 *   way the live worker cuts lines: turns under 1 s are left out, longer ones go in pieces of at
 *   most 12 s, each placed at its end in the meeting, so the labels' writes come in meeting order.
 * - `--far` learns on the headset mix and matches on the first microphone of the array, the case
 *   of a room microphone against a headset.
 * - `--opus` passes all audio through one Opus encode at 48 kb/s, stereo, the way akou writes a
 *   part, before it is embedded. Needs ffmpeg.
 * - `--gallery N` enrols up to N more people, from one meeting of each other training series, as
 *   impostors: they are in every store and never matched, to test a large store.
 * - The gate allows 0.5 % wrong names here. Only numbers are written.
 */

import { spawnSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { ffmpegCommand } from "../../src/main/asr/decode.ts";
import { ASR_RATE } from "../../src/main/asr/engine.ts";
import { AMI, readWav } from "./nightly.ts";
import { parseRttm, type Turn } from "./score.ts";
import {
  evaluateVoices,
  finish,
  LabelBuilder,
  MAX_CHUNK_SECONDS,
  MIN_CHUNK_SECONDS,
  openEmbedder,
  type VoiceLabel,
  voiceModelPath,
} from "./voices.ts";

const SETUP =
  "https://raw.githubusercontent.com/pyannote/AMI-diarization-setup/67c2d539286e89f68952d5dcf83912bd9f01dfae";
const RTTMS = `${SETUP}/only_words/rttms`;

/** Each series and the split of the diarization setup its references sit in. */
export const SERIES = [
  { id: "ES2004", split: "test" },
  { id: "IS1009", split: "test" },
  { id: "ES2002", split: "train" },
  { id: "ES2011", split: "dev" },
] as const;

/** The highest wrong-name rate the gate allows on AMI. */
export const AMI_MAX_WRONG = 0.005;
/** The bit rate akou writes parts at. */
export const OPUS_BITRATE = "48k";
/** Tries of one download that keeps answering HTTP 503. */
export const FETCH_TRIES = 5;

/** Pieces of a turn as the live worker would cut it: 12 s at most, none under 1 s. */
export function pieces(t: Turn): { start: number; end: number }[] {
  const out: { start: number; end: number }[] = [];
  for (let s = t.start; t.end - s >= MIN_CHUNK_SECONDS; s += MAX_CHUNK_SECONDS)
    out.push({ start: s, end: Math.min(t.end, s + MAX_CHUNK_SECONDS) });
  return out;
}

/** `fetch`, tried again after a pause while it answers HTTP 503. */
export async function fetchRetrying(
  url: string,
  get: (url: string) => Promise<Response> = (u) => fetch(u, { redirect: "follow" }),
  waitMs = 2000,
  tries = FETCH_TRIES,
): Promise<Response> {
  for (let i = 1; ; i++) {
    const r = await get(url);
    if (r.status !== 503 || i >= tries) return r;
    await Bun.sleep(waitMs * i);
  }
}

async function fetchOnce(url: string, path: string): Promise<void> {
  if (existsSync(path)) return;
  const r = await fetchRetrying(url);
  if (!r.ok) throw new Error(`GET ${url}: HTTP ${r.status}`);
  writeFileSync(`${path}.part`, new Uint8Array(await r.arrayBuffer()));
  renameSync(`${path}.part`, path);
}

/** A wav file after one Opus pass (cached beside it), back to 16 kHz mono. */
function opusPass(wav: string): Float32Array {
  const opus = wav.replace(/\.wav$/, ".opus");
  const ffmpeg = ffmpegCommand();
  if (!existsSync(opus)) {
    const enc = spawnSync(ffmpeg, [
      "-nostdin",
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      wav,
      "-ac",
      "2",
      "-c:a",
      "libopus",
      "-b:a",
      OPUS_BITRATE,
      `${opus}.part.opus`,
    ]);
    if (enc.status !== 0) throw new Error(`ffmpeg could not encode ${wav}: ${enc.stderr}`);
    renameSync(`${opus}.part.opus`, opus);
  }
  const dec = spawnSync(
    ffmpeg,
    [
      "-nostdin",
      "-hide_banner",
      "-loglevel",
      "error",
      "-i",
      opus,
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
  if (dec.status !== 0 || !dec.stdout) throw new Error(`ffmpeg could not decode ${opus}`);
  const b = dec.stdout;
  return new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + (b.length - (b.length % 4))));
}

type Embed = (samples: Float32Array) => Float32Array;
type Line = { vec: Float32Array; secs: number; t: number };

/** One meeting's lines per person, embedded from `samples`, in meeting order. */
function embedTurns(samples: Float32Array, turns: readonly Turn[], embed: Embed) {
  const by = new Map<string, Line[]>();
  for (const t of [...turns].sort((a, b) => a.start - b.start))
    for (const p of pieces(t)) {
      const cut = samples.subarray(Math.round(p.start * ASR_RATE), Math.round(p.end * ASR_RATE));
      if (cut.length < MIN_CHUNK_SECONDS * ASR_RATE) continue;
      const list = by.get(t.speaker) ?? [];
      list.push({ vec: embed(cut), secs: p.end - p.start, t: p.end });
      by.set(t.speaker, list);
    }
  for (const list of by.values()) list.sort((a, b) => a.t - b.t);
  return by;
}

function toLabels(by: Map<string, Line[]>, call: string, at: number): VoiceLabel[] {
  const out: VoiceLabel[] = [];
  for (const [person, lines] of [...by].sort(([a], [b]) => a.localeCompare(b))) {
    const b = new LabelBuilder();
    for (const l of lines) b.add(l.vec, l.secs, l.t);
    const label = b.label(call, person, at);
    if (label) out.push(label);
  }
  return out;
}

/** One meeting of each training series that is not one of `SERIES`, in the setup's order. */
export function galleryMeetings(list: string): string[] {
  const skip = new Set<string>(SERIES.map((s) => s.id));
  const seen = new Set<string>();
  const out: string[] = [];
  for (const m of list.split(/\s+/).filter(Boolean)) {
    const series = m.slice(0, 6);
    if (skip.has(series) || seen.has(series)) continue;
    seen.add(series);
    out.push(m);
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
    console.error(
      "usage: bun scripts/eval/voices-ami.ts <dir> [--far] [--opus] [--gallery N] [--model <id|file.onnx>]",
    );
    return 64;
  }
  const far = argv.includes("--far");
  const opus = argv.includes("--opus");
  const gallery = Number(flag("--gallery") ?? 0);
  if (!Number.isInteger(gallery) || gallery < 0) {
    console.error("voices-ami: --gallery takes a whole number of people");
    return 64;
  }
  mkdirSync(dir, { recursive: true });
  const embed = openEmbedder(voiceModelPath(flag("--model"), flag("--models")));
  const read = async (id: string, mic: string) => {
    const wav = join(dir, `${id}.${mic}.wav`);
    await fetchOnce(`${AMI.audio}/${id}/audio/${id}.${mic}.wav`, wav);
    return opus ? opusPass(wav) : readWav(new Uint8Array(readFileSync(wav)));
  };
  const turnsOf = async (id: string, split: string) => {
    const rttm = join(dir, `${id}.rttm`);
    await fetchOnce(`${RTTMS}/${split}/${id}.rttm`, rttm);
    return parseRttm(readFileSync(rttm, "utf8"));
  };
  const learn: VoiceLabel[] = [];
  const match: VoiceLabel[] = [];
  let at = 0;
  for (const s of SERIES)
    for (const letter of ["a", "b", "c", "d"]) {
      const id = `${s.id}${letter}`;
      const turns = await turnsOf(id, s.split);
      const head = toLabels(embedTurns(await read(id, "Mix-Headset"), turns, embed), id, at);
      learn.push(...head);
      match.push(
        ...(far ? toLabels(embedTurns(await read(id, "Array1-01"), turns, embed), id, at) : head),
      );
      at++;
      console.error(`voices-ami: ${id} embedded`);
    }
  let enrolled = 0;
  if (gallery > 0) {
    const listFile = join(dir, "train.meetings.txt");
    await fetchOnce(`${SETUP}/lists/train.meetings.txt`, listFile);
    const known = new Set(learn.map((l) => l.person));
    for (const id of galleryMeetings(readFileSync(listFile, "utf8"))) {
      if (enrolled >= gallery) break;
      const turns = await turnsOf(id, "train");
      const labels = toLabels(embedTurns(await read(id, "Mix-Headset"), turns, embed), id, at++);
      for (const l of labels) {
        if (enrolled >= gallery || known.has(l.person)) continue;
        known.add(l.person);
        learn.push(l);
        enrolled++;
      }
      console.error(`voices-ami: gallery ${id} embedded, ${enrolled} enrolled`);
    }
  }
  const run = evaluateVoices({ learn, match, maxWrong: AMI_MAX_WRONG });
  return finish(
    {
      ...run,
      result: { ...run.result, far: far ? 1 : 0, opus: opus ? 1 : 0, gallery: enrolled },
    },
    flag("--out"),
    "voices-ami",
  );
}

if (import.meta.main) process.exit(await main(process.argv.slice(2)));
