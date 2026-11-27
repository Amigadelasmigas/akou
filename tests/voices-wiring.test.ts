/**
 * Voices in the running app: a headless app with the fake helper and recognizer, its voices Host
 * fed through `app.watch`, and the thresholds passed in so no test depends on the measured values.
 * The centroids are written as the live worker writes them; names go through the API as the CLI
 * and an agent send them.
 */

import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import type { LogEvent } from "../src/core/log/events.ts";
import { encodeVec } from "../src/main/asr/speakers.ts";
import { LIVE_EMBEDDER } from "../src/main/voices/host.ts";
import { VoiceStore, voicesPath } from "../src/main/voices/store.ts";
import type { VoiceThresholds } from "../src/main/voices/thresholds.ts";
import { type AppRig, appRig } from "./api-helpers.ts";

const LONG = 30_000;
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

const store = (rig: AppRig) => new VoiceStore(voicesPath(rig.app.configDir), { log: () => {} });

async function people(rig: AppRig): Promise<Record<string, string[]>> {
  const out: Record<string, string[]> = {};
  for (const p of (await store(rig).load(LIVE_EMBEDDER, 4)).people)
    out[p.name] = p.exemplars.map((e) => `${e.call}/${e.spk}`);
  return out;
}

async function voiceEvents(rig: AppRig, id: string) {
  await rig.app.voices?.idle();
  return (await rig.app.events(id, 0)).filter(
    (e): e is Extract<LogEvent, { type: "speaker.voice" }> => e.type === "speaker.voice",
  );
}

async function centroid(rig: AppRig, id: string, spk: string, vec: Float32Array, secs: number) {
  await rig.app.write(id, { type: "speaker.centroid", spk, vec: encodeVec(vec), secs, cons: 0.8 });
}

async function name(rig: AppRig, id: string, spk: string, n: string, client: string) {
  const r = await rig.api(
    "POST",
    `/calls/${id}/speakers`,
    { spk, name: n },
    { "x-akou-client": client },
  );
  expect(r.status).toBe(200);
}

describe("voices in the app", () => {
  let rig: AppRig;
  beforeAll(async () => {
    rig = await appRig({ voiceThresholds: () => T });
    await store(rig).upsert("Ben", {
      model: LIVE_EMBEDDER,
      dim: 4,
      vec: at(0),
      secs: 60,
      cons: 0.8,
      call: "earlier",
      spk: "c1",
      at: new Date().toISOString(),
    });
  }, LONG);
  afterAll(async () => {
    await rig?.close();
  }, LONG);

  test(
    "a fake call through app.watch: a known voice is named, a CLI name teaches, an MCP name does not",
    async () => {
      const id = await rig.startCall();
      try {
        await centroid(rig, id, "c5", at(2), 30);
        expect(await voiceEvents(rig, id)).toEqual([
          expect.objectContaining({ spk: "c5", name: "Ben", status: "named", score: 1, by: "app" }),
        ]);
        await centroid(rig, id, "c6", at(90), 30);
        await centroid(rig, id, "c7", at(-90), 30);
        await name(rig, id, "c6", "Cy", "cli");
        await name(rig, id, "c7", "Di", "claude-code");
        await rig.app.voices?.idle();
        expect(await people(rig)).toEqual({ Ben: ["earlier/c1"], Cy: [`${id}/c6`] });
        // The voice moves away from Ben: the name is withdrawn by the app.
        await centroid(rig, id, "c5", at(60), 40);
        expect((await voiceEvents(rig, id)).map((e) => [e.spk, e.name, e.status])).toEqual([
          ["c5", "Ben", "named"],
          ["c5", "Ben", "refused"],
        ]);
      } finally {
        await rig.api("POST", "/calls/live/stop");
      }

      // [VO-4] Trash then restore keeps the call's exemplars; deleting it for good drops them.
      expect((await rig.api("DELETE", `/calls/${id}`)).status).toBe(200);
      expect((await rig.api("POST", `/calls/${id}/restore`)).status).toBe(200);
      await rig.app.voices?.idle();
      expect(await people(rig)).toEqual({ Ben: ["earlier/c1"], Cy: [`${id}/c6`] });
      expect((await rig.api("DELETE", `/calls/${id}`)).status).toBe(200);
      const empty = await rig.api("DELETE", "/trash");
      expect(empty.body).toMatchObject({ ok: true, deleted: 1 });
      await rig.app.voices?.idle();
      // The positive control: what an earlier call taught stays.
      expect(await people(rig)).toEqual({ Ben: ["earlier/c1"] });
    },
    LONG,
  );
});

describe("[VO-8] server mode", () => {
  test(
    "has no voices Host, so nothing learns or matches",
    async () => {
      const server = await appRig({
        settings: { "server.enabled": true, "api.bind": "127.0.0.1" },
        voiceThresholds: () => T,
      });
      // The positive control: the desktop app has one.
      const app = await appRig({ voiceThresholds: () => T });
      try {
        expect(server.app.voices).toBeNull();
        expect(app.app.voices).not.toBeNull();
      } finally {
        await server.close();
        await app.close();
      }
    },
    LONG,
  );
});
