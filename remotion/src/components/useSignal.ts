import {useCurrentFrame, useVideoConfig} from 'remotion';
import signalsRaw from '../data/signals.json';

type SignalsPayload = {
  fps: number;
  fs_used: number;
  n_frames: number;
  duration_s: number;
  signals: Record<string, number[]>;
};
const SIGNALS = signalsRaw as SignalsPayload;

/**
 * Sample a 1-D signal at the current scene time and return:
 *  - raw value at this frame (clamped),
 *  - normalized [0,1] over a sliding window centered on now,
 *  - global min/max over that window.
 * `startSec` lets a scene shift its phase relative to the source clip.
 */
export const useSignal = (
  key: string,
  windowSec = 4.0,
  startSec = 0
): {value: number; norm: number; lo: number; hi: number; sourceTimeSec: number} => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const fs = SIGNALS.fs_used;
  const sig = SIGNALS.signals[key];
  if (!sig) return {value: 0, norm: 0.5, lo: 0, hi: 1, sourceTimeSec: 0};

  // Loop the source so a long scene keeps showing fresh signal.
  const dur = SIGNALS.duration_s;
  const tScene = frame / fps + startSec;
  const tSource = ((tScene % dur) + dur) % dur;

  const idx = Math.min(sig.length - 1, Math.max(0, Math.round(tSource * fs)));
  const win = Math.max(2, Math.round(windowSec * fs));
  const start = Math.max(0, idx - Math.floor(win / 2));
  const end = Math.min(sig.length, start + win);
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = start; i < end; i++) {
    const v = sig[i];
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const v = sig[idx];
  let norm = 0.5;
  if (Number.isFinite(lo) && Number.isFinite(hi) && hi - lo > 1e-9) {
    norm = (v - lo) / (hi - lo);
  }
  return {value: v, norm, lo, hi, sourceTimeSec: tSource};
};

export const sampleSignalAt = (
  key: string,
  tSourceSec: number
): number => {
  const sig = SIGNALS.signals[key];
  if (!sig) return 0;
  const idx = Math.min(sig.length - 1,
                       Math.max(0, Math.round(tSourceSec * SIGNALS.fs_used)));
  return sig[idx];
};

export const SIGNALS_META = SIGNALS;
