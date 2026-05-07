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

type Props = {
  /** key into signals.json -> "signals" */
  signalKey: string;
  /** seconds of signal visible at once */
  windowSec: number;
  /** seconds offset added to currentFrame/30 to align with the underlying clip's playhead */
  startSec?: number;
  width: number;
  height: number;
  color?: string;
  label?: string;
  yPad?: number;
};

const norm = (xs: number[]): {y: number[]; lo: number; hi: number} => {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of xs) {
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi) || hi - lo < 1e-9) {
    return {y: xs.map(() => 0.5), lo, hi};
  }
  const range = hi - lo;
  return {y: xs.map((v) => (Number.isFinite(v) ? (v - lo) / range : 0.5)), lo, hi};
};

export const ScrollingLine: React.FC<Props> = ({
  signalKey,
  windowSec,
  startSec = 0,
  width,
  height,
  color = '#7adfff',
  label,
  yPad = 8,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const tNow = startSec + frame / fps;

  const sig = SIGNALS.signals[signalKey];
  if (!sig) {
    return (
      <div style={{color: 'red', fontSize: 18, padding: 8}}>
        unknown signal: {signalKey}
      </div>
    );
  }
  const fs = SIGNALS.fs_used;
  const idxNow = Math.min(sig.length - 1, Math.max(0, Math.round(tNow * fs)));
  const winSamples = Math.max(2, Math.round(windowSec * fs));
  const idxStart = Math.max(0, idxNow - winSamples + 1);
  const slice = sig.slice(idxStart, idxNow + 1);

  const {y, lo, hi} = norm(slice);
  const points = y
    .map((v, i) => {
      const x = (i / Math.max(y.length - 1, 1)) * width;
      const py = height - yPad - v * (height - 2 * yPad);
      return `${x.toFixed(2)},${py.toFixed(2)}`;
    })
    .join(' ');

  return (
    <svg width={width} height={height} style={{display: 'block'}}>
      <rect width={width} height={height} fill="rgba(255,255,255,0.03)" />
      <polyline points={points} fill="none" stroke={color} strokeWidth={2.5} />
      {label && (
        <text
          x={12}
          y={22}
          fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
          fontSize={16}
          fill="#cfd5dc"
        >
          {label}
        </text>
      )}
      {Number.isFinite(lo) && Number.isFinite(hi) && (
        <text
          x={width - 12}
          y={22}
          textAnchor="end"
          fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
          fontSize={13}
          fill="#7c8590"
        >
          [{lo.toFixed(3)}, {hi.toFixed(3)}]
        </text>
      )}
    </svg>
  );
};
