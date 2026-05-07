import {useCurrentFrame, useVideoConfig} from 'remotion';
import {SIGNALS_META} from './useSignal';

type Props = {
  signalKey: string;
  width: number;
  height: number;
  /** how many samples to embed (m). m=3 → 6 patterns, m=4 → 24 */
  m?: number;
  /** delay between samples (in source-signal samples) */
  tau?: number;
  startSec?: number;
};

/**
 * Visualize permutation entropy's *building block*: take m consecutive samples
 * of the signal, plot them as dots, label their ordinal pattern (e.g. "▼▲ —
 * pattern #2 of 6"). As the scene plays, the pattern label updates on every
 * step, giving an immediate intuition for what the entropy counts.
 */
export const OrdinalGlyph: React.FC<Props> = ({
  signalKey,
  width,
  height,
  m = 3,
  tau = 4,
  startSec = 0,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const fs = SIGNALS_META.fs_used;
  const sig = SIGNALS_META.signals[signalKey];
  if (!sig) return null;
  const dur = SIGNALS_META.duration_s;
  const tSource = ((frame / fps + startSec) % dur + dur) % dur;
  const idx = Math.min(sig.length - 1 - (m - 1) * tau,
                       Math.max(0, Math.round(tSource * fs)));
  const samples: number[] = [];
  for (let i = 0; i < m; i++) samples.push(sig[idx + i * tau]);
  const sorted = samples.map((v, i) => ({v, i}))
                        .sort((a, b) => a.v - b.v);
  const ranks: number[] = new Array(m);
  sorted.forEach((s, r) => (ranks[s.i] = r));

  const pad = 30;
  const innerW = width - 2 * pad;
  const innerH = height - 2 * pad - 30;
  const minV = Math.min(...samples);
  const maxV = Math.max(...samples);
  const span = Math.max(1e-9, maxV - minV);
  const xs = samples.map((_, i) => pad + (i / Math.max(1, m - 1)) * innerW);
  const ys = samples.map((v) => pad + (1 - (v - minV) / span) * innerH);

  // Pattern label: "▲ ▼ ▲" style symbols comparing successive samples
  const arrows = samples.slice(1).map((v, i) => (v > samples[i] ? '↗' : '↘')).join(' ');
  const patternId = ranks.join('');

  return (
    <svg width={width} height={height}>
      <rect x={1} y={1} width={width - 2} height={height - 2}
            rx={10}
            fill="rgba(255,255,255,0.04)"
            stroke="rgba(255,255,255,0.18)" />
      <polyline
        points={xs.map((x, i) => `${x},${ys[i]}`).join(' ')}
        fill="none"
        stroke="#7adfff"
        strokeWidth={3}
      />
      {samples.map((_, i) => (
        <g key={i}>
          <circle cx={xs[i]} cy={ys[i]} r={9} fill="#ffd166" />
          <text x={xs[i]} y={ys[i] + 4}
                fontFamily="ui-monospace, monospace" fontSize={12}
                fill="#04070b" textAnchor="middle" fontWeight={700}>
            {ranks[i] + 1}
          </text>
        </g>
      ))}
      <text x={width / 2} y={height - 12}
            textAnchor="middle"
            fontFamily="ui-monospace, monospace" fontSize={20}
            fill="#cfd5dc">
        ordinal pattern: {arrows} · #{patternId}
      </text>
    </svg>
  );
};
