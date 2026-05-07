import {useCurrentFrame, useVideoConfig} from 'remotion';
import {SIGNALS_META} from './useSignal';

type Props = {
  signalKey: string;
  width: number;
  height: number;
  color?: string;
  cumColor?: string;
  label?: string;
  startSec?: number;
};

/**
 * A scrolling time series with an *accumulating* trail overlay (running
 * integral of |signal|, normalised). Useful for "motion has happened up
 * to now" intuition - the trail rises monotonically while the raw signal
 * jitters around.
 */
export const CumulativeLine: React.FC<Props> = ({
  signalKey,
  width,
  height,
  color = '#ff8fb1',
  cumColor = '#ffd166',
  label,
  startSec = 0,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const fs = SIGNALS_META.fs_used;
  const sig = SIGNALS_META.signals[signalKey];
  if (!sig) return null;

  const dur = SIGNALS_META.duration_s;
  const tNow = startSec + frame / fps;
  const tWrap = ((tNow % dur) + dur) % dur;
  const idxNow = Math.min(sig.length - 1, Math.max(0, Math.round(tWrap * fs)));

  // Build cumulative trail of |signal| up to idxNow
  const cum: number[] = new Array(idxNow + 1);
  let acc = 0;
  for (let i = 0; i <= idxNow; i++) {
    const v = Math.abs(sig[i] || 0);
    acc += v;
    cum[i] = acc;
  }
  const cumMax = cum[idxNow] || 1e-9;

  // Raw signal range over the full clip (for stable y-axis)
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < sig.length; i++) {
    const v = sig[i];
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  const range = Math.max(hi - lo, 1e-9);
  const padY = 10;

  const sigPts: string[] = [];
  const cumPts: string[] = [];
  for (let i = 0; i <= idxNow; i++) {
    const x = (i / Math.max(sig.length - 1, 1)) * width;
    const ySig = (1 - (sig[i] - lo) / range) * (height - 2 * padY) + padY;
    const yCum = (1 - cum[i] / cumMax) * (height - 2 * padY) + padY;
    sigPts.push(`${x.toFixed(2)},${ySig.toFixed(2)}`);
    cumPts.push(`${x.toFixed(2)},${yCum.toFixed(2)}`);
  }

  // Cursor at idxNow
  const xCursor = (idxNow / Math.max(sig.length - 1, 1)) * width;

  return (
    <svg width={width} height={height} style={{display: 'block'}}>
      <rect width={width} height={height} fill="rgba(255,255,255,0.04)" rx={6} />
      {/* cumulative trail (fill below) */}
      <polygon
        points={`0,${height} ${cumPts.join(' ')} ${xCursor.toFixed(2)},${height}`}
        fill={cumColor}
        opacity={0.18}
      />
      <polyline points={cumPts.join(' ')} fill="none" stroke={cumColor} strokeWidth={2.5} />
      {/* raw signal */}
      <polyline points={sigPts.join(' ')} fill="none" stroke={color} strokeWidth={1.6}
                opacity={0.85} />
      {/* time cursor */}
      <line x1={xCursor} y1={2} x2={xCursor} y2={height - 2}
            stroke="#ffffff" strokeWidth={1.5} opacity={0.7} />
      {label && (
        <text x={14} y={26}
              fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
              fontSize={18} fill="#cfd5dc">
          {label}
        </text>
      )}
      <text x={width - 14} y={26}
            textAnchor="end"
            fontFamily="ui-monospace, monospace"
            fontSize={16} fill={cumColor}>
        Σ |motion| = {cumMax.toFixed(1)}
      </text>
    </svg>
  );
};
