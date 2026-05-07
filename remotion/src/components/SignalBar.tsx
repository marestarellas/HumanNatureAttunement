import {useSignal} from './useSignal';

type Props = {
  signalKey: string;
  width: number;
  height: number;
  color?: string;
  label?: string;
  windowSec?: number;
  vertical?: boolean;
  startSec?: number;
};

/**
 * A live-updating bar whose fill tracks the signal's value relative to its
 * recent min/max. Vertical by default — gives a "VU-meter" feel.
 */
export const SignalBar: React.FC<Props> = ({
  signalKey,
  width,
  height,
  color = '#7adfff',
  label,
  windowSec = 6,
  vertical = true,
  startSec = 0,
}) => {
  const {value, norm, lo, hi} = useSignal(signalKey, windowSec, startSec);
  const fill = Math.max(0.02, Math.min(1, norm));
  const fillW = vertical ? width : width * fill;
  const fillH = vertical ? height * fill : height;
  const fillX = 0;
  const fillY = vertical ? height - fillH : 0;
  return (
    <div style={{position: 'relative', width, height}}>
      <svg width={width} height={height}>
        <rect
          x={0}
          y={0}
          width={width}
          height={height}
          fill="rgba(255,255,255,0.05)"
          stroke="rgba(255,255,255,0.18)"
          strokeWidth={1.5}
          rx={6}
        />
        <rect
          x={fillX}
          y={fillY}
          width={fillW}
          height={fillH}
          fill={color}
          opacity={0.85}
          rx={6}
        />
        {/* glowing tip */}
        <rect
          x={fillX}
          y={vertical ? Math.max(0, fillY - 3) : fillW - 3}
          width={vertical ? width : 3}
          height={vertical ? 3 : height}
          fill={color}
        />
      </svg>
      {label && (
        <div
          style={{
            position: 'absolute',
            top: vertical ? -22 : height + 4,
            left: 0,
            right: 0,
            textAlign: 'center',
            color: '#cfd5dc',
            fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
            fontSize: 13,
          }}
        >
          {label}
        </div>
      )}
      <div
        style={{
          position: 'absolute',
          bottom: vertical ? -22 : -22,
          left: 0,
          right: 0,
          textAlign: 'center',
          color: color,
          fontFamily: 'ui-monospace, monospace',
          fontSize: 12,
          opacity: 0.7,
        }}
      >
        {Number.isFinite(value) ? value.toFixed(2) : '—'}
      </div>
    </div>
  );
};
