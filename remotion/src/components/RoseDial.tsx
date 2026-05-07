import {useSignal} from './useSignal';

type Props = {
  uKey: string;             // signal: mean horizontal flow (e.g. flow_u_mean)
  vKey: string;             // signal: mean vertical flow (e.g. flow_v_mean)
  size: number;
  windowSec?: number;
  label?: string;
};

/**
 * A compass-like rose dial showing the instantaneous mean flow direction and
 * magnitude. Concrete "where the wave is going right now" intuition.
 */
export const RoseDial: React.FC<Props> = ({
  uKey,
  vKey,
  size,
  windowSec = 1.0,
  label = 'mean flow',
}) => {
  const u = useSignal(uKey, windowSec).value || 0;
  const v = useSignal(vKey, windowSec).value || 0;
  const mag = Math.hypot(u, v);
  const ang = Math.atan2(v, u);
  // Normalize arrow to fit the dial; track recent magnitudes through the
  // hook itself by reading the magnitude signal if available — fallback to
  // a soft cap of 0.6 px/frame which is typical for water flow at this scale.
  const cap = 0.6;
  const r = (size / 2) * Math.min(1, mag / cap) * 0.85;
  const cx = size / 2;
  const cy = size / 2;
  const tipX = cx + r * Math.cos(ang);
  const tipY = cy + r * Math.sin(ang);
  return (
    <div style={{position: 'relative', width: size, height: size}}>
      <svg width={size} height={size}>
        <circle cx={cx} cy={cy} r={size / 2 - 4}
                fill="rgba(255,255,255,0.04)"
                stroke="rgba(255,255,255,0.18)" strokeWidth={1.5} />
        {/* tick marks */}
        {Array.from({length: 12}).map((_, i) => {
          const t = (i / 12) * 2 * Math.PI;
          const r0 = size / 2 - 4;
          const r1 = r0 - (i % 3 === 0 ? 14 : 7);
          return (
            <line
              key={i}
              x1={cx + r0 * Math.cos(t)}
              y1={cy + r0 * Math.sin(t)}
              x2={cx + r1 * Math.cos(t)}
              y2={cy + r1 * Math.sin(t)}
              stroke="rgba(255,255,255,0.35)"
              strokeWidth={1.2}
            />
          );
        })}
        <line
          x1={cx} y1={cy} x2={tipX} y2={tipY}
          stroke="#ffd166" strokeWidth={4} strokeLinecap="round"
        />
        <circle cx={cx} cy={cy} r={5} fill="#ffd166" />
        <circle cx={tipX} cy={tipY} r={6} fill="#ffd166" />
      </svg>
      <div
        style={{
          position: 'absolute',
          bottom: -22,
          left: 0,
          right: 0,
          textAlign: 'center',
          color: '#cfd5dc',
          fontFamily: 'ui-monospace, monospace',
          fontSize: 13,
        }}
      >
        {label} · |v|={mag.toFixed(3)}
      </div>
    </div>
  );
};
