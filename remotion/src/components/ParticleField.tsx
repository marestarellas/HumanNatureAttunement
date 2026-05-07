import {useCurrentFrame, useVideoConfig} from 'remotion';
import {sampleSignalAt, SIGNALS_META} from './useSignal';

type Props = {
  width: number;
  height: number;
  count?: number;
  /** signal whose magnitude scales speed */
  speedKey?: string;
  /** signal whose value scales horizontal drift sign */
  uKey?: string;
  vKey?: string;
  color?: string;
  startSec?: number;
};

/**
 * Decorative particle field driven by the optical-flow signal. Particles drift
 * in the (u_mean, v_mean) direction with speed proportional to flow_mag_mean.
 * Adds living, intuitive motion behind the methods talk.
 */
export const ParticleField: React.FC<Props> = ({
  width,
  height,
  count = 80,
  speedKey = 'flow_mag_mean',
  uKey = 'flow_u_mean',
  vKey = 'flow_v_mean',
  color = '#7adfff',
  startSec = 0,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const dur = SIGNALS_META.duration_s;
  const tSource = ((frame / fps + startSec) % dur + dur) % dur;

  const speed = sampleSignalAt(speedKey, tSource) || 0;
  const u = sampleSignalAt(uKey, tSource) || 0;
  const v = sampleSignalAt(vKey, tSource) || 0;
  const dirNorm = Math.hypot(u, v) + 1e-6;
  const ux = u / dirNorm;
  const uy = v / dirNorm;
  const px = (frame / fps) * 60 * (0.4 + speed * 5);

  const particles = [];
  for (let i = 0; i < count; i++) {
    const seed = i * 9301 + 49297;
    const r1 = ((seed * 1103515245) % 2147483647) / 2147483647;
    const r2 = ((seed * 22695477) % 2147483647) / 2147483647;
    const baseX = r1 * width;
    const baseY = r2 * height;
    const x = ((baseX + ux * px) % (width + 60) + width + 60) % (width + 60) - 30;
    const y = ((baseY + uy * px) % (height + 60) + height + 60) % (height + 60) - 30;
    const a = 0.3 + 0.7 * (((seed * 7) % 1000) / 1000);
    particles.push(
      <circle
        key={i}
        cx={x}
        cy={y}
        r={1.4 + (((seed * 13) % 1000) / 1000) * 1.6}
        fill={color}
        opacity={a * (0.3 + Math.min(1, speed * 4))}
      />
    );
  }
  return (
    <svg width={width} height={height} style={{position: 'absolute', top: 0, left: 0,
                                               pointerEvents: 'none'}}>
      {particles}
    </svg>
  );
};
