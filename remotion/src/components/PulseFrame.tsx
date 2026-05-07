import {OffthreadVideo, staticFile} from 'remotion';
import {useSignal} from './useSignal';

type Props = {
  src: string;                       // staticFile-resolvable path under public/
  width: number;
  height: number;
  signalKey: string;                 // signal driving the pulse
  windowSec?: number;
  /** how strongly the border color/glow reacts (px) */
  glowStrength?: number;
  borderColor?: string;
  rounded?: number;
};

/**
 * A video tile whose border glow + inner pulse react to a 1-D signal. The
 * video itself plays straight; the *frame around it* makes the math visible.
 */
export const PulseFrame: React.FC<Props> = ({
  src,
  width,
  height,
  signalKey,
  windowSec = 4,
  glowStrength = 60,
  borderColor = '#ffd166',
  rounded = 14,
}) => {
  const {norm} = useSignal(signalKey, windowSec);
  const glow = glowStrength * (0.15 + 0.85 * norm);
  const scale = 1 + 0.015 * norm;
  return (
    <div
      style={{
        position: 'relative',
        width,
        height,
        borderRadius: rounded,
        overflow: 'hidden',
        boxShadow: `0 0 ${glow}px ${Math.max(2, glow / 6)}px ${borderColor}`,
        outline: `2px solid ${borderColor}`,
        transform: `scale(${scale})`,
        transformOrigin: 'center center',
        transition: 'none',
      }}
    >
      <OffthreadVideo src={staticFile(src)} muted />
    </div>
  );
};
