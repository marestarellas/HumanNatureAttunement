import {interpolate, useCurrentFrame, spring, useVideoConfig} from 'remotion';

type Props = {title: string; subtitle?: string};

export const MethodTitle: React.FC<Props> = ({title, subtitle}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = spring({frame, fps, config: {damping: 200}});
  const y = interpolate(o, [0, 1], [20, 0]);
  return (
    <div
      style={{
        position: 'absolute',
        top: 36,
        left: 60,
        opacity: o,
        transform: `translateY(${y}px)`,
        color: '#e7eef5',
        fontFamily: 'Inter, system-ui, sans-serif',
      }}
    >
      <div style={{fontSize: 44, fontWeight: 700, letterSpacing: -0.5}}>{title}</div>
      {subtitle && (
        <div style={{fontSize: 22, opacity: 0.7, marginTop: 6}}>{subtitle}</div>
      )}
    </div>
  );
};
