import {useCurrentFrame, useVideoConfig, interpolate} from 'remotion';

export const Caption: React.FC<{children: React.ReactNode}> = ({children}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = interpolate(frame, [0, fps * 0.6], [0, 1], {extrapolateRight: 'clamp'});
  return (
    <div
      style={{
        position: 'absolute',
        bottom: 70,
        left: 60,
        right: 60,
        opacity: o,
        color: '#dbe3ec',
        fontSize: 22,
        fontFamily: 'Inter, system-ui, sans-serif',
        lineHeight: 1.4,
        textShadow: '0 2px 12px rgba(0,0,0,0.6)',
      }}
    >
      {children}
    </div>
  );
};
