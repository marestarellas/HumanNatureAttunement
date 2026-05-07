import {useCurrentFrame, useVideoConfig, interpolate} from 'remotion';

export const Caption: React.FC<{children: React.ReactNode}> = ({children}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = interpolate(frame, [0, fps * 0.6], [0, 1], {extrapolateRight: 'clamp'});
  return (
    <div
      style={{
        position: 'absolute',
        bottom: 60,
        left: 80,
        right: 80,
        opacity: o,
        color: '#eaf1f8',
        fontSize: 30,
        fontWeight: 400,
        fontFamily: 'Inter, system-ui, sans-serif',
        lineHeight: 1.35,
        textShadow: '0 2px 12px rgba(0,0,0,0.7)',
        maxWidth: 1700,
      }}
    >
      {children}
    </div>
  );
};
