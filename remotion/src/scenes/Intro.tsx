import {AbsoluteFill, OffthreadVideo, staticFile, useCurrentFrame, useVideoConfig, interpolate} from 'remotion';

export const Intro: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const o = interpolate(frame, [0, fps * 0.7, fps * 2.5, fps * 3], [0, 1, 1, 0.6], {
    extrapolateRight: 'clamp',
  });
  return (
    <AbsoluteFill style={{backgroundColor: '#04070b'}}>
      <AbsoluteFill style={{opacity: 0.55}}>
        <OffthreadVideo src={staticFile('source.mp4')} muted />
      </AbsoluteFill>
      <AbsoluteFill
        style={{
          alignItems: 'center',
          justifyContent: 'center',
          flexDirection: 'column',
          color: '#eaf1f8',
          fontFamily: 'Inter, system-ui, sans-serif',
          opacity: o,
          textShadow: '0 4px 24px rgba(0,0,0,0.75)',
        }}
      >
        <div style={{fontSize: 64, fontWeight: 700, letterSpacing: -1}}>
          Quantifying sea-wave dynamics from video
        </div>
        <div style={{fontSize: 26, marginTop: 18, opacity: 0.85, maxWidth: 1200, textAlign: 'center'}}>
          Six families of methods, each producing a 1-D signal that couples with
          EEG / HRV / EMG.
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};
