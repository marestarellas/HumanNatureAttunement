import {AbsoluteFill, OffthreadVideo, staticFile} from 'remotion';
import {MethodTitle} from '../components/MethodTitle';
import {Caption} from '../components/Caption';
import {SignalBar} from '../components/SignalBar';

export const FrameDiff: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: '#0a0e14'}}>
      <MethodTitle
        title="Frame difference / motion trail"
        subtitle="|I(t) − I(t−1)| accumulated → where motion lives"
      />
      <div style={{position: 'absolute', top: 170, left: 60, width: 880, height: 660,
                   borderRadius: 14, overflow: 'hidden',
                   boxShadow: '0 8px 32px rgba(0,0,0,0.6)'}}>
        <OffthreadVideo src={staticFile('source.mp4')} muted />
      </div>
      <div style={{position: 'absolute', top: 170, left: 980, width: 880, height: 660,
                   borderRadius: 14, overflow: 'hidden',
                   boxShadow: '0 8px 32px rgba(0,0,0,0.6)'}}>
        {/* Exponential motion-energy accumulator — much more intuitive than the
            raw absdiff: foamy crests visibly persist as bright trails. */}
        <OffthreadVideo src={staticFile('motion_trail.mp4')} muted />
      </div>
      <div style={{position: 'absolute', bottom: 130, left: 60, right: 60,
                   display: 'flex', gap: 30, alignItems: 'flex-end'}}>
        <SignalBar signalKey="frame_diff" width={1600} height={70}
                   color="#ff8fb1" vertical={false}
                   label="motion energy (frame_diff)" />
        <SignalBar signalKey="frame_diff" width={130} height={70}
                   color="#ffd166" vertical={false} />
      </div>
      <Caption>
        Subtracting consecutive frames isolates pixels that <i>changed</i>;
        accumulating with a leaky integrator turns it into a heat-map of where
        the wave is breaking right now.
      </Caption>
    </AbsoluteFill>
  );
};
