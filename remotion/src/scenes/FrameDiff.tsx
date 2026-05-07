import {AbsoluteFill, OffthreadVideo, staticFile} from 'remotion';
import {MethodTitle} from '../components/MethodTitle';
import {Caption} from '../components/Caption';
import {CumulativeLine} from '../components/CumulativeLine';

export const FrameDiff: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: '#0a0e14'}}>
      <MethodTitle
        title="Frame difference / motion trail"
        subtitle="|I(t) − I(t−1)| and its time integral"
      />
      <div style={{position: 'absolute', top: 170, left: 60, width: 880, height: 660,
                   borderRadius: 14, overflow: 'hidden',
                   boxShadow: '0 8px 32px rgba(0,0,0,0.6)'}}>
        <OffthreadVideo src={staticFile('source.mp4')} muted
                        style={{width: '100%', height: '100%', objectFit: 'cover'}} />
      </div>
      <div style={{position: 'absolute', top: 170, left: 980, width: 880, height: 660,
                   borderRadius: 14, overflow: 'hidden',
                   boxShadow: '0 8px 32px rgba(0,0,0,0.6)'}}>
        {/* Exponential motion-energy accumulator -- foamy crests persist as bright trails. */}
        <OffthreadVideo src={staticFile('motion_trail.mp4')} muted
                        style={{width: '100%', height: '100%', objectFit: 'cover'}} />
      </div>
      {/* Bottom: instantaneous motion energy + its cumulative integral overlay */}
      <div style={{position: 'absolute', bottom: 150, left: 60, right: 60}}>
        <CumulativeLine signalKey="frame_diff" width={1800} height={170}
                        color="#ff8fb1" cumColor="#ffd166"
                        label="instantaneous motion energy + Σ|motion| (orange fill)" />
      </div>
      <Caption>
        Subtracting consecutive frames isolates pixels that <i>changed</i>;
        the orange fill is the running integral — exactly the quantity the
        right-hand motion-trail panel renders as a glowing afterimage.
      </Caption>
    </AbsoluteFill>
  );
};
