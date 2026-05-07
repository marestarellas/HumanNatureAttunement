import {AbsoluteFill} from 'remotion';
import {MethodTitle} from '../components/MethodTitle';
import {Caption} from '../components/Caption';
import {ScrollingLine} from '../components/ScrollingLine';
import {SignalBar} from '../components/SignalBar';
import {PulseFrame} from '../components/PulseFrame';

export const GlobalEnvelope: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: '#0a0e14'}}>
      <MethodTitle
        title="Global temporal envelope"
        subtitle="every pixel collapses to one number per frame · luminance · channel means · frame-diff"
      />
      {/* The video itself "breathes" with luminance — outer glow pulsing */}
      <div style={{position: 'absolute', top: 170, left: 80, width: 720, height: 720}}>
        <PulseFrame src="source.mp4" width={720} height={720}
                    signalKey="luminance" borderColor="#ffd166" />
      </div>

      {/* Vertical bars: R, G, B, luminance — VU-meter style */}
      <div style={{position: 'absolute', top: 200, left: 850,
                   display: 'flex', gap: 26, alignItems: 'flex-end'}}>
        <SignalBar signalKey="red_mean"   width={70} height={460}
                   color="#ff6b6b" label="R" />
        <SignalBar signalKey="green_mean" width={70} height={460}
                   color="#a0e7a0" label="G" />
        <SignalBar signalKey="blue_mean"  width={70} height={460}
                   color="#7adfff" label="B" />
        <div style={{width: 18}} />
        <SignalBar signalKey="luminance"  width={90} height={460}
                   color="#ffd166" label="luminance" />
        <SignalBar signalKey="frame_diff" width={90} height={460}
                   color="#ff8fb1" label="frame_diff" />
      </div>

      {/* One scrolling line for context */}
      <div style={{position: 'absolute', top: 720, left: 850, right: 60}}>
        <ScrollingLine signalKey="luminance" windowSec={6}
                       width={1010} height={140}
                       color="#ffd166" label="luminance(t)" />
      </div>

      <Caption>
        Each frame collapses to one number — the simplest "video envelope."
        It's the visual equivalent of the audio envelope already used to
        entrain EEG, so the same coupling pipeline applies symmetrically.
      </Caption>
    </AbsoluteFill>
  );
};
