import {AbsoluteFill, OffthreadVideo, staticFile} from 'remotion';
import {MethodTitle} from '../components/MethodTitle';
import {Caption} from '../components/Caption';
import {ScrollingLine} from '../components/ScrollingLine';
import {RoseDial} from '../components/RoseDial';
import {ParticleField} from '../components/ParticleField';

export const OpticalFlow: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: '#0a0e14'}}>
      <MethodTitle
        title="Optical flow (Farneback)"
        subtitle="grid of arrows · direction wheel · particles drift with the field"
      />
      {/* Source with a particle field that drifts in the flow direction —
          gives the audience an immediate "this is the wave moving" feeling. */}
      <div style={{position: 'absolute', top: 170, left: 60, width: 720, height: 720,
                   borderRadius: 14, overflow: 'hidden'}}>
        <OffthreadVideo src={staticFile('source.mp4')} muted />
        <ParticleField width={720} height={720} count={120} color="#ffd166" />
      </div>

      {/* The actual flow estimate, drawn as discrete arrows on a grid */}
      <div style={{position: 'absolute', top: 170, left: 800, width: 720, height: 720,
                   borderRadius: 14, overflow: 'hidden'}}>
        <OffthreadVideo src={staticFile('flow_arrows.mp4')} muted />
      </div>

      {/* Wind-rose dial showing the instantaneous mean direction */}
      <div style={{position: 'absolute', top: 220, left: 1560, width: 320, height: 320}}>
        <RoseDial uKey="flow_u_mean" vKey="flow_v_mean"
                  size={300} label="mean (u,v)" />
      </div>

      {/* Magnitude + curl as horizontal bars below */}
      <div style={{position: 'absolute', bottom: 160, left: 60, right: 60,
                   display: 'flex', flexDirection: 'column', gap: 10}}>
        <ScrollingLine signalKey="flow_mag_mean"     windowSec={5} width={1800} height={100}
                       color="#ffd166" label="flow_mag_mean (bulk wave motion)" />
        <ScrollingLine signalKey="flow_curl_abs_mean" windowSec={5} width={1800} height={100}
                       color="#7adfff" label="flow_curl_abs_mean (rotational/turbulent)" />
      </div>

      <Caption>
        Dense Farneback flow → arrows on a grid. Hue = direction, length = speed.
        The dial collapses everything to one mean (u,v); bulk magnitude and
        |curl| pulse at the swell period.
      </Caption>
    </AbsoluteFill>
  );
};
