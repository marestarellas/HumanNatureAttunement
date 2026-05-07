import {AbsoluteFill, OffthreadVideo, staticFile} from 'remotion';
import {MethodTitle} from '../components/MethodTitle';
import {Caption} from '../components/Caption';
import {ScrollingLine} from '../components/ScrollingLine';

export const Outro: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: '#0a0e14'}}>
      <MethodTitle
        title="→ HNA.modules.coupling"
        subtitle="windowed cross-correlation · coherence · PLV · wPLI · mutual information"
      />
      <div style={{position: 'absolute', top: 180, left: 60, width: 480, height: 480,
                   borderRadius: 14, overflow: 'hidden',
                   boxShadow: '0 8px 32px rgba(0,0,0,0.6)'}}>
        <OffthreadVideo src={staticFile('source.mp4')} muted style={{width: "100%", height: "100%", objectFit: "cover"}} />
      </div>
      <div style={{position: 'absolute', top: 180, left: 580, right: 60,
                   display: 'flex', flexDirection: 'column', gap: 8}}>
        {[
          ['luminance',                            '#ffd166'],
          ['flow_mag_mean',                        '#7adfff'],
          ['fractal_dim',                          '#a0e7a0'],
          ['modal_1',                              '#ff8fb1'],
          ['wc_flow_mag_mean__perm_entropy',       '#c8a2ff'],
          ['wc_patch_entropy__hjorth_complexity',  '#ffa07a'],
        ].map(([k, c]) => (
          <ScrollingLine
            key={k}
            signalKey={k as string}
            windowSec={6}
            width={1280}
            height={70}
            color={c as string}
            label={k as string}
          />
        ))}
      </div>
      <Caption>
        Each video-derived signal is a drop-in input for the existing coupling
        pipeline — the same statistics that already link sea-wave audio with
        EEG now extend symmetrically to the visual side.
      </Caption>
    </AbsoluteFill>
  );
};
