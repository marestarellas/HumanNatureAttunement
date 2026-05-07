import {AbsoluteFill, OffthreadVideo, staticFile} from 'remotion';
import {MethodTitle} from '../components/MethodTitle';
import {Caption} from '../components/Caption';
import {ScrollingLine} from '../components/ScrollingLine';
import {SignalBar} from '../components/SignalBar';

export const SpatialComplexity: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: '#0a0e14'}}>
      <MethodTitle
        title="Spatial complexity"
        subtitle="patch entropy · edge density · radial 2D-FFT slope · box-counting fractal dim"
      />
      {/* Three views of the same frame, side by side: raw, edges, patch entropy
          heatmap. Lets the viewer see WHAT each measure is reading. */}
      <div style={{position: 'absolute', top: 165, left: 30, width: 600, height: 600,
                   borderRadius: 12, overflow: 'hidden'}}>
        <OffthreadVideo src={staticFile('source.mp4')} muted />
        <div style={{position: 'absolute', top: 12, left: 12, color: '#cfd5dc',
                     fontFamily: 'ui-monospace, monospace', fontSize: 16,
                     background: 'rgba(0,0,0,0.5)', padding: '4px 10px', borderRadius: 6}}>
          source
        </div>
      </div>
      <div style={{position: 'absolute', top: 165, left: 660, width: 600, height: 600,
                   borderRadius: 12, overflow: 'hidden'}}>
        <OffthreadVideo src={staticFile('edges.mp4')} muted />
        <div style={{position: 'absolute', top: 12, left: 12, color: '#cfd5dc',
                     fontFamily: 'ui-monospace, monospace', fontSize: 16,
                     background: 'rgba(0,0,0,0.5)', padding: '4px 10px', borderRadius: 6}}>
          Canny edges → edge_density · fractal_dim
        </div>
      </div>
      <div style={{position: 'absolute', top: 165, left: 1290, width: 600, height: 600,
                   borderRadius: 12, overflow: 'hidden'}}>
        <OffthreadVideo src={staticFile('patch_heatmap.mp4')} muted />
        <div style={{position: 'absolute', top: 12, left: 12, color: '#cfd5dc',
                     fontFamily: 'ui-monospace, monospace', fontSize: 16,
                     background: 'rgba(0,0,0,0.5)', padding: '4px 10px', borderRadius: 6}}>
          per-patch entropy heatmap
        </div>
      </div>

      {/* Live values as horizontal bars (more intuitive than scrolling lines) */}
      <div style={{position: 'absolute', bottom: 170, left: 60, right: 60,
                   display: 'flex', gap: 18, alignItems: 'flex-end'}}>
        <SignalBar signalKey="edge_density"      width={420} height={60}
                   color="#ffd166" vertical={false} label="edge_density" />
        <SignalBar signalKey="fractal_dim"       width={420} height={60}
                   color="#a0e7a0" vertical={false} label="fractal_dim" />
        <SignalBar signalKey="patch_entropy"     width={420} height={60}
                   color="#ff8fb1" vertical={false} label="patch_entropy" />
        <SignalBar signalKey="spatial_psd_slope" width={420} height={60}
                   color="#7adfff" vertical={false} label="spatial_psd_slope" />
      </div>
      <div style={{position: 'absolute', bottom: 110, left: 60, right: 60}}>
        <ScrollingLine signalKey="fractal_dim" windowSec={5} width={1800} height={50}
                       color="#a0e7a0" label="fractal_dim(t)" />
      </div>

      <Caption>
        Each spatial-complexity measure reads a different aspect of the frame —
        edge density counts foam, fractal dim measures the foam's roughness,
        per-patch entropy lights up high-information regions like a heatmap.
      </Caption>
    </AbsoluteFill>
  );
};
