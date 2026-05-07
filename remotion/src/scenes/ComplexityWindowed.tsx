import {AbsoluteFill} from 'remotion';
import {MethodTitle} from '../components/MethodTitle';
import {Caption} from '../components/Caption';
import {ScrollingLine} from '../components/ScrollingLine';
import {OrdinalGlyph} from '../components/OrdinalGlyph';
import {SignalBar} from '../components/SignalBar';
import {SIGNALS_META} from '../components/useSignal';

const WINDOW_SEC = 2.0;

export const ComplexityWindowed: React.FC = () => {
  const sourceDur = SIGNALS_META.duration_s;
  return (
    <AbsoluteFill style={{backgroundColor: '#0a0e14'}}>
      <MethodTitle
        title="Time-resolved complexity"
        subtitle="3 samples → an ordinal pattern → entropy of how often each pattern occurs"
      />

      {/* Top: the source signal scrolling, with sliding window highlight */}
      <div style={{position: 'absolute', top: 170, left: 60, right: 60, height: 220}}>
        <ScrollingLine
          signalKey="flow_mag_mean"
          windowSec={sourceDur}
          width={1800}
          height={220}
          color="#ffd166"
          label={`flow_mag_mean — analysis window ${WINDOW_SEC.toFixed(1)} s`}
        />
        <div
          style={{
            position: 'absolute',
            top: 0,
            height: 220,
            left: '60%',                /* visually static, the data scrolls under it */
            width: `${(WINDOW_SEC / sourceDur) * 100}%`,
            background: 'rgba(255, 230, 100, 0.16)',
            border: '2px solid rgba(255, 230, 100, 0.6)',
            borderRadius: 6,
          }}
        />
      </div>

      {/* Middle: the *intuition* — three consecutive samples and their ordinal pattern */}
      <div style={{position: 'absolute', top: 420, left: 60, width: 720, height: 240}}>
        <OrdinalGlyph signalKey="flow_mag_mean"
                      width={720} height={240}
                      m={3} tau={4} />
      </div>
      <div style={{position: 'absolute', top: 430, left: 800, right: 60,
                   color: '#cfd5dc', fontFamily: 'Inter, system-ui, sans-serif',
                   fontSize: 22, lineHeight: 1.5, maxWidth: 1000}}>
        For m = 3 samples there are 3! = 6 possible orderings.
        Permutation entropy = how evenly those 6 patterns are visited inside
        the analysis window. Highly ordered signal → one pattern dominates →
        low entropy.
      </div>

      {/* Bottom: four time-resolved complexity signals + live bars */}
      <div style={{position: 'absolute', bottom: 170, left: 60, right: 60,
                   display: 'flex', gap: 24, alignItems: 'flex-end'}}>
        <SignalBar signalKey="wc_flow_mag_mean__perm_entropy"
                   width={400} height={70} vertical={false}
                   color="#7adfff" label="perm_entropy(t)" />
        <SignalBar signalKey="wc_flow_mag_mean__hjorth_complexity"
                   width={400} height={70} vertical={false}
                   color="#a0e7a0" label="hjorth_complexity(t)" />
        <SignalBar signalKey="wc_flow_mag_mean__higuchi_fd"
                   width={400} height={70} vertical={false}
                   color="#ff8fb1" label="higuchi_fd(t)" />
        <SignalBar signalKey="wc_flow_mag_mean__spectral_entropy"
                   width={400} height={70} vertical={false}
                   color="#c8a2ff" label="spectral_entropy(t)" />
      </div>
      <div style={{position: 'absolute', bottom: 100, left: 60, right: 60}}>
        <ScrollingLine signalKey="wc_flow_mag_mean__perm_entropy"
                       windowSec={sourceDur} width={1800} height={50}
                       color="#7adfff" label="perm_entropy(t)" />
      </div>

      <Caption>
        Sliding the window turns each complexity measure into its own 1-D
        signal — directly cross-correlatable with brain, heart, or muscle
        rhythms.
      </Caption>
    </AbsoluteFill>
  );
};
