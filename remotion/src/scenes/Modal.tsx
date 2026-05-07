import {AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {MethodTitle} from '../components/MethodTitle';
import {Caption} from '../components/Caption';
import {SignalBar} from '../components/SignalBar';
import {sampleSignalAt, SIGNALS_META} from '../components/useSignal';
import summary from '../data/summary.json';

const COLORS = ['#ffd166', '#7adfff', '#a0e7a0', '#ff8fb1'];

/**
 * Visual: 4 modes laid out left→right. Each mode tile pulses (scale + glow)
 * with its own temporal coefficient (modal_k), so the eye sees that *modes
 * are oscillators*. Below: an explicit "video ≈ a₁·m₁ + a₂·m₂ + a₃·m₃ + a₄·m₄"
 * equation where each aₖ box also pulses with the matching coefficient.
 */
export const Modal: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const dur = SIGNALS_META.duration_s;
  const tSource = ((frame / fps) % dur + dur) % dur;
  const freqs: number[] = (summary as any).modal?.frequencies_hz ?? [];
  const energies: number[] = (summary as any).modal?.energies ?? [];

  const coeffs = [1, 2, 3, 4].map((k) => sampleSignalAt(`modal_${k}`, tSource));
  // Normalize for display by per-mode running absolute max in the JSON
  const norms = [1, 2, 3, 4].map((k) => {
    const arr = (SIGNALS_META.signals as Record<string, number[]>)[`modal_${k}`] || [];
    const m = arr.reduce((acc, v) => Math.max(acc, Math.abs(v) || 0), 1e-6);
    return coeffs[k - 1] / m;
  });

  return (
    <AbsoluteFill style={{backgroundColor: '#0a0e14'}}>
      <MethodTitle
        title="Spatio-temporal modal decomposition"
        subtitle="video ≈ Σₖ aₖ(t) · modeₖ — each mode is an oscillator"
      />
      <div style={{position: 'absolute', top: 200, left: 60, right: 60,
                   display: 'flex', gap: 20, justifyContent: 'space-between'}}>
        {[1, 2, 3, 4].map((k) => {
          const a = norms[k - 1];
          const scale = 1 + 0.06 * a;
          const glow = 12 + 60 * Math.abs(a);
          return (
            <div key={k} style={{flex: 1, textAlign: 'center'}}>
              <div style={{display: 'flex', justifyContent: 'center'}}>
                <Img src={staticFile(`mode_${k}.png`)}
                     style={{width: 380, height: 380, borderRadius: 14,
                             objectFit: 'cover',
                             transform: `scale(${scale})`,
                             boxShadow: `0 0 ${glow}px ${COLORS[k - 1]}`,
                             outline: `2px solid ${COLORS[k - 1]}`,
                             transformOrigin: 'center'}} />
              </div>
              <div style={{color: COLORS[k - 1], fontFamily: 'ui-monospace, monospace',
                           fontSize: 20, marginTop: 14, fontWeight: 700}}>
                mode {k}
              </div>
              <div style={{color: '#cfd5dc', fontFamily: 'ui-monospace, monospace',
                           fontSize: 16, marginTop: 4}}>
                {freqs[k - 1] != null ? `${freqs[k - 1].toFixed(3)} Hz` : '—'}
                {energies[k - 1] != null && (
                  <span style={{color: '#7c8590', marginLeft: 10}}>
                    E={energies[k - 1].toFixed(2)}
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* The reconstruction equation — each coefficient is a live bar */}
      <div style={{position: 'absolute', bottom: 130, left: 60, right: 60,
                   display: 'flex', alignItems: 'flex-end',
                   justifyContent: 'space-between',
                   color: '#cfd5dc', fontFamily: 'ui-monospace, monospace',
                   fontSize: 30}}>
        <span>video(t) ≈</span>
        {[1, 2, 3, 4].map((k, i) => (
          <div key={k} style={{display: 'flex', flexDirection: 'column',
                               alignItems: 'center', gap: 4}}>
            <SignalBar signalKey={`modal_${k}`} width={300} height={50}
                       color={COLORS[k - 1]} vertical={false}
                       windowSec={dur}
                       label={`a${k}(t)`} />
            <span style={{color: COLORS[k - 1], fontSize: 22, marginTop: 16}}>
              · mode{k} {i < 3 ? '+' : ''}
            </span>
          </div>
        ))}
      </div>

      <Caption>
        SVD/DMD factorizes the video into spatial patterns (modes) × temporal
        coefficients (oscillators). The brightest pulsing tile is the dominant
        wave; its frequency is{' '}
        {freqs[0] != null ? `${freqs[0].toFixed(3)} Hz` : '—'}.
      </Caption>
    </AbsoluteFill>
  );
};
