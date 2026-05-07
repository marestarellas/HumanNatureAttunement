import {AbsoluteFill, OffthreadVideo, staticFile, useCurrentFrame, useVideoConfig, interpolate} from 'remotion';
import {MethodTitle} from '../components/MethodTitle';
import {Caption} from '../components/Caption';
import summary from '../data/summary.json';

/**
 * The "row of pixels accumulates into a strip" idea, made visible:
 * - Left:  source clip with a vertical red line marking the sampled column.
 * - Mid:   `column_sweep.mp4`, which has been pre-rendered to grow column-by-column,
 *          revealing the timestack image as time passes.
 * - Right: a 1-D PSD panel that slides in towards the end of the scene.
 */
export const Timestack: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps, durationInFrames} = useVideoConfig();
  const total = durationInFrames / fps;
  const t = frame / fps;
  const psdReveal = interpolate(t, [total * 0.55, total * 0.85], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });
  const peakHz: number | null = (summary as any).timestack?.dominant_freq_hz ?? null;
  const peakSec: number | null = (summary as any).timestack?.dominant_period_s ?? null;

  return (
    <AbsoluteFill style={{backgroundColor: '#0a0e14'}}>
      <MethodTitle
        title="Timestack (oceanographic)"
        subtitle="one column · sampled every frame · 1-D FFT → wave period"
      />
      <div style={{position: 'absolute', top: 170, left: 60, width: 540, height: 540,
                   borderRadius: 14, overflow: 'hidden'}}>
        <OffthreadVideo src={staticFile('source.mp4')} muted />
        <div
          style={{
            position: 'absolute',
            top: 0, bottom: 0, left: '50%',
            width: 4,
            background: '#ff5577',
            boxShadow: '0 0 20px #ff5577',
          }}
        />
        <div style={{position: 'absolute', bottom: 12, left: 12, right: 12, color: '#cfd5dc',
                     fontFamily: 'ui-monospace, monospace', fontSize: 16,
                     background: 'rgba(0,0,0,0.55)', padding: '6px 10px', borderRadius: 6}}>
          column being sampled →
        </div>
      </div>

      <div style={{position: 'absolute', top: 170, left: 620, width: 540, height: 540,
                   borderRadius: 14, overflow: 'hidden', border: '1px solid #1c2530'}}>
        <OffthreadVideo src={staticFile('column_sweep.mp4')} muted />
        <div style={{position: 'absolute', bottom: 12, left: 12, right: 12, color: '#cfd5dc',
                     fontFamily: 'ui-monospace, monospace', fontSize: 16,
                     background: 'rgba(0,0,0,0.55)', padding: '6px 10px', borderRadius: 6}}>
          x-t image accumulates →
        </div>
      </div>

      {/* Spectrum panel slides in from the right */}
      <div
        style={{
          position: 'absolute',
          top: 170, left: 1180, width: 700, height: 540,
          opacity: psdReveal,
          transform: `translateX(${(1 - psdReveal) * 60}px)`,
          background: 'rgba(255,255,255,0.04)',
          border: '1px solid #1c2530',
          borderRadius: 14,
          color: '#dbe3ec',
          padding: 24,
          fontFamily: 'ui-monospace, monospace',
        }}
      >
        <div style={{fontSize: 20, color: '#cfd5dc', marginBottom: 14}}>1-D FFT of the timestack column</div>
        {/* a tiny stylized PSD: peak at the wave frequency */}
        <svg width={650} height={380}>
          <line x1={40} y1={340} x2={640} y2={340}
                stroke="rgba(255,255,255,0.35)" strokeWidth={1.5} />
          <line x1={40} y1={20} x2={40} y2={340}
                stroke="rgba(255,255,255,0.35)" strokeWidth={1.5} />
          {peakHz != null && (
            <>
              {(() => {
                const fmax = Math.max(0.6, peakHz * 4);
                const peakX = 40 + (peakHz / fmax) * 600;
                return (
                  <>
                    {/* a smooth lorentzian-like bump */}
                    <path
                      d={(() => {
                        const pts: string[] = [];
                        for (let i = 0; i <= 200; i++) {
                          const f = (i / 200) * fmax;
                          const dx = (f - peakHz) / (fmax * 0.04);
                          const y = 340 - 280 * Math.exp(-dx * dx) - 8 * Math.exp(-((f - 0) ** 2) / 0.001);
                          const x = 40 + (f / fmax) * 600;
                          pts.push((i === 0 ? 'M' : 'L') + x.toFixed(1) + ',' + y.toFixed(1));
                        }
                        return pts.join(' ');
                      })()}
                      fill="none"
                      stroke="#7adfff"
                      strokeWidth={3}
                    />
                    <line x1={peakX} y1={20} x2={peakX} y2={340}
                          stroke="#ffd166" strokeWidth={2} strokeDasharray="6 4" />
                    <text x={peakX + 8} y={48} fill="#ffd166" fontSize={20}>
                      peak {peakHz.toFixed(3)} Hz
                    </text>
                    {peakSec != null && (
                      <text x={peakX + 8} y={74} fill="#cfd5dc" fontSize={18}>
                        period {peakSec.toFixed(2)} s
                      </text>
                    )}
                    <text x={40} y={365} fill="#7c8590" fontSize={14}>0</text>
                    <text x={620} y={365} fill="#7c8590" fontSize={14}>
                      {fmax.toFixed(2)} Hz
                    </text>
                  </>
                );
              })()}
            </>
          )}
        </svg>
      </div>

      <Caption>
        Sample one pixel column on every frame, stack them into an x-t image,
        then 1-D FFT the result — the dominant peak is the wave period.
        Argus-style coastal monitoring (Holman & Stanley).
      </Caption>
    </AbsoluteFill>
  );
};
