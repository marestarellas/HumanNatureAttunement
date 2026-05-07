import {AbsoluteFill, Sequence} from 'remotion';
import {Intro} from './scenes/Intro';
import {GlobalEnvelope} from './scenes/GlobalEnvelope';
import {FrameDiff} from './scenes/FrameDiff';
import {OpticalFlow} from './scenes/OpticalFlow';
import {SpatialComplexity} from './scenes/SpatialComplexity';
import {Modal} from './scenes/Modal';
import {Timestack} from './scenes/Timestack';
import {ComplexityWindowed} from './scenes/ComplexityWindowed';
import {Outro} from './scenes/Outro';

export const FPS = 30;

const sec = (s: number) => Math.round(s * FPS);

// Scene durations (seconds) — match remotion/STORYBOARD.md
const D = {
  intro: 3,
  envelope: 8,
  framediff: 5,
  flow: 8,
  spatial: 8,
  modal: 8,
  timestack: 8,
  complexity: 10,
  outro: 10,
} as const;

const starts = {
  intro: 0,
  envelope: D.intro,
  framediff: D.intro + D.envelope,
  flow: D.intro + D.envelope + D.framediff,
  spatial: D.intro + D.envelope + D.framediff + D.flow,
  modal: D.intro + D.envelope + D.framediff + D.flow + D.spatial,
  timestack: D.intro + D.envelope + D.framediff + D.flow + D.spatial + D.modal,
  complexity:
    D.intro + D.envelope + D.framediff + D.flow + D.spatial + D.modal + D.timestack,
  outro:
    D.intro +
    D.envelope +
    D.framediff +
    D.flow +
    D.spatial +
    D.modal +
    D.timestack +
    D.complexity,
};

export const SHOWCASE_DURATION_FRAMES = sec(
  starts.outro + D.outro
);

export const Showcase: React.FC = () => {
  return (
    <AbsoluteFill style={{backgroundColor: '#0a0e14'}}>
      <Sequence from={sec(starts.intro)} durationInFrames={sec(D.intro)}>
        <Intro />
      </Sequence>
      <Sequence from={sec(starts.envelope)} durationInFrames={sec(D.envelope)}>
        <GlobalEnvelope />
      </Sequence>
      <Sequence from={sec(starts.framediff)} durationInFrames={sec(D.framediff)}>
        <FrameDiff />
      </Sequence>
      <Sequence from={sec(starts.flow)} durationInFrames={sec(D.flow)}>
        <OpticalFlow />
      </Sequence>
      <Sequence from={sec(starts.spatial)} durationInFrames={sec(D.spatial)}>
        <SpatialComplexity />
      </Sequence>
      <Sequence from={sec(starts.modal)} durationInFrames={sec(D.modal)}>
        <Modal />
      </Sequence>
      <Sequence from={sec(starts.timestack)} durationInFrames={sec(D.timestack)}>
        <Timestack />
      </Sequence>
      <Sequence from={sec(starts.complexity)} durationInFrames={sec(D.complexity)}>
        <ComplexityWindowed />
      </Sequence>
      <Sequence from={sec(starts.outro)} durationInFrames={sec(D.outro)}>
        <Outro />
      </Sequence>
    </AbsoluteFill>
  );
};
