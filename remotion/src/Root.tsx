import {Composition} from 'remotion';
import {Showcase, SHOWCASE_DURATION_FRAMES, FPS} from './Showcase';

export const Root: React.FC = () => {
  return (
    <>
      <Composition
        id="Showcase"
        component={Showcase}
        durationInFrames={SHOWCASE_DURATION_FRAMES}
        fps={FPS}
        width={1920}
        height={1080}
      />
    </>
  );
};
