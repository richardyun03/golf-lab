import { useRef, useEffect, useState, useCallback } from "react";
import { Play, Pause, SkipBack, SkipForward } from "lucide-react";
import type { SwingPhase } from "../lib/api";
import { PHASE_ORDER, PHASE_LABELS } from "../lib/api";

interface Props {
  videoUrl: string;
  fps: number;
  phases: Record<SwingPhase, number>;
  frameCount: number;
  activePhase: SwingPhase | null;
  onPhaseSelect: (phase: SwingPhase) => void;
}

export default function VideoPlayer({
  videoUrl,
  fps,
  phases,
  frameCount,
  activePhase,
  onPhaseSelect,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(0);

  const seekToFrame = useCallback(
    (frame: number) => {
      const video = videoRef.current;
      if (!video || !fps) return;
      video.currentTime = frame / fps;
      setCurrentFrame(frame);
    },
    [fps]
  );

  // Seek to active phase when it changes
  useEffect(() => {
    if (activePhase && phases[activePhase] !== undefined) {
      seekToFrame(phases[activePhase]);
    }
  }, [activePhase, phases, seekToFrame]);

  // Track current frame during playback
  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;
    const onTime = () => {
      setCurrentFrame(Math.round(video.currentTime * fps));
    };
    video.addEventListener("timeupdate", onTime);
    return () => video.removeEventListener("timeupdate", onTime);
  }, [fps]);

  const togglePlay = () => {
    const video = videoRef.current;
    if (!video) return;
    if (video.paused) {
      video.play();
      setPlaying(true);
    } else {
      video.pause();
      setPlaying(false);
    }
  };

  const stepFrame = (delta: number) => {
    seekToFrame(Math.max(0, Math.min(frameCount - 1, currentFrame + delta)));
  };

  // Find which phase the current frame is in
  const currentPhase = [...PHASE_ORDER]
    .reverse()
    .find((p) => currentFrame >= phases[p]);

  return (
    <div>
      <div className="relative rounded-xl overflow-hidden bg-black">
        <video
          ref={videoRef}
          src={videoUrl}
          className="w-full"
          onEnded={() => setPlaying(false)}
          playsInline
        />
        {/* Phase badge overlay */}
        {currentPhase && (
          <div className="absolute top-3 left-3 bg-black/60 backdrop-blur-sm text-white text-xs font-medium px-3 py-1.5 rounded-full">
            {PHASE_LABELS[currentPhase]}
          </div>
        )}
        {/* Frame counter */}
        <div className="absolute top-3 right-3 bg-black/60 backdrop-blur-sm text-white text-xs font-mono px-3 py-1.5 rounded-full">
          {currentFrame} / {frameCount}
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center justify-center gap-2 mt-3">
        <button
          onClick={() => stepFrame(-1)}
          className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
          title="Previous frame"
        >
          <SkipBack size={18} />
        </button>
        <button
          onClick={togglePlay}
          className="p-3 rounded-full bg-emerald-600 hover:bg-emerald-500 text-white transition-colors"
        >
          {playing ? <Pause size={20} /> : <Play size={20} className="ml-0.5" />}
        </button>
        <button
          onClick={() => stepFrame(1)}
          className="p-2 rounded-lg text-gray-400 hover:text-white hover:bg-gray-800 transition-colors"
          title="Next frame"
        >
          <SkipForward size={18} />
        </button>
      </div>

      {/* Phase quick-nav */}
      <div className="flex gap-1 mt-3 overflow-x-auto">
        {PHASE_ORDER.map((phase) => (
          <button
            key={phase}
            onClick={() => onPhaseSelect(phase)}
            className={`shrink-0 text-xs px-3 py-1.5 rounded-full transition-colors ${
              activePhase === phase
                ? "bg-emerald-600 text-white"
                : "bg-gray-800 text-gray-400 hover:text-white"
            }`}
          >
            {PHASE_LABELS[phase]}
          </button>
        ))}
      </div>
    </div>
  );
}
