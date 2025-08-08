import { useEffect, useRef } from "react";
import { useAudioPipe } from "../hooks/useAudioPipe";

export default function AvatarHost() {
  const { state, intensity } = useAudioPipe();
  const ref = useRef<HTMLDivElement>(null);

  // pseudo: wire to Rive inputs (jawOpen, glow)
  useEffect(() => {
    // if using @rive-app/canvas, grab state machine inputs and set:
    // jawOpen.value = intensity
    // glow.value = intensity
    // and switch a 'mode' input by state
  }, [state, intensity]);

  return (
    <div className="avatar">
      {/* Rive canvas here */}
      <div className="overlay">
        <span className={`badge ${state}`}>{state}</span>
      </div>
    </div>
  );
}
