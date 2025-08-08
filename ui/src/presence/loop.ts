import { useEffect, useRef } from "react";

export default function FFTWave({ analyser }: { analyser: AnalyserNode | null }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (!analyser || !canvasRef.current) return;
    const ctx = canvasRef.current.getContext('2d')!;
    const data = new Uint8Array(analyser.frequencyBinCount);
    const draw = () => {
      analyser.getByteFrequencyData(data);
      ctx.clearRect(0,0,canvasRef.current!.width,canvasRef.current!.height);
      for (let i=0;i<data.length;i++){
        const x = (i / data.length) * canvasRef.current!.width;
        const h = (data[i] / 255) * canvasRef.current!.height;
        ctx.fillRect(x, canvasRef.current!.height - h, 2, h);
      }
      requestAnimationFrame(draw);
    };
    draw();
  }, [analyser]);
  return <canvas ref={canvasRef} width={600} height={120} />;
}
import AvatarHost from "./components/AvatarHost";
import FFTWave from "./components/FFTWave";
import { useAudioPipe } from "./hooks/useAudioPipe";

export default function App(){
  const { analyser } = useAudioPipe();
  return (
    <div className="grid">
      <AvatarHost />
      <FFTWave analyser={analyser!} />
    </div>
  );
}
