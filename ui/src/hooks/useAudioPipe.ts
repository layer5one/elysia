import { useEffect, useRef, useState } from "react";
import type { AvatarEvent } from "../types";

export function useAudioPipe(wsUrl = "ws://localhost:8765") {
  const ctxRef = useRef<AudioContext | null>(null);
  const nodeRef = useRef<AudioWorkletNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const [state, setState] = useState<'idle'|'listening'|'thinking'|'speaking'|'error'>('idle');
  const [intensity, setIntensity] = useState(0);

  useEffect(() => {
    let ws: WebSocket;
    (async () => {
      const ctx = new AudioContext({ sampleRate: 24000 });
      ctxRef.current = ctx;
      await ctx.audioWorklet.addModule('/audio/worklet-processor.js');
      const node = new AudioWorkletNode(ctx, 'pcm-bridge', { numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1] });
      nodeRef.current = node;
      const analyser = ctx.createAnalyser(); analyser.fftSize = 2048;
      node.connect(analyser); analyser.connect(ctx.destination);
      analyserRef.current = analyser;

      ws = new WebSocket(wsUrl);
      ws.onmessage = ev => {
        const msg: AvatarEvent = JSON.parse(ev.data);
        if (msg.type === 'state') setState(msg.value);
        if (msg.type === 'tts_begin') setState('speaking');
        if (msg.type === 'tts_end') setState('idle');
        if (msg.type === 'tts_chunk') {
          const b = atob(msg.pcm);
          const buf = new Float32Array(b.length / 4);
          const dv = new DataView(new ArrayBuffer(b.length));
          for (let i = 0; i < b.length; i++) dv.setUint8(i, b.charCodeAt(i));
          const f32 = new Float32Array(dv.buffer);
          node.port.postMessage({ type: 'push', buffer: f32 });
        }
      };

      // RMS + sibilant band (2–6kHz)
      const data = new Float32Array(analyser.frequencyBinCount);
      const sample = () => {
        analyser.getFloatTimeDomainData(data);
        let sum = 0; for (let i=0;i<data.length;i++) sum += data[i]*data[i];
        const rms = Math.sqrt(sum / data.length);
        setIntensity(Math.min(1, rms * 8)); // scale

        requestAnimationFrame(sample);
      };
      sample();
    })();
    return () => { ws?.close(); ctxRef.current?.close(); }
  }, [wsUrl]);

  return { state, intensity, analyser: analyserRef.current };
}
