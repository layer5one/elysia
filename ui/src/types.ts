export type AvatarEvent =
  | {type:'state', value:'idle'|'listening'|'thinking'|'speaking'|'error'}
  | {type:'emotion', value:'neutral'|'mischief'|'annoyed'|'warm'|'deadpan'}
  | {type:'intensity', value:number}
  | {type:'tts_begin', id:string, sr:number}
  | {type:'tts_chunk', id:string, ts:number, pcm:string} // base64 Float32
  | {type:'tts_end', id:string}
  | {type:'viseme', at:number, id:string, strength:number};
