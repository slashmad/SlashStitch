import {spawn} from 'node:child_process';
import {createInterface} from 'node:readline';
import {once} from 'node:events';
import {resolve} from 'node:path';
import {deflateSync} from 'node:zlib';
export function png(width=160,height=120,color=[180,90,40]) {
 const crc=(data)=>{let c=0xffffffff;for(const b of data){c^=b;for(let i=0;i<8;i++)c=(c>>>1)^((c&1)?0xedb88320:0);}return(c^0xffffffff)>>>0;};
 const chunk=(kind,data)=>{const type=Buffer.from(kind);const out=Buffer.alloc(data.length+12);out.writeUInt32BE(data.length);type.copy(out,4);data.copy(out,8);out.writeUInt32BE(crc(Buffer.concat([type,data])),out.length-4);return out;};
 const header=Buffer.alloc(13);header.writeUInt32BE(width);header.writeUInt32BE(height,4);header[8]=8;header[9]=2;
 const raw=Buffer.alloc((width*3+1)*height);for(let y=0;y<height;y++)for(let x=0;x<width;x++)for(let c=0;c<3;c++)raw[y*(width*3+1)+1+x*3+c]=color[c];
 return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),chunk('IHDR',header),chunk('IDAT',deflateSync(raw)),chunk('IEND',Buffer.alloc(0))]);
}
export class WorkerClient {
 constructor(directory,extraEnv={}){
  this.pending=new Map();this.events=[];this.state='idle';this.serial=0;this.listeners=new Set();this.stderr='';
  this.proc=spawn(extraEnv.SLASHSTITCH_TEST_WORKER||process.env.SLASHSTITCH_TEST_WORKER||resolve('engine/target/debug',process.platform==='win32'?'slashstitch-worker.exe':'slashstitch-worker'),[],{env:{...process.env,TMPDIR:directory,SLASHSTITCH_PREVIEW_ROOT:directory,SLASHSTITCH_FORCE_CPU:'1',RAYON_NUM_THREADS:'2',...extraEnv},stdio:'pipe'});
  this.proc.stderr.on('data',b=>{this.stderr+=b.toString();});
  this.lines=createInterface({input:this.proc.stdout});
  this.lines.on('line',line=>{const msg=JSON.parse(line);if(msg.event){this.events.push(msg);if(msg.event==='job-state')this.state=msg.data.state;for(const listener of this.listeners)listener(msg);}else{this.pending.get(msg.id)?.(msg);this.pending.delete(msg.id);}});
  this.proc.on('close',code=>{for(const resolve of this.pending.values())resolve({ok:false,error:`Worker exited ${code}: ${this.stderr}`});this.pending.clear();});
  this.proc.on('error',error=>{for(const resolve of this.pending.values())resolve({ok:false,error:error.message});this.pending.clear();});
 }
 async call(action,payload=null){
  const id=String(++this.serial);
  const response=await new Promise((resolve,reject)=>{
   const timer=setTimeout(()=>{this.pending.delete(id);reject(Error(`Timed out: ${action}; ${this.stderr}`));},30000);
   this.pending.set(id,msg=>{clearTimeout(timer);resolve(msg);});
   this.proc.stdin.write(JSON.stringify({id,action,payload})+'\n',error=>{if(error){clearTimeout(timer);this.pending.delete(id);reject(error);}});
  });
  return response;
 }
 async success(action,payload=null){const reply=await this.call(action,payload);if(!reply.ok)throw Error(reply.error);await this.idle();return reply.result;}
 wait(check,timeout=30000){
  const current=this.events.find(check);if(current)return Promise.resolve(current);
  return new Promise((resolve,reject)=>{const listener=msg=>{if(check(msg)){clearTimeout(timer);this.listeners.delete(listener);resolve(msg);}};const timer=setTimeout(()=>{this.listeners.delete(listener);reject(Error(`Timed out waiting for event; ${this.stderr}`));},timeout);this.listeners.add(listener);});
 }
 idle(){if(this.state==='idle')return Promise.resolve();return new Promise((resolve,reject)=>{const listener=msg=>{if(msg.event==='job-state'&&msg.data.state==='idle'){clearTimeout(timer);this.listeners.delete(listener);resolve();}};const timer=setTimeout(()=>{this.listeners.delete(listener);reject(Error(`Worker did not become idle; ${this.stderr}`));},30000);this.listeners.add(listener);});}
 async close(){
  if(this.proc.exitCode!==null)return;
  try{await this.call('job-cancel');await this.idle();const exited=once(this.proc,'close');await this.call('shutdown');await exited;}
  finally{this.lines.close();if(this.proc.exitCode===null)this.proc.kill();}
 }
}
