// Isolated Linux UI smoke: requires weston, never uses the desktop display or capture hardware.
import {spawn} from 'node:child_process';
import {mkdtempSync, openSync, existsSync, readFileSync, rmSync, closeSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
import {png} from '../tests/helpers/worker-client.mjs';
const require=createRequire(import.meta.url);
const temporary=mkdtempSync(join(tmpdir(),'slashapp-smoke-'));
const wait=ms=>new Promise(resolve=>setTimeout(resolve,ms));
async function until(check,label,timeout=20000) {
 const deadline=Date.now()+timeout;
 while(Date.now()<deadline) {const result=await check();if(result)return result;await wait(50);}
 throw new Error(`Timed out: ${label}`);
}
const log=openSync(join(temporary,'process.log'),'w');
let display,application,socket;
try {
 const env={...process.env,XDG_RUNTIME_DIR:temporary,WAYLAND_DISPLAY:'slashapp-test',SLASHSTITCH_DISPLAY:'wayland'};
 delete env.ELECTRON_RUN_AS_NODE;
 display=spawn('weston',['--backend=headless','--socket=slashapp-test','--idle-time=0','--width=1440','--height=900'],{env,stdio:['ignore',log,log]});
 display.on('error',error=>console.error(error.message));
 await until(()=>existsSync(join(temporary,'slashapp-test')),'headless Wayland socket');
 const userData=join(temporary,'profile');
 // --no-sandbox is restricted to this throwaway integration test, never start.mjs or packaged config.
 application=spawn(process.env.SLASHSTITCH_TEST_EXECUTABLE || require('electron'),[
   ...(process.env.SLASHSTITCH_TEST_EXECUTABLE?[]:[resolve('.')]),
   '--no-sandbox','--disable-gpu','--remote-debugging-port=0',`--user-data-dir=${userData}`
 ],{env,stdio:['ignore',log,log]});
 application.on('error',error=>console.error(error.message));
 const portFile=join(userData,'DevToolsActivePort');
 await until(()=>existsSync(portFile),'DevTools port');
 const port=readFileSync(portFile,'utf8').split('\n')[0];
 const page=await until(async()=>{
  try {return (await (await fetch(`http://127.0.0.1:${port}/json/list`)).json()).find(page=>page.type==='page');}
  catch {return null;}
 },'renderer target');
 socket=new WebSocket(page.webSocketDebuggerUrl);
 await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
 let serial=0;const pending=new Map();const errors=[];
 socket.addEventListener('message',event=>{
  const message=JSON.parse(String(event.data));
  if(message.method==='Runtime.exceptionThrown')errors.push(JSON.stringify(message.params));
  if(message.id) {const call=pending.get(message.id);if(call){clearTimeout(call.timer);pending.delete(message.id);message.error?call.reject(new Error(JSON.stringify(message.error))):call.resolve(message.result);}}
 });
 function send(method,params={}) {return new Promise((resolve,reject)=>{const id=++serial;const timer=setTimeout(()=>{pending.delete(id);reject(new Error(`CDP timeout: ${method} ${params.expression ?? ""}`));},5000);pending.set(id,{resolve,reject,timer});socket.send(JSON.stringify({id,method,params}));});}
 async function evaluate(expression) {
  const response=await send('Runtime.evaluate',{expression:`(async () => (${expression}))()`,awaitPromise:true,returnByValue:true});
  if(response.exceptionDetails)throw new Error(JSON.stringify(response.exceptionDetails));
  return response.result.value;
 }
 await send('Runtime.enable');
 await until(()=>evaluate('Boolean(document.getElementById("start-job") && window.Host)'), 'loaded demo UI');
 assert.equal(await evaluate('getComputedStyle(document.body).backgroundColor'),'rgb(22, 24, 29)');
 assert.equal(await evaluate('typeof require'),'undefined');
 assert.equal(await evaluate('(await window.Host.call("unlisted-action")).ok'),false);
 const sourceA=join(temporary,'source-a.png'),sourceB=join(temporary,'source-b.png');
 writeFileSync(sourceA,png());writeFileSync(sourceB,png(160,120,[70,140,220]));
 const center=await evaluate('(() => {const r=document.querySelector(".viewport-canvas").getBoundingClientRect();return{x:r.x+r.width/2,y:r.y+r.height/2};})()');
 async function drop(path){
   const data={items:[],files:[path],dragOperationsMask:1};
   for(const type of ['dragEnter','dragOver','drop'])await send('Input.dispatchDragEvent',{type,x:center.x,y:center.y,data});
 }
 await drop(sourceA);
 await until(()=>evaluate('document.querySelector(".panel-head strong").textContent.includes("(1)")'),'import through real drop handler');
 await until(()=>evaluate('JSON.parse((await window.Host.call("shell-info")).text).jobState === "idle"'),'import finalized');
 await drop(sourceB);
 await until(()=>evaluate('document.querySelector(".panel-head strong").textContent.includes("(2)")'),'second import');
 await until(()=>evaluate('JSON.parse((await window.Host.call("shell-info")).text).jobState === "idle"'),'second import finalized');
 await evaluate('Array.from(document.querySelectorAll("button")).find(b=>b.textContent.trim()==="Undo").click()');
 await until(()=>evaluate('document.querySelector(".panel-head strong").textContent.includes("(1)")'),'undo removes imported image and metadata');
 await until(()=>evaluate('JSON.parse((await window.Host.call("shell-info")).text).jobState === "idle"'),'undo finalized');
 await evaluate('Array.from(document.querySelectorAll("button")).find(b=>b.textContent.trim()==="Redo").click()');
 await until(()=>evaluate('document.querySelector(".panel-head strong").textContent.includes("(2)")'),'redo restores imported image and metadata');
 await until(()=>evaluate('JSON.parse((await window.Host.call("shell-info")).text).jobState === "idle"'),'redo finalized');
 const rejected=await evaluate('window.Host.call("export-panorama",{})');assert.equal(rejected.ok,false);
 assert.equal(await evaluate('JSON.parse((await window.Host.call("shell-info")).text).jobState'),'idle');
 await evaluate(`window.Host.call("export-panorama",{output_path:${JSON.stringify(join(temporary,'unsupported.bad'))},width:256,height:128,multiband:false})`);
 await until(()=>evaluate('document.getElementById("job-status").textContent.includes("unsupported format")'),'visible export error');
 await until(()=>evaluate('JSON.parse((await window.Host.call("shell-info")).text).jobState === "idle"'),'failed export finalized');
 await wait(250);
 assert.equal(await evaluate('document.getElementById("job-status").textContent.includes("unsupported format")'),true);
 await evaluate(`document.querySelector('[aria-label="Dismiss error"]').click()`);
 await evaluate('Array.from(document.querySelectorAll("button")).find(b=>b.textContent.trim()==="Crop").click()');
 await evaluate('(() => {const h=document.querySelector(".crop-handle.h-nw");h.focus();h.dispatchEvent(new KeyboardEvent("keydown",{key:"ArrowRight",bubbles:true}));return true;})()');
 await until(()=>evaluate('document.querySelector(".crop-box").style.left === "1%"'),'crop keyboard updates selection');
 await evaluate(`window.Host.call("export-panorama",{output_path:${JSON.stringify(join(temporary,'cancelled.tif'))},width:1024,height:8192,multiband:false})`);
 await until(()=>evaluate('JSON.parse((await window.Host.call("shell-info")).text).jobState === "running"'),'export running');
 const refused=await evaluate('window.Host.call("clear-images")');assert.equal(refused.ok,false);
 await evaluate('document.getElementById("stop-job").click()');
 await until(()=>evaluate('document.getElementById("job-status").textContent === "idle"'),'cancel finalized without stale error');
 await evaluate('window.Host.call("clear-images")');
 await until(()=>evaluate('document.querySelector(".panel-head strong").textContent.includes("(0)")'),'cleared project');
 await until(()=>evaluate('JSON.parse((await window.Host.call("shell-info")).text).jobState === "idle"'),'clear finalized');
 if(process.env.SLASHSTITCH_SMOKE_SCREENSHOT) {
   const image=await send('Page.captureScreenshot',{format:'png'});
   writeFileSync(process.env.SLASHSTITCH_SMOKE_SCREENSHOT,Buffer.from(image.data,'base64'));
 }
 if(await evaluate('Boolean(document.querySelector(".splitter"))')) {
  const rect=await evaluate('(() => {const r=document.querySelector(".splitter").getBoundingClientRect();return {x:r.x+r.width/2,y:r.y+50,width:document.querySelector(".left-panel").getBoundingClientRect().width};})()');
  await send('Input.dispatchMouseEvent',{type:'mousePressed',x:rect.x,y:rect.y,button:'left',clickCount:1});
  await send('Input.dispatchMouseEvent',{type:'mouseMoved',x:rect.x+50,y:rect.y,button:'left',buttons:1});
  await send('Input.dispatchMouseEvent',{type:'mouseReleased',x:rect.x+50,y:rect.y,button:'left',clickCount:1});
  await until(async()=>(await evaluate('document.querySelector(".left-panel").getBoundingClientRect().width'))>rect.width,'dragged splitter');
 }
 assert.deepEqual(errors,[],'renderer exceptions');
 await evaluate('window.Host.call("close-window")');
 const result=await until(()=>application.exitCode !== null ? {code:application.exitCode} : null,'Application graceful exit',10000);
 assert.equal(result.code,0);
 console.log('PASS: real file drop, import undo/redo, persistent export errors, rejected starts, busy mutation guard, crop keyboard, cancellation and clean exit.');
} catch(error) {
 console.error(readFileSync(join(temporary,'process.log'),'utf8'));
 throw error;
} finally {
 socket?.close();application?.kill();display?.kill();closeSync(log);
 // Weston and Electron are test-owned only. Await exits before removing their temporary files.
 for(const child of [application,display]) if(child && child.exitCode===null && child.signalCode===null) await Promise.race([new Promise(resolve=>child.once('exit',resolve)),wait(1000).then(()=>child.kill('SIGKILL'))]);
 rmSync(temporary,{recursive:true,force:true,maxRetries:20,retryDelay:100});
}
