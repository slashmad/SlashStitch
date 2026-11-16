// Isolated Linux UI smoke: requires weston, never uses the desktop display or capture hardware.
import {spawn} from 'node:child_process';
import {mkdtempSync, openSync, existsSync, readFileSync, rmSync, closeSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {createRequire} from 'node:module';
import assert from 'node:assert/strict';
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
 display=spawn('weston',['--backend=headless-backend.so','--socket=slashapp-test','--idle-time=0','--width=1440','--height=900'],{env,stdio:['ignore',log,log]});
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
 await evaluate('document.getElementById("start-job").click()');
 await until(()=>evaluate('document.getElementById("job-status").textContent === "running"'),'running job');
 // Exercise the real BrowserWindow close handler while the job is active.
 await evaluate('window.Host.call("close-window")');
 await until(()=>evaluate('document.getElementById("job-status").textContent.includes("A job is active")'),'blocked Close');
 assert.equal(await evaluate('JSON.parse((await window.Host.call("shell-info")).text).jobState'),'running');
 await evaluate('window.Host.call("quit-app")');
 await wait(100);
 assert.equal(await evaluate('JSON.parse((await window.Host.call("shell-info")).text).jobState'),'running');
 await evaluate('document.getElementById("stop-job").click()');
 await until(()=>evaluate('document.getElementById("job-status").textContent === "idle"'),'finalized job');
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
 console.log('PASS: dark UI, isolated renderer, closed IPC actions, start/stop/finalize, guarded Close, graceful exit.');
} catch(error) {
 console.error(readFileSync(join(temporary,'process.log'),'utf8'));
 throw error;
} finally {
 socket?.close();application?.kill();display?.kill();closeSync(log);
 // Weston and Electron are test-owned only. Await exits before removing their temporary files.
 for(const child of [application,display]) if(child && child.exitCode===null && child.signalCode===null) await Promise.race([new Promise(resolve=>child.once('exit',resolve)),wait(1000).then(()=>child.kill('SIGKILL'))]);
 rmSync(temporary,{recursive:true,force:true,maxRetries:20,retryDelay:100});
}
