import test from 'node:test';
import assert from 'node:assert/strict';
import {EventEmitter} from 'node:events';
import {build} from 'esbuild';
import {mkdtempSync,writeFileSync,readFileSync,rmSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {createRequire} from 'node:module';
import {png} from './helpers/worker-client.mjs';
const require=createRequire(import.meta.url);
const temp=mkdtempSync(join(tmpdir(),'slashstitch-host-tests-'));
const app=new EventEmitter();app.setName=()=>{};app.getName=()=> 'SlashStitch';app.getVersion=()=> '0.1.0';app.getPath=()=>temp;app.whenReady=()=>Promise.resolve();app.requestSingleInstanceLock=()=>true;app.commandLine={getSwitchValue:()=>'',appendSwitch:()=>{}};
app.quit=()=>{const e={prevented:false,preventDefault(){this.prevented=true;}};app.emit('before-quit',e);if(!e.prevented)app.emit('will-quit');};
const events=[];let window;let closingUi=null;
class BrowserWindow extends EventEmitter {
 constructor(){super();window=this;this.webContents=new EventEmitter();this.webContents.mainFrame={};this.webContents.setWindowOpenHandler=()=>{};this.webContents.send=(name,...args)=>{events.push({name,args});if(args[0]==="request-ui-snapshot"&&closingUi)queueMicrotask(()=>call("ui-snapshot",{id:args[1].id,ui:closingUi}));};}
 loadFile(){return Promise.resolve();}isMaximized(){return false;}getSize(){return[1280,820];}
 close(){this.emit('close',{preventDefault(){}});}static getAllWindows(){return window?[window]:[];}
}
let savePath=null,openPath=null,messageResponse=1;const handlers=new Map();
const dialog={showErrorBox:(...args)=>{throw Error(args.join(':'));},showSaveDialog:async()=>({canceled:!savePath,filePath:savePath}),showOpenDialog:async()=>({canceled:!openPath,filePaths:openPath?[openPath]:[]}),showMessageBox:async()=>({response:messageResponse})};
const protocolHandlers=new Map();const openedLinks=[];
globalThis.__electronMock={app,BrowserWindow,dialog,ipcMain:{handle:(name,callback)=>handlers.set(name,callback)},protocol:{registerSchemesAsPrivileged:()=>{},handle:(scheme,callback)=>protocolHandlers.set(scheme,callback)},net:{fetch:async url=>new Response(readFileSync(new URL(url)))},shell:{openExternal:async url=>{openedLinks.push(url);}}};
const savedDebug=process.env.SLASHSTITCH_DEBUG_WORKER,savedCpu=process.env.SLASHSTITCH_FORCE_CPU;process.env.SLASHSTITCH_DEBUG_WORKER='1';process.env.SLASHSTITCH_FORCE_CPU='1';
await build({entryPoints:['src/main/index.ts'],outfile:join(temp,'host.cjs'),bundle:true,platform:'node',format:'cjs',define:{__dirname:JSON.stringify(resolve('dist/main'))},plugins:[{name:'electron-test-surface',setup(build){build.onResolve({filter:/^electron$/},()=>({path:'electron',namespace:'mock'}));build.onLoad({filter:/.*/,namespace:'mock'},()=>({contents:'module.exports=globalThis.__electronMock;'}));build.onLoad({filter:/src[\\/]main[\\/]worker\.ts$/},async args=>({loader:'ts',contents:readFileSync(args.path,'utf8').replace('{ super(); }','{ super(); globalThis.__testWorkerBridge=this; }')}));}}],logLevel:'silent'});
require(join(temp,'host.cjs'));await new Promise(resolve=>setImmediate(resolve));
const call=(action,payload=null,senderFrame=window.webContents.mainFrame)=>handlers.get('host:call')({sender:window.webContents,senderFrame},action,payload);
const parsed=async(action,payload)=>{const res=await call(action,payload);assert.equal(res.ok,true,res.text);return JSON.parse(res.text);};
async function until(check){const end=Date.now()+30000;while(Date.now()<end){if(await check())return;await new Promise(r=>setTimeout(r,10));}throw Error('Host state timeout');}
test.after(async()=>{try{await globalThis.__testWorkerBridge?.stop();}finally{app.emit('will-quit');rmSync(temp,{recursive:true,force:true});if(savedDebug===undefined)delete process.env.SLASHSTITCH_DEBUG_WORKER;else process.env.SLASHSTITCH_DEBUG_WORKER=savedDebug;if(savedCpu===undefined)delete process.env.SLASHSTITCH_FORCE_CPU;else process.env.SLASHSTITCH_FORCE_CPU=savedCpu;delete globalThis.__electronMock;delete globalThis.__testWorkerBridge;}});
test('title bar links open only fixed destinations and the Sponsor preference persists',async()=>{
 assert.equal((await call('open-link',{key:'https://evil.example'})).ok,false);
 assert.equal((await call('open-link',{key:'__proto__'})).ok,false);
 assert.equal((await call('open-link',{key:'sponsor'})).ok,true);
 assert.equal(openedLinks.at(-1),'https://github.com/sponsors/slashmad');
 assert.equal((await call('open-link',{key:'bug',engine:'Vulkan · 8 threads <script>/home/user'})).ok,true);
 const bug=new URL(openedLinks.at(-1));
 assert.equal(bug.origin+bug.pathname,'https://github.com/slashmad/SlashStitch/issues/new');
 const body=bug.searchParams.get('body');
 assert.match(body,/SlashStitch \d+\.\d+\.\d+/);assert.match(body,/Engine: Vulkan · 8 threads script\/home\/user/);assert.doesNotMatch(body,/[<>]/);
 assert.equal((await parsed('shell-info')).showSponsor,true);
 assert.deepEqual(await parsed('set-ui-preferences',{showSponsor:false}),{showSponsor:false});
 assert.equal(JSON.parse(readFileSync(join(temp,'settings.json'),'utf8')).showSponsor,false);
 assert.equal((await parsed('shell-info')).showSponsor,false);
 assert.deepEqual(await parsed('set-ui-preferences',{showSponsor:'yes'}),{showSponsor:false});
 await parsed('set-ui-preferences',{showSponsor:true});
});
test('real host: validates sender, failed starts, saved projects, recovery and worker restart',async()=>{
 assert.equal((await call('echo',{},{})).ok,false);assert.equal((await call('ui-snapshot',{id:'unrequested'})).ok,false);
 assert.equal((await call('export-panorama',{})).ok,false);assert.equal((await parsed('shell-info')).jobState,'idle');
 await parsed('set-view',{width:256,height:200});await until(async()=>(await parsed('shell-info')).jobState==='idle');
 const source=join(temp,'source.png');writeFileSync(source,png());await parsed('load-images',{paths:[source],replace:true});await until(async()=>(await parsed('shell-info')).jobState==='idle');
 assert.equal((await parsed('shell-info')).dirty,true);
 savePath=join(temp,'saved.ssp');const ui={crop:{left:.1,top:.2,right:.9,bottom:.8},exportWidth:800};
 await parsed('save-project',{ui});let disk=JSON.parse(readFileSync(savePath));assert.equal(disk.schema_version,1);assert.equal(disk.images[0].path,'source.png');assert.deepEqual(disk.ui,ui);assert.equal((await parsed('shell-info')).dirty,false);
 await parsed('update-image-pose',{id:0,yaw_deg:25});await until(async()=>(await parsed('shell-info')).jobState==='idle');
 await parsed('set-color-settings',{output_bit_depth:16,color_profile:'adobe_rgb'});await until(async()=>(await parsed('shell-info')).jobState==='idle');
 assert.equal((await parsed('shell-info')).dirty,true);await parsed('save-project',{ui});disk=JSON.parse(readFileSync(savePath));assert.equal(disk.images[0].yaw_deg,25);assert.equal(disk.output_bit_depth,16);assert.equal(disk.color_profile,'adobe_rgb');
 await parsed('clear-images');await until(async()=>(await parsed('shell-info')).jobState==='idle');
 openPath=savePath;const reopened=await parsed('open-project');assert.equal(reopened.images.length,1);assert.equal(reopened.images[0].yaw_deg,25);assert.equal(reopened.project.output_bit_depth,16);assert.equal(reopened.project.color_profile,'adobe_rgb');assert.deepEqual(reopened.project.ui,ui);
 await until(async()=>(await parsed('shell-info')).jobState==='idle');
 const bad=join(temp,'bad.ssp');writeFileSync(bad,'{broken');openPath=bad;assert.equal((await call('open-project')).ok,false);assert.equal((await parsed('snapshot-project')).images.length,1);
 await until(async()=>(await parsed('shell-info')).jobState==='idle');
 globalThis.__testWorkerBridge.child.kill('SIGKILL');await until(async()=>!(await parsed('shell-info')).engineAvailable);
 const disconnected=await parsed('shell-info');assert.equal(disconnected.jobState,'disconnected');assert.ok(events.some(e=>e.args[0]==='engine-disconnected'));
 // The last complete snapshot remains saveable while the engine is offline.
 await parsed('save-project',{ui});assert.equal(JSON.parse(readFileSync(savePath)).images.length,1);
 const recovered=await parsed('restart-engine');assert.equal(recovered.images.length,1);assert.equal(recovered.images[0].yaw_deg,25);
 await until(async()=>(await parsed('shell-info')).jobState==='idle');assert.equal((await parsed('shell-info')).engineAvailable,true);
 // Explicit recovery must select CPU even when no ambient CPU flag is present.
 delete process.env.SLASHSTITCH_FORCE_CPU;
 globalThis.__testWorkerBridge.child.kill('SIGKILL');await until(async()=>!(await parsed('shell-info')).engineAvailable);
 const cpuRecovered=await parsed('restart-engine-cpu');assert.equal(cpuRecovered.images[0].yaw_deg,25);
 assert.match((await parsed('engine-info')).gpu_backend,/CPU/);
 await until(async()=>(await parsed('shell-info')).jobState==='idle');process.env.SLASHSTITCH_FORCE_CPU='1';
 await parsed('update-image-pose',{id:0,yaw_deg:30});await until(async()=>(await parsed('shell-info')).jobState==='idle');
 closingUi={crop:{left:.25,top:.1,right:.8,bottom:.9},exportWidth:1234};messageResponse=0;
 window.close();
 await until(()=>JSON.parse(readFileSync(savePath)).ui?.exportWidth===1234&&globalThis.__testWorkerBridge.child===null);
 assert.deepEqual(JSON.parse(readFileSync(savePath)).ui,closingUi);
 assert.equal(JSON.parse(readFileSync(savePath)).images[0].yaw_deg,30);
});
