import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,writeFileSync,readFileSync,rmSync,linkSync,readdirSync,existsSync,chmodSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {deflateSync,inflateSync} from 'node:zlib';
import {WorkerClient,png} from './helpers/worker-client.mjs';

function makePngChunk(kind,data){
 const crc=(buf)=>{let c=0xffffffff;for(const b of buf){c^=b;for(let i=0;i<8;i++)c=(c>>>1)^((c&1)?0xedb88320:0);}return(c^0xffffffff)>>>0;};
 const type=Buffer.from(kind);const out=Buffer.alloc(data.length+12);
 out.writeUInt32BE(data.length);type.copy(out,4);data.copy(out,8);
 out.writeUInt32BE(crc(Buffer.concat([type,data])),out.length-4);
 return out;
}
function png16Gradient(width=512,height=64){
 const header=Buffer.alloc(13);header.writeUInt32BE(width);header.writeUInt32BE(height,4);header[8]=16;header[9]=2;
 const rowBytes=1+width*6;const raw=Buffer.alloc(rowBytes*height);
 for(let y=0;y<height;y++){
  const row=y*rowBytes;
  for(let x=0;x<width;x++){
   const v=Math.round((x*65535)/Math.max(1,width-1));
   const off=row+1+x*6;
   raw.writeUInt16BE(v,off);raw.writeUInt16BE(v>>1,off+2);raw.writeUInt16BE(65535-v,off+4);
  }
 }
 return Buffer.concat([Buffer.from([137,80,78,71,13,10,26,10]),makePngChunk('IHDR',header),makePngChunk('IDAT',deflateSync(raw)),makePngChunk('IEND',Buffer.alloc(0))]);
}
function parsePngChunks(buf){
 const chunks=[];let pos=8;
 while(pos+8<=buf.length){
  const len=buf.readUInt32BE(pos);const type=buf.toString('ascii',pos+4,pos+8);
  const data=buf.subarray(pos+8,pos+8+len);chunks.push({type,data});pos+=12+len;
 }
 return chunks;
}
function extractPngIcc(buf){
 const c=parsePngChunks(buf).find(x=>x.type==='iCCP');if(!c)return null;
 const nul=c.data.indexOf(0);if(nul<0)return null;
 return inflateSync(c.data.subarray(nul+2));
}
function decodePngRgbSampleRow(buf){
 const chunks=parsePngChunks(buf);
 const ihdr=chunks.find(c=>c.type==='IHDR').data;
 const w=ihdr.readUInt32BE(0),h=ihdr.readUInt32BE(4),bd=ihdr[8];
 const idat=inflateSync(Buffer.concat(chunks.filter(c=>c.type==='IDAT').map(c=>c.data)));
 const bpp=bd===16?6:3;const rowBytes=w*bpp;const stride=1+rowBytes;
 const prev=Buffer.alloc(rowBytes);const curr=Buffer.alloc(rowBytes);const mid=Math.floor(h/2);
 for(let y=0;y<=mid;y++){
  const f=idat[y*stride];const src=y*stride+1;
  for(let i=0;i<rowBytes;i++){
   const x=idat[src+i];const a=i>=bpp?curr[i-bpp]:0;const b=prev[i];const c=(i>=bpp?prev[i-bpp]:0);
   let v=x;
   if(f===1)v=(x+a)&0xff;else if(f===2)v=(x+b)&0xff;else if(f===3)v=(x+((a+b)>>1))&0xff;
   else if(f===4){const p=a+b-c,pa=Math.abs(p-a),pb=Math.abs(p-b),pc=Math.abs(p-c);v=(x+(pa<=pb&&pa<=pc?a:pb<=pc?b:c))&0xff;}
   curr[i]=v;
  }
  curr.copy(prev);
 }
 const reds=[];
 for(let x=0;x<w;x++){reds.push(bd===16?curr.readUInt16BE(x*6):curr[x*3]);}
 return{w,h,bd,reds};
}
async function fixture(t,env={}){
 const directory=mkdtempSync(join(tmpdir(),'slashstitch-regression-'));
 const a=join(directory,'a.png'),b=join(directory,'b.png');writeFileSync(a,png());writeFileSync(b,png(160,120,[50,140,220]));
 const worker=new WorkerClient(directory,env);
 await worker.success('set-view',{width:256,height:200});worker.events=[];
 t.after(async()=>{await worker.close();rmSync(directory,{recursive:true,force:true});});
 return{worker,directory,a,b};
}
test('real worker: import undo/redo restores complete source and control-point sets',async t=>{
 const{worker,a,b}=await fixture(t);
 const first=await worker.success('load-images',{paths:[a],replace:true});
 const second=await worker.success('load-images',{paths:[b]});
 assert.equal(second.images.length,2);
 const restored=await worker.success('restore-poses',{images:first.images,control_points:[]});
 assert.equal(restored.images.length,1);assert.equal(restored.metas.length,1);
 const redo=await worker.success('restore-poses',{images:second.images,control_points:[]});
 assert.equal(redo.images.length,2);assert.equal(redo.metas.length,2);
 const empty=await worker.success('restore-poses',{images:[],control_points:[]});assert.equal(empty.images.length,0);assert.equal(empty.metas.length,0);
});
test('real worker: invalid start does not claim a job and oversized integers never wrap',async t=>{
 const{worker}=await fixture(t);
 assert.equal((await worker.call('export-panorama',{})).ok,false);
 assert.equal((await worker.call('job-status')).result.state,'idle');
 assert.equal((await worker.call('export-panorama',{output_path:'/tmp/unused.tif',width:4294967297,height:64})).ok,false);
 assert.equal((await worker.call('job-status')).result.state,'idle');
 assert.equal(worker.events.filter(e=>e.event==='job-state').length,0);
});
test('real worker: mutation gate, responsive cancel, and no stale image resurrection',async t=>{
 const{worker,a,b,directory}=await fixture(t);await worker.success('load-images',{paths:[a,b],replace:true});
 worker.events=[];
 // Queue the probes before awaiting replies. A tiny blank-image alignment may
 // finish before the next host round trip on an optimized Apple Silicon build.
 const [started,clear,load,cancel]=await Promise.all([
  worker.call('export-panorama',{output_path:join(directory,'cancelled-gate.tif'),width:1024,height:8192,multiband:false}),
  worker.call('clear-images'),
  worker.call('load-images',{paths:[a]}),
  worker.call('job-cancel'),
 ]);
 assert.equal(started.ok,true);assert.equal(clear.ok,false);assert.equal(load.ok,false);
 assert.equal(cancel.ok,true);await worker.idle();
 const doc=await worker.success('snapshot-project');assert.equal(doc.images.length,2);
 await worker.success('clear-images');const cleared=await worker.success('snapshot-project');assert.equal(cleared.images.length,0);
});
test('real worker: zero matches are a visible failure and do not commit fake alignment',async t=>{
 const{worker,a,b}=await fixture(t);await worker.success('load-images',{paths:[a,b],replace:true});
 const before=await worker.success('snapshot-project');worker.events=[];
 await worker.call('auto-align',{});await worker.idle();
 assert.ok(worker.events.some(e=>e.event==='worker-error'&&/connected overlap graph/.test(e.data.text)));
 assert.ok(!worker.events.some(e=>e.event==='align-complete'));
 assert.deepEqual(await worker.success('snapshot-project'),before);
});
test('real worker: partial import reports failures, total failure preserves prior project',async t=>{
 const{worker,a,directory}=await fixture(t);const missing=join(directory,'missing.png');
 const result=await worker.success('load-images',{paths:[a,missing],replace:true});assert.equal(result.images.length,1);assert.equal(result.failures.length,1);
 assert.equal((await worker.call('load-images',{paths:[missing],replace:true})).ok,false);await worker.idle();
 assert.equal((await worker.success('snapshot-project')).images.length,1);
});
test('real worker: export rejects source paths and hard links without changing bytes',async t=>{
 const{worker,a,directory}=await fixture(t);await worker.success('load-images',{paths:[a],replace:true});const original=readFileSync(a);const alias=join(directory,'linked.png');linkSync(a,alias);
 for(const path of[a,alias]){worker.events=[];await worker.call('export-panorama',{output_path:path,width:320,height:160,overwrite:true});await worker.idle();assert.ok(worker.events.some(e=>e.event==='worker-error'&&/source image/.test(e.data.text)));assert.deepEqual(readFileSync(a),original);}
});
test('real worker: cancelling TIFF export preserves existing output and removes temporary files',async t=>{
 const{worker,a,directory}=await fixture(t);await worker.success('load-images',{paths:[a],replace:true});
 const output=join(directory,'previous.tif');const original=Buffer.from('previous completed export');writeFileSync(output,original);worker.events=[];
 await worker.call('export-panorama',{output_path:output,width:1024,height:8192,multiband:false,overwrite:true});
 await worker.wait(e=>e.event==='job-progress'&&e.data.percent>25&&e.data.stage.startsWith('Rendering &'));
 await worker.call('job-cancel');await worker.idle();assert.deepEqual(readFileSync(output),original);
 assert.ok(worker.events.some(e=>e.event==='job-cancelled'&&/cancel/i.test(e.data.text)));
 assert.ok(!readdirSync(directory).some(f=>f.endsWith('.part')));
});
test('real worker: existing output needs approval and successful output embeds a color profile',async t=>{
 const{worker,a,directory}=await fixture(t);await worker.success('load-images',{paths:[a],replace:true});
 const output=join(directory,'out.png');worker.events=[];
 await worker.call('export-panorama',{output_path:output,width:256,height:128,multiband:false});await worker.idle();
 assert.ok(existsSync(output));assert.ok(readFileSync(output).includes(Buffer.from('iCCP')));const original=readFileSync(output);
 worker.events=[];await worker.call('export-panorama',{output_path:output,width:128,height:64});await worker.idle();assert.deepEqual(readFileSync(output),original);assert.ok(worker.events.some(e=>e.event==='worker-error'&&/overwrite approval/.test(e.data.text)));
 for(const extension of ['tif','jpg']){
  const target=join(directory,'roundtrip.'+extension);worker.events=[];
  await worker.call('export-panorama',{output_path:target,width:256,height:128,multiband:false});await worker.idle();
  assert.ok(worker.events.some(e=>e.event==='export-complete'),JSON.stringify(worker.events));
  assert.ok(readFileSync(target).includes(Buffer.from(extension==='jpg'?'ICC_PROFILE':'acsp')));
  const imported=await worker.success('load-images',{paths:[target],replace:true});assert.deepEqual([imported.images[0].width,imported.images[0].height],[256,128]);
 }
});
test('real worker: project snapshots restore UI, geometry and CPs; failed restore is atomic',async t=>{
 const{worker,a,b}=await fixture(t);await worker.success('load-images',{paths:[a,b],replace:true});
 const doc=await worker.success('snapshot-project',{ui:{crop:{left:.1,top:.2,right:.9,bottom:.8},exportWidth:800}});
 doc.control_points=[{image_a:0,x_a:20,y_a:30,image_b:1,x_b:25,y_b:35}];doc.images[1].yaw_deg=23;
 const restored=await worker.success('restore-project',doc);assert.deepEqual(restored.project,doc);assert.equal(restored.metas.length,2);
 const bad=structuredClone(doc);bad.images[0].path+='missing';assert.equal((await worker.call('restore-project',bad)).ok,false);await worker.idle();assert.deepEqual(await worker.success('snapshot-project'),doc);
 const badVersion={...doc,schema_version:99};assert.equal((await worker.call('restore-project',badVersion)).ok,false);await worker.idle();assert.deepEqual(await worker.success('snapshot-project'),doc);
});
test('real worker: preview paths are private, atomic and distinct across workers',async t=>{
 const one=await fixture(t),two=await fixture(t);
 const a=await one.worker.success('load-images',{paths:[one.a],replace:true});const b=await two.worker.success('load-images',{paths:[two.a],replace:true});
 assert.notEqual(a.preview.preview_path,b.preview.preview_path);assert.ok(a.preview.preview_path.startsWith(one.directory));assert.ok(b.preview.preview_path.startsWith(two.directory));assert.ok(readFileSync(a.preview.preview_path).length>100);
});
test('real worker: EXIF Orientation rotates dimensions consistently',async t=>{
 const{worker}=await fixture(t);const result=await worker.success('load-images',{paths:[resolve('tests/fixtures/orientation-6.jpg')],replace:true});
 assert.deepEqual([result.images[0].width,result.images[0].height],[80,120]);assert.deepEqual([result.metas[0].width,result.metas[0].height],[80,120]);
});
test('real worker: malformed database fails validation without replacing active profiles',async t=>{
 const{worker,directory}=await fixture(t);writeFileSync(join(directory,'bad.xml'),'<lensdatabase><lens>');
 const before=await worker.success('engine-info');assert.equal((await worker.call('validate-lensfun',{path:directory})).ok,false);await worker.idle();
 assert.equal((await worker.call('reload-lensfun',{path:directory})).ok,false);await worker.idle();const after=await worker.success('engine-info');assert.equal(after.lenses_count,before.lenses_count);
});

test('real worker: simulated disk-full/write limit preserves prior TIFF and cleans partial file',{skip:process.platform!=='linux'},async t=>{
 const directory=mkdtempSync(join(tmpdir(),'slashstitch-write-failure-'));
 const wrapper=join(directory,'limited-worker.sh');
 writeFileSync(wrapper,`#!/bin/sh\nulimit -f 256\ntrap '' XFSZ\nexec "$SLASHSTITCH_REAL_WORKER"\n`);chmodSync(wrapper,0o700);
 const source=join(directory,'source.png');writeFileSync(source,png());
 const output=join(directory,'complete.tif');const original=Buffer.from('previous finished output');writeFileSync(output,original);
 const worker=new WorkerClient(directory,{SLASHSTITCH_TEST_WORKER:wrapper,SLASHSTITCH_REAL_WORKER:process.env.SLASHSTITCH_TEST_WORKER||resolve('engine/target/debug/slashstitch-worker')});
 t.after(async()=>{await worker.close();rmSync(directory,{recursive:true,force:true});});
 await worker.success('set-view',{width:256,height:200});await worker.success('load-images',{paths:[source],replace:true});worker.events=[];
 await worker.call('export-panorama',{output_path:output,width:1024,height:512,overwrite:true,multiband:false});await worker.idle();
 assert.deepEqual(readFileSync(output),original);assert.ok(worker.events.some(e=>e.event==='worker-error'&&/large|space|I\/O/i.test(e.data.text)));
 assert.ok(!readdirSync(directory).some(f=>f.endsWith('.part')));
});

test('real worker: high-bit-depth precision is reported and transparent input is rejected',async t=>{
 const{worker}=await fixture(t);
 const imported=await worker.success('load-images',{paths:[resolve('tests/fixtures/16bit.tif')],replace:true});
 assert.equal(imported.metas[0].bit_depth,16);
 assert.equal(imported.metas[0].icc_status,'missing');
 const rejected=await worker.call('load-images',{paths:[resolve('tests/fixtures/transparent.png')],replace:true});assert.equal(rejected.ok,false);assert.match(rejected.error,/Transparent sources/);await worker.idle();
 assert.equal((await worker.success('snapshot-project')).images.length,1);
});

test('real worker: 10-bit export preserves >256 tonal levels and transforms & embeds target ICC across preview and export',async t=>{
 const{worker,directory,a}=await fixture(t);
 const gradPath=join(directory,'grad16.png');writeFileSync(gradPath,png16Gradient(1024,64));
 const loaded=await worker.success('load-images',{paths:[gradPath],replace:true});
 assert.equal(loaded.metas[0].bit_depth,16);assert.equal(loaded.metas[0].icc_status,'missing');
 const srgbView=await worker.success('set-view',{projection:'rectilinear',pano_hfov_deg:55,view_yaw_deg:0,view_pitch_deg:0,width:512,height:64});
 const srgbPreview=readFileSync(srgbView.preview_path);
 assert.ok(extractPngIcc(srgbPreview),'sRGB preview embeds ICC');

 // 8-bit PNG in sRGB has at most 256 unique levels.
 const out8=join(directory,'out8-srgb.png');
 worker.events=[];
 await worker.call('export-panorama',{output_path:out8,width:1024,height:128,multiband:false,bit_depth:8,color_profile:'srgb'});
 await worker.idle();
 const ev8=worker.events.find(e=>e.event==='export-complete');assert.ok(ev8);
 assert.equal(ev8.data.bit_depth,8);assert.equal(ev8.data.container_bit_depth,8);assert.equal(ev8.data.icc_embedded,true);
 assert.ok(new Set(decodePngRgbSampleRow(readFileSync(out8)).reds).size<=256);

 // Switch project to 10-bit Adobe RGB and verify preview updates with embedded Adobe RGB ICC and shifted pixels.
 const switched=await worker.success('set-color-settings',{output_bit_depth:10,color_profile:'adobe_rgb'});
 assert.equal(switched.project.output_bit_depth,10);assert.equal(switched.project.color_profile,'adobe_rgb');
 assert.equal(switched.preview.color_profile,'adobe_rgb');
 const adobePreview=readFileSync(switched.preview.preview_path);
 assert.ok(extractPngIcc(adobePreview),'Adobe RGB preview embeds ICC');
 assert.notEqual(Buffer.compare(srgbPreview,adobePreview),0,'Preview renders differently when target profile changes');

 // 10-bit PNG export in Adobe RGB preserves >256 quantized 10-bit levels inside RGB16 PNG.
 const out10Png=join(directory,'out10-adobe.png');
 worker.events=[];
 await worker.call('export-panorama',{output_path:out10Png,width:1024,height:128,multiband:false});
 await worker.idle();
 const ev10=worker.events.find(e=>e.event==='export-complete');assert.ok(ev10);
 assert.equal(ev10.data.bit_depth,10);assert.equal(ev10.data.container_bit_depth,16);assert.equal(ev10.data.color_profile,'adobe_rgb');assert.equal(ev10.data.icc_embedded,true);
 const decoded10=decodePngRgbSampleRow(readFileSync(out10Png));
 assert.equal(decoded10.bd,16);
 assert.ok(new Set(decoded10.reds).size>300,`Expected >300 10-bit gradient levels, got ${new Set(decoded10.reds).size}`);

 // 10-bit TIFF export in Display P3 and re-import verifies source bit depth & embedded ICC inspection.
 const out10Tif=join(directory,'out10-p3.tif');
 await worker.success('set-color-settings',{color_profile:'display_p3'});
 worker.events=[];
 await worker.call('export-panorama',{output_path:out10Tif,width:256,height:128,multiband:false,bit_depth:10});
 await worker.idle();
 assert.ok(worker.events.some(e=>e.event==='export-complete'));
 // Also mix in an 8-bit source without ICC and verify per-source inspection distinguishes them cleanly.
 const mixed=await worker.success('load-images',{paths:[out10Tif,a],replace:true});
 assert.equal(mixed.metas[0].bit_depth,16);assert.equal(mixed.metas[0].icc_status,'embedded');assert.equal(mixed.metas[0].icc_preset_match,'display_p3');
 assert.equal(mixed.metas[1].bit_depth,8);assert.equal(mixed.metas[1].icc_status,'missing');

 // User-triggered adoption works for embedded source ICC and fails cleanly for missing ICC.
 const adopted=await worker.success('set-color-settings',{adopt_source_icc_index:0});
 assert.equal(adopted.project.color_profile,'display_p3');
 const missingAdopt=await worker.call('set-color-settings',{adopt_source_icc_index:1});
 assert.equal(missingAdopt.ok,false);await worker.idle();

 // Custom ICC loaded from file path (using Display P3 ICC file as valid RGB ICC) & JPEG rejects 10-bit synchronously.
 const customIccPath=join(directory,'test-profile.icc');
 writeFileSync(customIccPath,extractPngIcc(readFileSync(out10Png)));
 const customRes=await worker.success('set-color-settings',{custom_icc_path:customIccPath});
 assert.ok(['adobe_rgb','custom_icc'].includes(customRes.project.color_profile));
 // Explicit custom_icc payload preserves custom_icc in project snapshot and restores cleanly.
 const explicitCustom=await worker.success('set-color-settings',{
  color_profile:'custom_icc',
  custom_icc_name:'Test Adobe ICC',
  custom_icc_base64:extractPngIcc(readFileSync(out10Png)).toString('base64')
 });
 assert.equal(explicitCustom.project.color_profile,'custom_icc');assert.equal(explicitCustom.project.custom_icc_name,'Test Adobe ICC');
 // Full 16-bit export preserves >1024 unique tonal levels in ProPhoto RGB PNG.
 const gradWide=join(directory,'grad2048.png');writeFileSync(gradWide,png16Gradient(2048,64));
 await worker.success('load-images',{paths:[gradWide],replace:true});
 await worker.success('set-view',{projection:'rectilinear',pano_hfov_deg:55,view_yaw_deg:0,view_pitch_deg:0,width:512,height:64});
 await worker.success('set-color-settings',{output_bit_depth:16,color_profile:'prophoto_rgb'});
 const info=await worker.success('engine-info');
 assert.deepEqual(info.supported_bit_depths,[8,10,16]);
 const out16Png=join(directory,'out16-prophoto.png');
 worker.events=[];
 await worker.call('export-panorama',{output_path:out16Png,width:2048,height:128,multiband:false,bit_depth:16});
 await worker.idle();
 const ev16=worker.events.find(e=>e.event==='export-complete');assert.ok(ev16);
 assert.equal(ev16.data.bit_depth,16);assert.equal(ev16.data.container_bit_depth,16);assert.equal(ev16.data.color_profile,'prophoto_rgb');assert.equal(ev16.data.icc_embedded,true);
 const decoded16=decodePngRgbSampleRow(readFileSync(out16Png));
 assert.equal(decoded16.bd,16);
 assert.ok(new Set(decoded16.reds).size>1200,`Expected >1200 16-bit gradient levels, got ${new Set(decoded16.reds).size}`);
 // JPEG rejects both 10-bit and 16-bit.
 for(const badBits of [10,16]){
  worker.events=[];
  await worker.call('export-panorama',{output_path:join(directory,`fail-${badBits}.jpg`),width:256,height:128,bit_depth:badBits});
  await worker.idle();
  assert.ok(worker.events.some(e=>e.event==='worker-error'&&/JPEG only supports 8-bit/.test(e.data.text)),JSON.stringify(worker.events));
 }
 assert.equal((await worker.call('job-status')).result.state,'idle');
 // JPEG succeeds when explicitly using 8-bit.
 worker.events=[];
 await worker.call('export-panorama',{output_path:join(directory,'ok8.jpg'),width:256,height:128,bit_depth:8,multiband:false});
 await worker.idle();
 assert.ok(worker.events.some(e=>e.event==='export-complete'&&e.data.bit_depth===8));
});

test('real worker: canvas aspect is project state, not window size (C04)',async t=>{
 const{worker,a,b}=await fixture(t);await worker.success('load-images',{paths:[a,b],replace:true});
 const first=await worker.success('set-view',{projection:'rectilinear',width:512,height:256});
 assert.ok(Math.abs(first.canvas_aspect-2)<1e-9);
 const resized=await worker.success('render-preview',{width:256,height:256});
 assert.ok(Math.abs(resized.width/resized.height-2)<0.02,`preview ${resized.width}x${resized.height} must keep the project aspect`);
 const doc=await worker.success('snapshot-project');assert.ok(Math.abs(doc.canvas_aspect-2)<1e-9);
});
test('real worker: a source changed on disk is never mixed with stale caches (C05)',async t=>{
 const{worker,directory,a,b}=await fixture(t);await worker.success('load-images',{paths:[a,b],replace:true});
 const doc=await worker.success('snapshot-project');
 writeFileSync(a,png(160,120,[220,30,30]));const later=new Date(Date.now()+5000);(await import('node:fs')).utimesSync(a,later,later);
 worker.events=[];await worker.call('export-panorama',{output_path:join(directory,'stale.tif'),width:256,height:128,multiband:false});await worker.idle();
 assert.ok(worker.events.some(e=>e.event==='worker-error'&&/changed on disk/i.test(e.data.text)),JSON.stringify(worker.events.filter(e=>e.event==='worker-error')));
 await worker.success('restore-project',doc);
 worker.events=[];await worker.call('export-panorama',{output_path:join(directory,'fresh.tif'),width:256,height:128,multiband:false});await worker.idle();
 assert.ok(worker.events.some(e=>e.event==='export-complete'),JSON.stringify(worker.events.filter(e=>e.event==='worker-error')));
});
test('real worker: every commit publishes the complete project state (C09)',async t=>{
 const{worker,a}=await fixture(t);worker.events=[];await worker.success('load-images',{paths:[a],replace:true});
 const synced=worker.events.find(e=>e.event==='project-synced');assert.ok(synced);
 assert.equal(synced.data.images.length,1);assert.equal(synced.data.metas.length,1);assert.ok(synced.data.project);
});
test('real worker: exact preview renders a region and newer requests supersede older ones',async t=>{
 const{worker,a,b}=await fixture(t);await worker.success('load-images',{paths:[a,b],replace:true});
 worker.events=[];
 assert.equal((await worker.call('render-exact',{token:1,width:400,height:200,region:[0,0,1,1]})).ok,true);
 assert.equal((await worker.call('render-exact',{token:2,width:300,height:120,region:[0.4,0.3,0.6,0.5]})).ok,true);
 const done=await worker.wait(m=>m.event==='exact-preview-ready'&&m.data.token===2);
 assert.equal(done.data.width,300);assert.equal(done.data.height,120);assert.deepEqual(done.data.region,[0.4,0.3,0.6,0.5]);
 assert.equal(done.data.exact,true);assert.ok(existsSync(done.data.preview_path));
 assert.equal((await worker.call('render-exact',{token:3,width:64,height:64,region:[0.5,0.5,0.4,0.6]})).ok,true);
 const failed=await worker.wait(m=>m.event==='exact-preview-failed'&&m.data.token===3);assert.match(failed.data.text,/region/i);
});
test('real worker: exposure compensation is a saved project setting',async t=>{
 const{worker,a,b}=await fixture(t);await worker.success('load-images',{paths:[a,b],replace:true});
 let doc=await worker.success('snapshot-project');assert.equal(doc.exposure_compensation,true);assert.equal(doc.photometric.gains.length,2);
 await worker.success('set-render-settings',{exposure_compensation:false});doc=await worker.success('snapshot-project');assert.equal(doc.exposure_compensation,false);
});
test('real worker: seam overlay returns a transparent frame/seam image and one colour per source',async t=>{
 const{worker,a,b}=await fixture(t);await worker.success('load-images',{paths:[a,b],replace:true});
 worker.events=[];assert.equal((await worker.call('render-seams',{token:7,width:320,height:160,region:[0,0,1,1],selected:1})).ok,true);
 const done=await worker.wait(m=>m.event==='seam-overlay-ready'&&m.data.token===7);
 assert.equal(done.data.colors.length,2);assert.equal(done.data.width,320);assert.ok(existsSync(done.data.preview_path));
 const doc=await worker.success('snapshot-project');assert.equal(doc.reject_transients,true);
 await worker.success('set-render-settings',{reject_transients:false});assert.equal((await worker.success('snapshot-project')).reject_transients,false);
});
test('real worker: advanced stitch settings are validated and saved with the project',async t=>{
 const{worker,a,b}=await fixture(t);await worker.success('load-images',{paths:[a,b],replace:true});
 let doc=await worker.success('snapshot-project');
 assert.equal(doc.interpolation,'spline36');assert.equal(doc.blend_width_deg,3);assert.equal(doc.correct_vignetting,true);assert.equal(doc.optimize_seams,true);
 await worker.success('set-render-settings',{interpolation:'lanczos3',blend_width_deg:6,correct_vignetting:false,optimize_seams:false});
 doc=await worker.success('snapshot-project');
 assert.equal(doc.interpolation,'lanczos3');assert.equal(doc.blend_width_deg,6);assert.equal(doc.correct_vignetting,false);assert.equal(doc.optimize_seams,false);
 assert.equal((await worker.call('set-render-settings',{interpolation:'nearest'})).ok,false);await worker.idle();
 assert.equal((await worker.call('set-render-settings',{blend_width_deg:100})).ok,false);await worker.idle();
 assert.equal((await worker.success('snapshot-project')).blend_width_deg,6);
});
test('real worker: every mutation publishes its new preview as an event (N05)',async t=>{
 const{worker,a}=await fixture(t);worker.events=[];await worker.success('load-images',{paths:[a],replace:true});
 const ev=worker.events.find(e=>e.event==='preview-updated');assert.ok(ev,'preview-updated event');assert.ok(existsSync(ev.data.preview_path));assert.equal(typeof ev.data.version,'number');
});
test('real worker: impossible exports are refused before any job or allocation (N01)',async t=>{
 const{worker,directory,a,b}=await fixture(t);await worker.success('load-images',{paths:[a,b],replace:true});worker.events=[];
 const reply=await worker.call('export-panorama',{output_path:join(directory,'huge.png'),width:50000,height:50000,multiband:true});
 assert.equal(reply.ok,false);assert.match(reply.error,/512 MiB|budget/);
 assert.ok(!worker.events.some(e=>e.event==='job-state'&&e.data.state==='running'),'no job started');
 assert.equal((await worker.call('ping',{})).ok,true);
});
