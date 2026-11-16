import assert from 'node:assert/strict';
import {mkdtempSync,mkdirSync,readdirSync,copyFileSync,rmSync,writeFileSync} from 'node:fs';
import {join,resolve} from 'node:path';
import {tmpdir} from 'node:os';
import {WorkerClient} from '../tests/helpers/worker-client.mjs';
const output=process.env.SLASHSTITCH_VALIDATION_DIR||resolve('build/validation/real-images');mkdirSync(output,{recursive:true});
const root=resolve(process.argv[2]||'testbilder/2025-04-15');
const results=[];
for(const folder of readdirSync(root,{withFileTypes:true}).filter(e=>e.isDirectory()).sort((a,b)=>a.name.localeCompare(b.name))){
 const paths=readdirSync(join(root,folder.name)).filter(f=>/\.jpe?g$/i.test(f)).sort().map(f=>join(root,folder.name,f));if(!paths.length)continue;
 const temporary=mkdtempSync(join(tmpdir(),'slashstitch-real-validation-'));
 const worker=new WorkerClient(temporary,{SLASHSTITCH_FORCE_CPU:undefined,RAYON_NUM_THREADS:'4'});
 const start=Date.now();
 try{
  await worker.success('set-view',{width:1024,height:512});
  const loaded=await worker.success('load-images',{paths,replace:true});assert.equal(loaded.images.length,paths.length);assert.equal(loaded.failures.length,0);
  worker.events=[];await worker.success('auto-align',{optimize_hfov:true,level_horizon:true});
  const alignment=worker.events.find(e=>e.event==='align-complete')?.data;assert.ok(alignment,JSON.stringify(worker.events.filter(e=>e.event==='worker-error')));assert.ok(alignment.report.control_points_used>0);
  assert.ok(alignment.report.final_rms_deg<1,`${folder.name}: unacceptable RMS`);
  const crop=(await worker.success('auto-crop')).crop;
  const target=join(temporary,'export.png');worker.events=[];await worker.success('export-panorama',{output_path:target,width:1024,height:512,crop,multiband:true});
  assert.ok(worker.events.some(e=>e.event==='export-complete'),JSON.stringify(worker.events.filter(e=>e.event==='worker-error')));
  copyFileSync(target,join(output,`${folder.name}-export.png`));copyFileSync(alignment.preview.preview_path,join(output,`${folder.name}-preview.png`));
  const result={dataset:folder.name,inputs:paths.length,report:alignment.report,crop,elapsed_seconds:(Date.now()-start)/1000};results.push(result);console.log(JSON.stringify(result));
 }finally{await worker.close();rmSync(temporary,{recursive:true,force:true});}
}
writeFileSync(join(output,'results.json'),JSON.stringify(results,null,2)+'\n');
