import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,readFileSync,writeFileSync,rmSync,linkSync,readdirSync,existsSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {createRequire} from 'node:module';
import {createHash} from 'node:crypto';
import {build} from 'esbuild';
const temporary=mkdtempSync(join(tmpdir(),'slashstitch-storage-tests-'));
const require=createRequire(import.meta.url);
await build({entryPoints:['src/main/storage.ts','src/main/lensfun-update.ts'],outdir:temporary,bundle:true,platform:'node',format:'cjs',logLevel:'silent'});
const{writeJsonAtomic,protectSources}=require(join(temporary,'storage.js'));
const{downloadLensfunSnapshot}=require(join(temporary,'lensfun-update.js'));
test.after(()=>rmSync(temporary,{recursive:true,force:true}));
test('atomic project write preserves old data on serialization failure and protects aliases',()=>{
 const source=join(temporary,'original.png'),alias=join(temporary,'alias.ssp'),project=join(temporary,'project.ssp');
 writeFileSync(source,'original');linkSync(source,alias);assert.throws(()=>protectSources(alias,[source]),/source image/);
 const doc={schema_version:1,images:[],ui:{crop:[0,0,1,1]}};writeJsonAtomic(project,doc);assert.deepEqual(JSON.parse(readFileSync(project)),doc);
 const cyclic={};cyclic.self=cyclic;assert.throws(()=>writeJsonAtomic(project,cyclic));assert.deepEqual(JSON.parse(readFileSync(project)),doc);
 assert.ok(!readdirSync(temporary).some(p=>p.endsWith('.part')));
});
const sha='a'.repeat(40),xml=Buffer.from('<lensdatabase><lens><maker>Test</maker><model>50mm</model></lens></lensdatabase>');
const blob=createHash('sha1').update(`blob ${xml.length}\0`).update(xml).digest('hex');
function fetcher({badHash=false,httpError=false}={}){return async url=>{
 if(url.includes('/commits/'))return Response.json({sha});
 if(url.includes('/git/trees/'))return Response.json({truncated:false,tree:[{type:'blob',path:'data/db/test.xml',sha:badHash?'b'.repeat(40):blob,size:xml.length}]});
 return httpError?new Response('failed',{status:503}):new Response(xml);
};}
test('Lensfun updater publishes only a complete validated immutable snapshot',async()=>{
 const root=join(temporary,'snapshots-success');let validated=false;
 const snapshot=await downloadLensfunSnapshot(root,async path=>{assert.deepEqual(readFileSync(join(path,'test.xml')),xml);validated=true;},fetcher());
 assert.equal(validated,true);assert.equal(snapshot.sha,sha);assert.equal(snapshot.files,1);assert.ok(existsSync(join(snapshot.path,'snapshot.json')));assert.ok(!readdirSync(root).some(p=>p.startsWith('.staging')));
});
test('Lensfun updater cleans up HTTP, integrity and XML failures without touching prior snapshots',async()=>{
 for(const mode of['http','hash','xml']){
  const root=join(temporary,'snapshots-'+mode);let activated=false;
  await assert.rejects(downloadLensfunSnapshot(root,async()=>{activated=true;if(mode==='xml')throw Error('invalid XML');},fetcher({httpError:mode==='http',badHash:mode==='hash'})));
  assert.deepEqual(readdirSync(root),[]);if(mode!=='xml')assert.equal(activated,false);
 }
});
test('Lensfun cancellation during validation never publishes a partial snapshot',async()=>{
 const root=join(temporary,'snapshots-abort');const controller=new AbortController();
 await assert.rejects(downloadLensfunSnapshot(root,async()=>controller.abort(new Error('cancelled')),fetcher(),controller.signal),/cancelled/);
 assert.deepEqual(readdirSync(root),[]);
});

test('Lensfun inventory cannot escape staging on Windows or Unix',async()=>{
 const root=join(temporary,'unsafe-inventory');let downloaded=false;
 const unsafe=async url=>{
  if(url.includes('/commits/'))return Response.json({sha});
  if(url.includes('/git/trees/'))return Response.json({tree:[{type:'blob',path:'data/db/..\\escape.xml',sha:blob}]});
  downloaded=true;return new Response(xml);
 };
 await assert.rejects(downloadLensfunSnapshot(root,async()=>{},unsafe),/Unsafe/);assert.equal(downloaded,false);assert.equal(existsSync(root),false);
});
