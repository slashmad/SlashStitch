import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {existsSync,mkdirSync,readFileSync,readdirSync,writeFileSync,rmSync} from 'node:fs';
import {dirname,join} from 'node:path';
const host=spawnSync('rustc',['-vV'],{encoding:'utf8'}).stdout.match(/^host: (.+)$/m)?.[1];
if(!host)throw new Error('Cannot identify Rust target');
const metadata=spawnSync('cargo',['metadata','--filter-platform',host,'--manifest-path','engine/Cargo.toml','--locked','--offline','--format-version','1'],{encoding:'utf8',maxBuffer:32*1024*1024});
if(metadata.status!==0)throw new Error(metadata.stderr||'Cargo metadata failed');
const data=JSON.parse(metadata.stdout),worker=data.packages.find(p=>p.name==='slashstitch-worker');
const tree=spawnSync('cargo',['tree','--manifest-path','engine/Cargo.toml','--locked','--offline','--package','slashstitch-worker','--edges','normal,build','--prefix','none','--format','{p}','--target',host],{encoding:'utf8',maxBuffer:16*1024*1024});
if(tree.status!==0)throw new Error(tree.stderr);
const active=new Set([...tree.stdout.matchAll(/^(\S+) v(\S+)/gm)].map(m=>`${m[1]}@${m[2]}`));
const output=join('build','notices');mkdirSync(output,{recursive:true});
const inventory=[];const missing=[];let text='SlashStitch Rust worker dependency notices\n\nSee the application LICENSE for the SlashStitch workspace. This inventory includes normal and build dependencies, including platform-specific optional code.\n\n';
for(const pkg of data.packages.filter(p=>active.has(`${p.name}@${p.version}`)&&p.source).sort((a,b)=>a.name.localeCompare(b.name)||a.version.localeCompare(b.version))){
 const root=dirname(pkg.manifest_path);
 const files=readdirSync(root,{withFileTypes:true}).filter(e=>e.isFile()&&/^(licen[sc]e|copying|notice|unlicense)/i.test(e.name)).map(e=>e.name);
 if(pkg.license_file&&existsSync(join(root,pkg.license_file))&&!files.includes(pkg.license_file))files.push(pkg.license_file);
 if(!pkg.license)throw new Error(`Missing license expression: ${pkg.name}@${pkg.version}`);
 let licenseRoot=root;
 if(!files.length){
  const supplemental=join('licenses','third-party',`${pkg.name}-${pkg.version}`);
  if(existsSync(supplemental)){
   const manifest=JSON.parse(readFileSync(join(supplemental,'SOURCE.json'),'utf8'));
   if(manifest.name!==pkg.name||manifest.version!==pkg.version)throw new Error('Supplemental notice identity mismatch');
   licenseRoot=supplemental;files.push(...manifest.files.map(f=>f.name));
   for(const file of manifest.files){if(createHash('sha256').update(readFileSync(join(supplemental,file.name))).digest('hex')!==file.sha256)throw new Error('Supplemental notice checksum mismatch');}
  }else{missing.push({name:pkg.name,version:pkg.version,repository:pkg.repository,manifest_path:pkg.manifest_path,license:pkg.license});continue;}
 }
 inventory.push({name:pkg.name,version:pkg.version,license:pkg.license,authors:pkg.authors,repository:pkg.repository,files});
 text+=`===== ${pkg.name} ${pkg.version} (${pkg.license}) =====\n`;
 text+=`Authors (package metadata): ${pkg.authors.join(', ')}\nUpstream: ${pkg.repository ?? pkg.homepage ?? 'See Cargo package metadata'}\n`;
 for(const file of files)text+=`\n--- ${file} ---\n${readFileSync(join(licenseRoot,file),'utf8')}\n`;
}
if(missing.length){writeFileSync(join('build','missing-notices.json'),JSON.stringify(missing,null,2)+'\n');throw new Error('Missing upstream license texts: '+missing.map(p=>p.name+'@'+p.version).join(', '));}
rmSync(join(output,'missing.json'),{force:true});rmSync(join('build','missing-notices.json'),{force:true});
writeFileSync(join(output,'RUST-THIRD-PARTY-NOTICES.txt'),text);
writeFileSync(join(output,'rust-dependencies.json'),JSON.stringify(inventory,null,2)+'\n');
console.log(`Prepared license texts for ${inventory.length} Rust dependencies`);
