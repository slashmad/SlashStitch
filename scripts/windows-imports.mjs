// Inspect the deployed PE bytes, independent of Visual Studio tools on the host.
export function windowsImports(bytes){
 const checked=(offset,length)=>{if(offset<0||offset+length>bytes.length)throw new Error('Truncated Windows worker');return offset;};
 checked(0,64);
 if(bytes.toString('ascii',0,2)!=='MZ')throw new Error('Expected a Windows PE executable');
 const pe=checked(bytes.readUInt32LE(0x3c),24);
 if(bytes.toString('ascii',pe,pe+4)!=='PE\0\0')throw new Error('Invalid PE header');
 const count=bytes.readUInt16LE(pe+6),optional=pe+24,size=bytes.readUInt16LE(pe+20);
 checked(optional,size);
 if(size<128||bytes.readUInt16LE(optional)!==0x20b)throw new Error('Expected a 64-bit PE optional header');
 const sections=optional+size;checked(sections,count*40);
 const offsetFor=rva=>{
  if(rva<bytes.readUInt32LE(optional+60))return checked(rva,1);
  for(let i=0;i<count;i++){
   const s=sections+i*40,base=bytes.readUInt32LE(s+12),rawSize=bytes.readUInt32LE(s+16);
   if(rva>=base&&rva-base<rawSize)return checked(bytes.readUInt32LE(s+20)+rva-base,1);
  }
  throw new Error('PE import RVA is outside the file');
 };
 const rva=bytes.readUInt32LE(optional+120);
 if(!rva)return [];
 let entry=offsetFor(rva);const imports=[];
 for(let i=0;i<4096;i++,entry+=20){
  checked(entry,20);
  if(bytes.subarray(entry,entry+20).every(b=>b===0))return imports;
  const start=offsetFor(bytes.readUInt32LE(entry+12)),end=bytes.indexOf(0,start);
  if(end<start||end-start>260)throw new Error('Invalid PE import name');
  imports.push(bytes.toString('ascii',start,end));
 }
 throw new Error('Unterminated PE import table');
}

export function verifyWindowsRuntime(bytes){
 const imports=windowsImports(bytes);
 const external=imports.filter(name=>/^(vcruntime|msvcp|msvcr|concrt|ucrtbase|api-ms-win-crt-)/i.test(name));
 if(external.length)throw new Error(`Windows worker requires an external C runtime: ${external.join(', ')}. Rebuild with +crt-static.`);
 console.log(`Verified Windows worker imports: ${imports.join(', ')}`);
 return imports;
}
