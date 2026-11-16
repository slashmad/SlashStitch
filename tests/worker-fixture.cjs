const {createInterface} = require('node:readline');
const {spawn} = require('node:child_process');
const {closeSync} = require('node:fs');
const input = createInterface({input: process.stdin});
process.stdin.on('error', () => {});
const send = value => process.stdout.write(JSON.stringify(value)+'\n');
let descendant;
input.on('line', async line => {
  const {id,action,payload} = JSON.parse(line);
  const reply = result => send({id,ok:true,result});
  if(action === 'malformed') {
    for(const value of ['null','[]','3','"text"','not-json','{"id":3,"ok":"yes"}']) process.stdout.write(value+'\n');
    reply('survived');
  } else if(action === 'ignore') {
    // Stay responsive to subsequent requests, but never answer this one.
  } else if(action === 'close-input') {
    reply('closing');
    input.close();
    closeSync(0);
    send({event:'input-closed',data:null});
    setTimeout(() => process.exit(0),200);
  } else if(action === 'descendant') {
    descendant=spawn(process.execPath,['-e','setInterval(()=>{},1000)'],{stdio:'ignore'});
    descendant.once('spawn',()=>reply(descendant.pid));
  } else if(action === 'shutdown') {
    if(descendant) {
      const done=new Promise(resolve=>descendant.once('exit',resolve));
      descendant.kill(); await done;
    }
    reply(null); input.close(); process.stdin.destroy();
  } else reply(payload);
});
