const {appendFileSync} = require('node:fs');
const {createInterface} = require('node:readline');
const cpu=process.env.FIXTURE_CPU==='1';
const trace=value=>appendFileSync(process.env.FIXTURE_TRACE,JSON.stringify({cpu,...value})+'\n');
trace({start:process.pid});
if(process.env.FIXTURE_MODE==='crash'&&(!cpu||process.env.FIXTURE_FAIL_CPU==='1')){
 process.stderr.write(cpu?'CPU probe failed\n':'simulated native GPU crash\n');
 process.exit(73);
}
const input=createInterface({input:process.stdin});
input.on('line',line=>{
 const {id,action,payload}=JSON.parse(line);trace({action});
 const reply=result=>process.stdout.write(JSON.stringify({id,ok:true,result})+'\n');
 if(action==='engine-info'){
  if(!cpu&&process.env.FIXTURE_MODE==='hang')return;
  reply({backend:cpu?'CPU':'GPU'});
 }else if(action==='increment'){trace({mutation:true});reply(payload);}
 else if(action==='crash')process.exit(73);
 else if(action==='shutdown'){reply(null);input.close();process.stdin.destroy();}
 else reply(payload);
});
