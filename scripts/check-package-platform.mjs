const target=process.argv[2];
if(process.platform!==target)throw new Error(`Build this installer on ${target}. Cross-packaging would include the wrong native Rust worker.`);
