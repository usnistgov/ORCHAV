// Deterministic local Chrome captures, using Node's built-in CDP WebSocket.
// Usage: node capture_film.mjs DEMO_DIR CAPTURE_DIR [FRAME_COUNT]
import {spawn} from 'node:child_process';
import {mkdtemp, mkdir, writeFile, rm, copyFile} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {resolve, join, dirname} from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';

const [demoArg,captureArg,countArg]=process.argv.slice(2);
if(!demoArg||!captureArg)throw Error('Specify demo and capture directories');
const demo=resolve(demoArg),out=resolve(captureArg),count=Number(countArg||320);
await mkdir(out,{recursive:true});
await copyFile(join(dirname(fileURLToPath(import.meta.url)),'film.html'),join(demo,'film.html'));
const profile=await mkdtemp(join(tmpdir(),'orchav-film-chrome-'));
const browser=spawn('/usr/bin/google-chrome',['--headless=new','--no-sandbox','--disable-gpu','--allow-file-access-from-files','--remote-debugging-port=0',`--user-data-dir=${profile}`,'about:blank'],{stdio:['ignore','ignore','pipe']});
let socket;
try{
  const endpoint=await new Promise((resolve,reject)=>{let message='';browser.stderr.on('data',chunk=>{message+=chunk;const m=message.match(/DevTools listening on (ws:\/\/[^\s]+)/);if(m)resolve(m[1])});browser.on('error',reject);browser.on('exit',code=>reject(Error(`Chrome exited ${code}`)));});
  const origin=new URL(endpoint);const target=await (await fetch(`http://${origin.host}/json/new?${encodeURIComponent(pathToFileURL(join(demo,'film.html')).href)}`,{method:'PUT'})).json();
  socket=new WebSocket(target.webSocketDebuggerUrl);await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject});
  let sequence=0;const pending=new Map();const errors=[];
  socket.onmessage=event=>{const m=JSON.parse(event.data);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(Error(JSON.stringify(m.error))):p.resolve(m.result)}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params)};
  const call=(method,params={})=>new Promise((resolve,reject)=>{const id=++sequence;pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}))});
  await call('Page.enable');await call('Runtime.enable');await call('Emulation.setDeviceMetricsOverride',{width:1920,height:1080,deviceScaleFactor:1,mobile:false});
  await call('Page.navigate',{url:pathToFileURL(join(demo,'film.html')).href});
  for(let attempt=0;attempt<100;attempt++){const ready=await call('Runtime.evaluate',{expression:'typeof window.renderFrame === "function"'});if(ready.result.value)break;await new Promise(resolve=>setTimeout(resolve,100));if(attempt===99)throw Error('Presentation did not load')}
  const frames=[];
  for(let index=0;index<count;index++){
    const r=await call('Runtime.evaluate',{expression:`window.renderFrame(${index})`,awaitPromise:true,returnByValue:true});
    if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));
    const shot=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false});
    await writeFile(join(out,`${String(index).padStart(4,'0')}.png`),Buffer.from(shot.data,'base64'));
    frames.push({index,...r.result.value});if(index%25===0)console.log(`Presentation ${index+1}/${count}`);
  }
  if(errors.length)throw Error(JSON.stringify(errors));
  await writeFile(join(out,'capture.json'),JSON.stringify({width:1920,height:1080,fps:25,frames},null,2)+'\n');
}finally{
  socket?.close();browser.kill('SIGTERM');await new Promise(resolve=>browser.once('exit',resolve));await rm(profile,{recursive:true,force:true});
}
