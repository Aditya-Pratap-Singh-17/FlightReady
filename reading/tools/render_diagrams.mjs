import {chromium} from '../../frontend/node_modules/playwright-core/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import crypto from 'node:crypto';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const dir=path.join(root,'reading/assets/architecture');
const markdown=(await fs.readFile(path.join(root,'reading/CODEBASE_VISUAL_GUIDE.md'),'utf8')).replace(/\r\n/g,'\n');
const definitions=JSON.parse(await fs.readFile(path.join(dir,'diagrams.json'),'utf8'));
const blocks=[...markdown.matchAll(/<!-- diagram: ([\w-]+) -->\s*```mermaid\s*\n([\s\S]*?)\n```/g)];
if(blocks.length!==definitions.length)throw new Error('Diagram count mismatch in Markdown');
const browser=await chromium.launch({channel:'msedge',headless:true});
const page=await browser.newPage({viewport:{width:1800,height:1200},deviceScaleFactor:2.5});
await page.route(/^https?:/,route=>route.abort());
await page.setContent('<html><head><meta charset="utf-8"><style>body{margin:0;background:white}#renderhost{display:inline-block;padding:0}svg{display:block;max-width:none!important}</style></head><body><div id="renderhost"></div></body></html>');
await page.addScriptTag({path:path.join(root,'reading/tools/mermaid.min.js')});
await page.evaluate(()=>window.mermaid.initialize({startOnLoad:false,securityLevel:'strict',theme:'base',htmlLabels:false,
  fontFamily:'Arial, sans-serif',fontSize:18,themeVariables:{primaryColor:'#eef4ed',primaryTextColor:'#1e3930',primaryBorderColor:'#749882',lineColor:'#657b70',secondaryColor:'#f5eddb',tertiaryColor:'#eaf0f5',background:'#ffffff',fontFamily:'Arial, sans-serif',fontSize:'18px',actorBkg:'#edf3ec',actorBorder:'#71927c',actorTextColor:'#223c30',signalColor:'#526e5c',signalTextColor:'#2e4438',noteBkgColor:'#faf3df',noteTextColor:'#4c4939',noteBorderColor:'#d3c7a5'},
  flowchart:{htmlLabels:false,useMaxWidth:false,nodeSpacing:26,rankSpacing:24,curve:'linear',padding:10,wrappingWidth:360},
  sequence:{useMaxWidth:false,mirrorActors:false,actorFontSize:17,messageFontSize:16,noteFontSize:15,diagramMarginX:14,diagramMarginY:18},
  class:{useMaxWidth:false,htmlLabels:false},state:{useMaxWidth:false}}));
const results=[];
for(const [,id,source] of blocks){
  const definition=definitions.find(d=>d.id===id);
  if(!definition||definition.source.trim()!==source.trim())throw new Error('Source mismatch '+id);
  const svg=await page.evaluate(async({id,source})=>{await window.mermaid.parse(source);return (await window.mermaid.render('guide-'+id,source)).svg;},{id,source});
  await fs.writeFile(path.join(dir,id+'.svg'),svg);
  const geometry=await page.evaluate(svg=>{
    const host=document.getElementById('renderhost');host.innerHTML=svg;
    const el=host.querySelector('svg'),box=el.viewBox.baseVal;
    el.setAttribute('width',String(Math.ceil(box.width)));el.setAttribute('height',String(Math.ceil(box.height)));
    el.style.width=Math.ceil(box.width)+'px';el.style.height=Math.ceil(box.height)+'px';
    return {width:box.width,height:box.height,labels:el.querySelectorAll('text').length,
      minFont:Math.min(...[...el.querySelectorAll('text')].map(n=>parseFloat(getComputedStyle(n).fontSize)||18))};
  },svg);
  await page.locator('#renderhost>svg').screenshot({path:path.join(dir,id+'.png')});
  results.push({id,title:definition.title,kind:definition.kind,source_sha256:crypto.createHash('sha256').update(source.trim()).digest('hex'),svg_sha256:crypto.createHash('sha256').update(svg).digest('hex'),parsed:true,rendered:true,...geometry});
  console.log(JSON.stringify({id,...geometry}));
}
await fs.writeFile(path.join(dir,'render-manifest.json'),JSON.stringify({renderer:'Mermaid 12.2.0; Microsoft Edge via Playwright',deviceScaleFactor:2.5,diagrams:results},null,2));
await browser.close();
