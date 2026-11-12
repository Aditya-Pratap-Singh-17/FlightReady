import {chromium} from '../../frontend/node_modules/playwright-core/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import assert from 'node:assert/strict';
import {pathToFileURL,fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const docs=path.join(root,'reading'),assets=path.join(docs,'assets/architecture');
const guideManifest=JSON.parse(await fs.readFile(path.join(assets,'guide-manifest.json'),'utf8'));
const browser=await chromium.launch({channel:'chrome',headless:true});
const page=await browser.newPage({viewport:{width:1440,height:1000}});
const errors=[],network=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('request',r=>{if(/^https?:/.test(r.url()))network.push(r.url());});
await page.route(/^https?:/,r=>r.abort());
await page.goto(pathToFileURL(path.join(docs,'CODEBASE_VISUAL_GUIDE.html')).href,{waitUntil:'load'});
await page.getByRole('heading',{name:'FlightReady: a visual codebase textbook',exact:true}).waitFor();
const inventory=await page.evaluate(()=>({
  chapters:document.querySelectorAll('#book a[id^="ch"]').length,
  diagrams:document.querySelectorAll('#book figure.diagram').length,
  diagramSvgs:document.querySelectorAll('#book figure.diagram svg').length,
  details:document.querySelectorAll('#book details').length,
  codeExamples:document.querySelectorAll('.code-wrap').length,
  internalLinks:[...document.querySelectorAll('#book a[href^="#"],.sidebar a[href^="#"]')].map(a=>a.getAttribute('href')),
  missingTargets:[...document.querySelectorAll('#book a[href^="#"],.sidebar a[href^="#"]')].filter(a=>!document.getElementById(decodeURIComponent(a.hash.slice(1)))).map(a=>a.getAttribute('href')),
  externalAssets:[...document.querySelectorAll('img[src],script[src],link[href]')].map(n=>n.src||n.href),
  svgMissingViewBox:[...document.querySelectorAll('figure.diagram svg')].filter(s=>!s.getAttribute('viewBox')).length,
  allSourceText:document.getElementById('book').textContent
}));
assert.equal(inventory.chapters,guideManifest.chapters);assert.equal(inventory.diagrams,guideManifest.diagrams);assert.equal(inventory.diagramSvgs,guideManifest.diagrams);
assert.equal(inventory.missingTargets.length,0);assert.equal(inventory.externalAssets.length,0);assert.equal(inventory.svgMissingViewBox,0);
await fs.mkdir(path.join(assets,'review'),{recursive:true});
await page.screenshot({path:path.join(assets,'review/html-desktop.png')});
await page.getByRole('navigation',{name:'Chapter navigation'}).getByRole('link').nth(8).click();
assert.equal(new URL(page.url()).hash,'#ch09');
await page.getByRole('button',{name:'Search the entire book',exact:false}).click();
await page.getByRole('searchbox',{name:'Search documentation text'}).fill('aggregate_quality');
await page.locator('#search-results button').first().waitFor();
const searchResults=await page.locator('#search-results button').count();assert(searchResults>0);
await page.screenshot({path:path.join(assets,'review/html-search.png')});
await page.locator('#search-results button').first().click();
assert.equal(await page.locator('#search-modal').evaluate(d=>d.open),false);
assert(new URL(page.url()).hash.startsWith('#search-block-'));
await page.locator('figure.diagram').first().getByRole('button').click();
const initial=await page.locator('#diagram-scale').textContent();
await page.getByRole('button',{name:'Zoom in',exact:true}).click();
assert.notEqual(await page.locator('#diagram-scale').textContent(),initial);
await page.getByRole('button',{name:'Fit',exact:true}).click();
assert.equal(await page.locator('#diagram-stage svg').count(),1);
await page.screenshot({path:path.join(assets,'review/html-diagram-zoom.png')});
await page.getByRole('button',{name:'Close',exact:true}).last().click();
await page.getByRole('button',{name:'Expand explanations'}).click();
assert.equal(await page.locator('#book details:not([open])').count(),0);
await page.getByRole('button',{name:'Collapse',exact:true}).click();
assert.equal(await page.locator('#book details[open]').count(),0);
await page.setViewportSize({width:390,height:844});
await page.goto(pathToFileURL(path.join(docs,'CODEBASE_VISUAL_GUIDE.html')).href);
await page.getByRole('button',{name:'Toggle chapter navigation'}).waitFor();
assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),true);
await page.screenshot({path:path.join(assets,'review/html-mobile.png'),fullPage:false});
await page.getByRole('button',{name:'Toggle chapter navigation'}).click();
assert(await page.locator('#sidebar').evaluate(e=>e.classList.contains('open')));
await page.getByRole('navigation',{name:'Chapter navigation'}).getByRole('link').nth(6).click();
assert.equal(new URL(page.url()).hash,'#ch07');
assert.equal(await page.locator('#sidebar').evaluate(e=>e.classList.contains('open')),false);
assert.deepEqual(errors,[]);assert.deepEqual(network,[]);
const canonical=s=>s.normalize('NFKC').replace(/\s+/g,'').replace(/\u00ad|\u200b/g,'');
const manifest=JSON.parse(await fs.readFile(path.join(assets,'export-content-manifest.json'),'utf8'));
const complete=canonical(inventory.allSourceText);
const missing=manifest.expected_text.map((text,block)=>({text,block})).filter(b=>!complete.includes(canonical(b.text)));
assert.equal(missing.length,0,'HTML text accounting failures: '+JSON.stringify(missing.slice(0,2)));
delete inventory.allSourceText;delete inventory.internalLinks;
const result={browser:'Installed Google Chrome, headless, file:// (no server)',...inventory,searchResults,zoomButtonsWorked:true,expandCollapseWorked:true,mobileNavigationWorked:true,documentHorizontalOverflow:false,networkRequests:network,browserErrors:errors,missingTextBlocks:missing.length,expectedTextBlocks:manifest.expected_text.length};
await fs.writeFile(path.join(assets,'html-verification.json'),JSON.stringify(result,null,2));
console.log(JSON.stringify(result));
await browser.close();
