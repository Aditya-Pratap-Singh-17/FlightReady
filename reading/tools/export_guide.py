"""Export the entire Markdown source to independent offline HTML and PDF."""
from pathlib import Path
from current_pipeline import SNAPSHOT
import base64
import hashlib
import html
import json
import re
import sys
import textwrap
from urllib.parse import unquote

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
from markdown_it import MarkdownIt
from pygments import highlight
from pygments.lexers import get_lexer_by_name, TextLexer
from pygments.formatters import HtmlFormatter
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4, A3, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak,
    NextPageTemplate, Image, LongTable, TableStyle, Preformatted, Flowable)
from reportlab.platypus.tableofcontents import TableOfContents
from PIL import Image as PILImage

ROOT=HERE.parents[1]
DOC=ROOT/'reading'
ASSETS=DOC/'assets/architecture'
SOURCE=DOC/'CODEBASE_VISUAL_GUIDE.md'
TEXT=SOURCE.read_text(encoding='utf-8')
PARSER=MarkdownIt('commonmark',{'html':True}).enable('table')
TOKENS=PARSER.parse(TEXT)
DIAGRAMS={d['id']:d for d in json.loads((ASSETS/'diagrams.json').read_text(encoding='utf-8'))}
RENDERS={d['id']:d for d in json.loads((ASSETS/'render-manifest.json').read_text(encoding='utf-8'))['diagrams']}
CHAPTERS=[('ch'+str(i).zfill(2),t.content) for i,t in enumerate([t for i,t in enumerate(TOKENS) if t.type=='inline' and i>0 and TOKENS[i-1].type=='heading_open' and TOKENS[i-1].tag=='h1' and re.match(r'^\d+\.',t.content)],1)]
HEADING_IDS={}
counts={}
for i,t in enumerate(TOKENS):
    if t.type=='heading_open':
        title=TOKENS[i+1].content
        slug=re.sub(r'[^a-z0-9]+','-',title.lower()).strip('-')
        counts[slug]=counts.get(slug,0)+1
        if counts[slug]>1:slug+='-'+str(counts[slug])
        t.attrSet('id',slug);HEADING_IDS[i]=slug

def diagram_id(source):
    norm=source.strip().replace('\r\n','\n')
    for id,d in DIAGRAMS.items():
        if d['source'].strip()==norm:return id
    raise ValueError('No exact rendered diagram for Mermaid fence')

def data_url(path):
    mime={'.svg':'image/svg+xml','.png':'image/png'}[path.suffix]
    return 'data:'+mime+';base64,'+base64.b64encode(path.read_bytes()).decode()

CSS=r'''
:root{--ink:#22382e;--dim:#627167;--paper:#fffefa;--ground:#f2f4ee;--line:#dce3d7;--green:#285d44;--soft:#eaf1e3;--gold:#bc9c5d;--rail:266px}*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:35px}body{margin:0;background:var(--ground);color:var(--ink);font:16px/1.8 'Segoe UI',Arial,sans-serif}a{color:#27634d;text-underline-offset:3px;text-decoration-thickness:1px}a:hover{color:#0f4930}button,input{font:inherit}button{cursor:pointer;border:1px solid var(--line);border-radius:7px;padding:7px 11px;background:var(--paper);color:var(--ink)}button:focus-visible,a:focus-visible,input:focus-visible{outline:3px solid #9bb895;outline-offset:3px}.sidebar{position:fixed;inset:0 auto 0 0;width:var(--rail);background:#193d2d;color:#c8d7c6;padding:28px 20px;overflow:auto;z-index:20}.brand{font-size:22px;color:#f7f9ee;font-weight:650;letter-spacing:-.7px}.brand small{display:block;font-size:10px;letter-spacing:.17em;font-weight:400;color:#afc6a6;margin-top:5px}.sidebar nav{display:flex;flex-direction:column;gap:4px;margin-top:28px}.sidebar nav a{display:flex;gap:10px;align-items:flex-start;padding:9px 10px;border-radius:6px;text-decoration:none;color:#b8cdb6;font-size:12px;line-height:1.5}.sidebar nav a>span{font-size:10px;opacity:.55;margin-top:2px;min-width:18px}.sidebar nav a.active,.sidebar nav a:hover{background:#31543a;color:#fbfff2}.sidebar-footer{font-size:10px;line-height:1.8;color:#9eb797;border-top:1px solid #3a5840;padding-top:20px;margin-top:26px}.sidebar-footer a{color:#d7e4ca}.main{margin-left:var(--rail);min-width:0;overflow-x:clip}.topbar{position:sticky;top:0;z-index:10;display:flex;align-items:center;justify-content:space-between;gap:15px;padding:15px 38px;background:#f7f9f2f2;border-bottom:1px solid var(--line);backdrop-filter:blur(10px)}.topbar>span{font-size:10px;letter-spacing:.12em;color:#66805e}.topbar>div{display:flex;gap:9px}.topbar button{font-size:11px;padding:6px 10px}.search-trigger{min-width:180px;text-align:left;color:var(--dim)}#reading-progress{position:fixed;top:0;left:var(--rail);height:3px;background:#789969;z-index:40;width:0}.book{min-width:0;overflow-wrap:anywhere;max-width:1050px;margin:0 auto;background:var(--paper);padding:58px 64px 75px;box-shadow:0 0 0 1px #e4e8dc;min-height:100vh}.book>h1:first-of-type{font-family:Georgia,serif;font-size:46px;line-height:1.16;font-weight:500;letter-spacing:-1.3px;margin:15px 0 25px;max-width:760px}.book>h1:first-of-type+p{font-size:19px;color:#496448;line-height:1.65}.book>h1:first-of-type+p+p{font-size:12px;color:#7c8976;border-bottom:1px solid var(--line);padding-bottom:28px;margin-bottom:35px}.book h1{font:500 34px/1.28 Georgia,serif;margin:65px 0 24px;color:#254b34;letter-spacing:-.6px;scroll-margin-top:85px}.book h2{font-size:23px;line-height:1.4;font-weight:600;letter-spacing:-.6px;margin:38px 0 17px;scroll-margin-top:85px}.book h3{font-size:16px;line-height:1.5;font-weight:650;margin:29px 0 12px;scroll-margin-top:85px;overflow-wrap:anywhere}.book p{margin:12px 0 20px}.book li{margin:7px 0}.book ul,.book ol{padding-left:26px}.book>hr{height:1px;background:var(--line);border:0;margin:50px 0}.book strong{font-weight:650}.book blockquote{margin:24px 0;background:#eff4e8;border-left:3px solid #9cb58d;padding:14px 24px;color:#486143}.book blockquote p{margin:0}.book code{font:12.5px/1.6 Consolas,'Courier New',monospace;background:#edf2e8;padding:2px 5px;border-radius:3px;overflow-wrap:anywhere}.book pre{border:1px solid #dce5d5;border-radius:8px;background:#f3f6ee!important;padding:20px!important;overflow:auto;font:12px/1.7 Consolas,monospace;tab-size:4;white-space:pre-wrap;overflow-wrap:anywhere;margin:20px 0}.book pre code{padding:0;background:none}.code-wrap{position:relative}.code-copy{position:absolute;top:9px;right:10px;font-size:10px;padding:3px 8px;opacity:.75}.table-wrap{overflow:auto;width:100%;margin:21px 0 30px;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;font-size:12px;line-height:1.65;table-layout:fixed}th{background:#edf3e7;color:#45633e;font-size:10px;letter-spacing:.04em;text-align:left;font-weight:650;padding:12px 13px;border-bottom:1px solid #cddac4}td{padding:13px;vertical-align:top;border-bottom:1px solid #e5eade;overflow-wrap:anywhere}tr:last-child td{border:0}tr:nth-child(even) td{background:#fafcf6}td code{font-size:11px;background:#eef3e8;padding:1px 2px}table a{font-size:11px}.book details{border:1px solid #dce5d5;border-radius:8px;background:#f7f9f2;padding:17px 21px;margin:23px 0}.book details summary{cursor:pointer;font-size:13px;font-weight:600;color:#426640;line-height:1.6}.book details[open] summary{margin-bottom:15px}.book details p{font-size:14px}.diagram{padding:18px;border:1px solid var(--line);border-radius:10px;margin:23px 0;background:white;position:relative;overflow:hidden}.diagram-canvas{overflow:auto;max-height:630px;display:flex;justify-content:center;align-items:flex-start;background:white;padding:12px}.diagram-canvas>svg{width:100%;max-width:100%;height:auto;display:block;min-width:270px}.diagram figcaption{font-size:11px;color:#72816c;padding-top:12px;border-top:1px solid #e3eade;margin-top:13px}.diagram-zoom{display:flex;align-items:center;gap:6px;font-size:10px;margin-left:auto;margin-bottom:8px}.diagram-hint{font-size:10px;color:#84917b}.chapter-evidence{font-size:11px}.book [id]{scroll-margin-top:85px}#search-modal,#diagram-modal{border:1px solid #becdb5;border-radius:12px;box-shadow:0 20px 80px #14351f40;padding:0;background:var(--paper);color:var(--ink)}dialog::backdrop{background:#0e2b1b99;backdrop-filter:blur(3px)}#search-modal{width:min(760px,calc(100vw - 28px));max-height:85vh}#search-modal header,#diagram-modal header{display:flex;align-items:center;justify-content:space-between;gap:15px;padding:17px 22px;border-bottom:1px solid var(--line)}#search-modal header strong,#diagram-modal header strong{font-size:14px}#search-input{margin:18px 22px;width:calc(100% - 44px);padding:12px 14px;border:1px solid #b7cbae;border-radius:8px;background:white;color:#233b2a;font-size:16px}#search-count{font-size:11px;color:var(--dim);padding:0 23px 12px}#search-results{max-height:53vh;overflow:auto;padding:0 16px 17px}#search-results button{display:block;width:100%;text-align:left;margin:6px 0;font-size:12px;padding:12px 14px;line-height:1.6}#search-results button strong{display:block;font-size:11px;color:#58814b;margin-bottom:4px}mark{background:#eee0ad;color:#3d482e;padding:0 2px;border-radius:2px}#diagram-modal{width:calc(100vw - 36px);max-width:1700px;height:calc(100vh - 36px);max-height:none}#diagram-modal header{flex-wrap:wrap}#diagram-toolbar{display:flex;align-items:center;gap:8px;font-size:12px}#diagram-toolbar button{font-size:12px}#diagram-viewport{height:calc(100% - 86px);overflow:auto;background:white;padding:25px;cursor:grab}#diagram-stage{display:table;margin:auto;background:white}#diagram-stage svg{display:block;max-width:none!important}.search-highlight{animation:highlight 3s ease-out}.mobile-menu{display:none}.book a[href^="assets/"]{font-size:11px}@keyframes highlight{0%,30%{background:#f0e7bf}100%{background:transparent}}@media(min-width:1800px){.book{max-width:1160px;padding:70px 85px}.book h1{font-size:38px}.book>h1:first-of-type{font-size:54px}}@media(max-width:1100px){:root{--rail:220px}.sidebar{padding:25px 14px}.book{padding:43px 36px}.topbar{padding:13px 25px}.book>h1:first-of-type{font-size:39px}.book h1{font-size:31px}.book h2{font-size:21px}table{font-size:11px}td,th{padding:10px}.book{font-size:15px}}@media(max-width:760px){:root{--rail:0px}.sidebar{width:260px;transform:translateX(-100%);transition:transform .2s}.sidebar.open{transform:translateX(0)}.main{margin:0}.mobile-menu{display:inline-block}.topbar{padding:12px 16px}.topbar>span{display:none}.topbar>div{flex:1;justify-content:space-between}.search-trigger{min-width:145px}.book{padding:32px 20px 50px;border:0;box-shadow:none;font-size:15px;line-height:1.8}.book>h1:first-of-type{font-size:36px}.book h1{font-size:29px;margin-top:49px}.book h2{font-size:21px}.book h3{font-size:16px}.book>h1:first-of-type+p{font-size:17px}.table-wrap{border-radius:6px}table{min-width:650px;font-size:12px}.book pre{font-size:11px;padding:16px!important}.diagram{padding:10px}.diagram-canvas{max-height:470px;padding:5px}.diagram-canvas svg{min-width:280px}.diagram figcaption{font-size:10px}.book details{padding:15px}.topbar button{font-size:10px}#diagram-modal header{padding:12px}#diagram-modal header strong{font-size:12px}#diagram-viewport{height:calc(100% - 110px);padding:12px}.book>hr{margin:38px 0}}@media print{.sidebar,.topbar,#reading-progress,.diagram-zoom,.code-copy{display:none!important}.main{margin:0}.book{padding:0;max-width:none}.book details{display:block}.book details>*{display:block}.diagram-canvas{max-height:none}table{min-width:0}.book h1{break-before:page}.table-wrap{overflow:visible}tr,figure{break-inside:avoid}a{color:inherit;text-decoration:none}body{background:white}}
'''

JS=r'''
const book=document.getElementById('book'),sidebar=document.getElementById('sidebar');
document.getElementById('menu-toggle').onclick=()=>sidebar.classList.toggle('open');
sidebar.querySelectorAll('a').forEach(a=>a.addEventListener('click',()=>sidebar.classList.remove('open')));
const chapterNodes=[...book.querySelectorAll('a[id^="ch"]')];
const chapterLinks=[...sidebar.querySelectorAll('nav a[href^="#ch"]')];
const progress=document.getElementById('reading-progress');
function syncScroll(){let current='ch01';for(const n of chapterNodes)if(n.getBoundingClientRect().top<180)current=n.id;chapterLinks.forEach(a=>a.classList.toggle('active',a.hash==='#'+current));const width=innerWidth>760?parseFloat(getComputedStyle(document.documentElement).getPropertyValue('--rail'))||0:0;progress.style.left=width+'px';progress.style.width=(innerWidth-width)*Math.min(1,scrollY/Math.max(1,document.documentElement.scrollHeight-innerHeight))+'px';}
addEventListener('scroll',syncScroll,{passive:true});addEventListener('resize',syncScroll);syncScroll();
document.getElementById('expand-all').onclick=()=>book.querySelectorAll('details').forEach(d=>d.open=true);
document.getElementById('collapse-all').onclick=()=>book.querySelectorAll('details').forEach(d=>d.open=false);
document.querySelectorAll('.code-copy').forEach(button=>button.onclick=async()=>{const value=button.parentNode.querySelector('pre').innerText;try{await navigator.clipboard.writeText(value);button.textContent='Copied';}catch{const r=document.createRange();r.selectNodeContents(button.parentNode.querySelector('pre'));getSelection().removeAllRanges();getSelection().addRange(r);button.textContent='Selected';}setTimeout(()=>button.textContent='Copy',1500);});
const search=document.getElementById('search-modal'),input=document.getElementById('search-input'),results=document.getElementById('search-results'),count=document.getElementById('search-count');
const searchable=[...book.querySelectorAll('p,li,td,pre,h2,h3')].filter(el=>!el.closest('svg')&&el.textContent.trim().length>12).map((el,i)=>{if(!el.id)el.id='search-block-'+i;return{el,text:el.textContent.trim(),lower:el.textContent.toLowerCase()};});
function openSearch(){search.showModal();setTimeout(()=>input.focus(),0);}
document.getElementById('open-search').onclick=openSearch;document.getElementById('close-search').onclick=()=>search.close();
addEventListener('keydown',e=>{if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='k'){e.preventDefault();openSearch();}});
function goTo(el){let p=el.parentElement;while(p){if(p.tagName==='DETAILS')p.open=true;p=p.parentElement;}search.close();el.scrollIntoView({block:'center',behavior:'auto'});history.replaceState(null,'','#'+el.id);el.classList.remove('search-highlight');requestAnimationFrame(()=>el.classList.add('search-highlight'));}
let searchTimer;
input.addEventListener('input',()=>{clearTimeout(searchTimer);searchTimer=setTimeout(()=>{const q=input.value.trim().toLowerCase();results.replaceChildren();if(q.length<2){count.textContent='Type at least two characters. Search covers all chapters, symbols, tables and expanded explanations.';return;}const matches=searchable.filter(r=>r.lower.includes(q));count.textContent=matches.length+' matching passages'+(matches.length>100?' (showing first 100)':'');for(const r of matches.slice(0,100)){const button=document.createElement('button'),heading=document.createElement('strong');let section=r.el;while(section&&!/H[123]/.test(section.tagName))section=section.previousElementSibling;heading.textContent=section?.textContent?.slice(0,100)||'Sourcebook passage';button.append(heading);const at=r.lower.indexOf(q),start=Math.max(0,at-75),end=Math.min(r.text.length,at+q.length+140),snippet=r.text.slice(start,end),offset=at-start;button.append(document.createTextNode((start?'...':'')+snippet.slice(0,offset)));const mark=document.createElement('mark');mark.textContent=snippet.slice(offset,offset+q.length);button.append(mark,document.createTextNode(snippet.slice(offset+q.length)+(end<r.text.length?'...':'')));button.onclick=()=>goTo(r.el);results.append(button);}},120);});
const zoom=document.getElementById('diagram-modal'),stage=document.getElementById('diagram-stage'),viewport=document.getElementById('diagram-viewport');let nativeWidth=800,scale=1;
function setScale(next){scale=Math.max(.15,Math.min(5,next));const svg=stage.querySelector('svg');if(svg){svg.style.width=nativeWidth*scale+'px';svg.style.height='auto';}document.getElementById('diagram-scale').textContent=Math.round(scale*100)+'%';}
document.querySelectorAll('.diagram-zoom').forEach(button=>button.onclick=()=>{const figure=button.closest('figure'),svg=figure.querySelector('svg').cloneNode(true);stage.replaceChildren(svg);nativeWidth=svg.viewBox.baseVal.width;document.getElementById('diagram-title').textContent=figure.dataset.title;zoom.showModal();setScale(Math.min(1,(viewport.clientWidth-60)/nativeWidth));viewport.scrollTop=0;viewport.scrollLeft=0;});
document.getElementById('close-diagram').onclick=()=>zoom.close();document.getElementById('zoom-in').onclick=()=>setScale(scale*1.25);document.getElementById('zoom-out').onclick=()=>setScale(scale/1.25);document.getElementById('zoom-fit').onclick=()=>{setScale(Math.min((viewport.clientWidth-60)/nativeWidth,(viewport.clientHeight-50)/stage.querySelector('svg').viewBox.baseVal.height));viewport.scrollTop=0;viewport.scrollLeft=0;};
viewport.addEventListener('wheel',e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();setScale(scale*(e.deltaY<0?1.1:1/1.1));}},{passive:false});
let drag=null;viewport.addEventListener('pointerdown',e=>{if(e.button!==0)return;drag={x:e.clientX,y:e.clientY,left:viewport.scrollLeft,top:viewport.scrollTop};viewport.setPointerCapture(e.pointerId);viewport.style.cursor='grabbing';});viewport.addEventListener('pointermove',e=>{if(drag){viewport.scrollLeft=drag.left+drag.x-e.clientX;viewport.scrollTop=drag.top+drag.y-e.clientY;}});for(const event of ['pointerup','pointercancel'])viewport.addEventListener(event,()=>{drag=null;viewport.style.cursor='grab';});
document.addEventListener('click',e=>{const a=e.target.closest('a[href^="#"]');if(!a)return;const target=document.getElementById(decodeURIComponent(a.hash.slice(1)));if(target){let p=target.parentElement;while(p){if(p.tagName==='DETAILS')p.open=true;p=p.parentElement;}}});
'''

def html_export():
    original_fence=PARSER.renderer.rules.get('fence')
    def fence(tokens,idx,options,env):
        t=tokens[idx]
        if t.info.strip()=='mermaid':
            id=diagram_id(t.content);d=DIAGRAMS[id]
            svg=(ASSETS/(id+'.svg')).read_text(encoding='utf-8')
            return '<figure class="diagram" data-diagram="'+id+'" data-title="'+html.escape(d['title'],quote=True)+'"><button class="diagram-zoom" type="button" aria-label="Zoom '+html.escape(d['title'],quote=True)+'">Open diagram / zoom</button><div class="diagram-canvas" role="img" aria-label="'+html.escape(d['title'],quote=True)+'">'+svg+'</div><figcaption>'+html.escape(d['title'])+' <span class="diagram-hint">Use Open diagram for pan, fit and zoom.</span></figcaption></figure>\n'
        try:lexer=get_lexer_by_name(t.info.strip() or 'text')
        except Exception:lexer=TextLexer()
        highlighted=highlight(t.content,lexer,HtmlFormatter(nowrap=True))
        return '<div class="code-wrap"><button class="code-copy" type="button">Copy</button><pre class="highlight"><code>'+highlighted+'</code></pre></div>\n'
    PARSER.renderer.rules['fence']=fence
    rendered=PARSER.renderer.render(TOKENS,PARSER.options,{})
    rendered=rendered.replace('<table>','<div class="table-wrap"><table>').replace('</table>','</table></div>')
    def embed(match):
        href=html.unescape(match.group(1));p=DOC/unquote(href.split('#')[0])
        if href.startswith('assets/architecture/') and p.suffix in ('.svg','.png'):
            return 'href="'+data_url(p)+'" download="'+p.name+'"'
        return match.group(0)
    rendered=re.sub(r'href="([^"]+)"',embed,rendered)
    navigation=''.join('<a href="#'+id+'"><span>'+str(i).zfill(2)+'</span>'+html.escape(title.partition('. ')[2])+'</a>' for i,(id,title) in enumerate(CHAPTERS,1))
    style=CSS+'\n'+HtmlFormatter().get_style_defs('.highlight')
    page='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="color-scheme" content="light"><title>FlightReady - Complete Visual Codebase Guide</title><style>'''+style+'''</style></head><body><div id="reading-progress"></div><aside class="sidebar" id="sidebar"><div class="brand">FlightReady<small>THE VISUAL SOURCEBOOK</small></div><nav aria-label="Chapter navigation">'''+navigation+'''<a href="#verification-map"><span>+</span>Source verification / diagrams</a></nav><div class="sidebar-footer">A complete source-derived textbook.<br>Offline HTML edition · 17 chapters<br><a href="#contents">Table of contents</a> · <a href="#understand">Question map</a><br><button id="expand-all">Expand explanations</button> <button id="collapse-all">Collapse</button></div></aside><div class="main"><header class="topbar"><span>CODEBASE / VISUAL FIELD GUIDE</span><div><button class="mobile-menu" id="menu-toggle" aria-label="Toggle chapter navigation">Chapters</button><button class="search-trigger" id="open-search">Search the entire book <small>Ctrl K</small></button><button onclick="window.print()">Print</button></div></header><article class="book" id="book">'''+rendered+'''</article></div><dialog id="search-modal" aria-label="Search documentation"><header><strong>Search the complete sourcebook</strong><button id="close-search" aria-label="Close search">Close</button></header><input id="search-input" type="search" aria-label="Search documentation text" placeholder="Try aggregate_quality, checkpoint or missing evidence"><div id="search-count">Type at least two characters. All chapters and symbol references are included.</div><div id="search-results"></div></dialog><dialog id="diagram-modal" aria-label="Zoomable architecture diagram"><header><strong id="diagram-title"></strong><div id="diagram-toolbar"><button id="zoom-out" aria-label="Zoom out">-</button><span id="diagram-scale">100%</span><button id="zoom-in" aria-label="Zoom in">+</button><button id="zoom-fit">Fit</button><button id="close-diagram">Close</button></div></header><div id="diagram-viewport"><div id="diagram-stage"></div></div></dialog><script>'''+JS+'''</script></body></html>'''
    (DOC/'CODEBASE_VISUAL_GUIDE.html').write_text(page,encoding='utf-8')
    return {'bytes':len(page.encode()),'embedded_svg':len(DIAGRAMS)}

INK=colors.HexColor('#243c30');GREEN=colors.HexColor('#2b664c');DIM=colors.HexColor('#65756a');LINE=colors.HexColor('#d9e3d5')
for name,file in [('Body','segoeui.ttf'),('BodyBold','segoeuib.ttf'),('BodyItalic','segoeuii.ttf'),('Mono','consola.ttf'),('Display','georgia.ttf'),('DisplayBold','georgiab.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'C:/Windows/Fonts/'+file))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='BodyBold',italic='BodyItalic',boldItalic='BodyBold')
pdfmetrics.registerFontFamily('Display',normal='Display',bold='DisplayBold',italic='Display',boldItalic='DisplayBold')
STYLES={
 'body':ParagraphStyle('body',fontName='Body',fontSize=10.2,leading=15.1,textColor=INK,spaceAfter=9,allowWidows=0,allowOrphans=0),
 'h1':ParagraphStyle('h1',fontName='Display',fontSize=24,leading=31,textColor=GREEN,spaceBefore=5,spaceAfter=19,keepWithNext=True),
 'h2':ParagraphStyle('h2',fontName='BodyBold',fontSize=15,leading=20,textColor=GREEN,spaceBefore=18,spaceAfter=10,keepWithNext=True),
 'h3':ParagraphStyle('h3',fontName='BodyBold',fontSize=11.5,leading=16,textColor=INK,spaceBefore=13,spaceAfter=8,keepWithNext=True),
 'cover':ParagraphStyle('cover',fontName='Display',fontSize=37,leading=45,textColor=GREEN,spaceBefore=50,spaceAfter=30),
 'cell':ParagraphStyle('cell',fontName='Body',fontSize=8.3,leading=11.9,textColor=INK,splitLongWords=True,allowWidows=1,allowOrphans=1),
 'th':ParagraphStyle('th',fontName='BodyBold',fontSize=8.3,leading=11.8,textColor=GREEN),
 'code':ParagraphStyle('code',fontName='Mono',fontSize=8.1,leading=11.2,textColor=colors.HexColor('#294336'),backColor=colors.HexColor('#f2f6ee'),borderColor=LINE,borderWidth=.5,borderPadding=11,spaceBefore=6,spaceAfter=14),
 'aside':ParagraphStyle('aside',fontName='BodyBold',fontSize=10.3,leading=14.5,textColor=GREEN,backColor=colors.HexColor('#edf3e6'),borderPadding=9,spaceBefore=13,spaceAfter=10,keepWithNext=True),
 'caption':ParagraphStyle('caption',fontName='Body',fontSize=9,leading=13,textColor=DIM,spaceBefore=9,spaceAfter=7),
 'sources':ParagraphStyle('sources',fontName='Body',fontSize=8.1,leading=11.3,textColor=DIM,spaceAfter=6),
 'toc':ParagraphStyle('toc',fontName='Body',fontSize=10.2,leading=17,textColor=INK,spaceBefore=7,leftIndent=14,firstLineIndent=-14,rightIndent=30),
}

def pdf_href(href):
    if href.startswith('#'):return href
    if href.startswith(('https:','http:','mailto:')):return href
    file,sep,fragment=href.partition('#')
    resolved=(DOC/unquote(file)).resolve()
    return resolved.as_uri()+('#'+fragment if sep else '')

def inline_pdf(children,cell=False):
    out=[]
    for t in children or []:
        if t.type=='text':out.append(html.escape(t.content))
        elif t.type=='code_inline':out.append('<font name="Mono" size="'+('8.0' if cell else '9')+'">'+html.escape(t.content)+'</font>')
        elif t.type=='strong_open':out.append('<b>')
        elif t.type=='strong_close':out.append('</b>')
        elif t.type=='em_open':out.append('<i>')
        elif t.type=='em_close':out.append('</i>')
        elif t.type=='link_open':out.append('<link href="'+html.escape(pdf_href(t.attrGet('href')),quote=True)+'" color="#2b664c">')
        elif t.type=='link_close':out.append('</link>')
        elif t.type in ('softbreak','hardbreak'):out.append(' ' if t.type=='softbreak' else '<br/>')
        elif t.type=='html_inline':
            if t.content.lower() in ('<br>','<br/>','<br />'):out.append('<br/>')
            else:out.append(html.escape(re.sub('<[^>]+>','',t.content)))
        elif t.type=='image':out.append(html.escape(t.content))
        else:out.append(html.escape(t.content))
    return ''.join(out)

def plain_inline(children):
    return ''.join(t.content if t.type in ('text','code_inline','image') else ' ' if t.type in ('softbreak','hardbreak') else '' for t in children or [])

class Anchor(Flowable):
    def __init__(self,id):super().__init__();self.id=id;self.width=0;self.height=0
    def draw(self):self.canv.bookmarkHorizontalAbsolute(self.id,self.canv._doctemplate.frame._y)

class BookDoc(BaseDocTemplate):
    def __init__(self,path,**kwargs):
        super().__init__(str(path),pagesize=A4,leftMargin=43,rightMargin=43,topMargin=50,bottomMargin=47,**kwargs)
        self.chapter='Visual sourcebook';self.figure_pages=[]
        templates=[]
        for id,size in [('portrait',A4),('landscape',landscape(A4)),('large-portrait',A3),('large-landscape',landscape(A3))]:
            w,h=size;frame=Frame(43,47,w-86,h-97,leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0,id=id+'-frame')
            templates.append(PageTemplate(id=id,frames=[frame],pagesize=size,onPage=self.decoration))
        self.addPageTemplates(templates)
    def beforeDocument(self):
        self.chapter='Visual sourcebook'
    def decoration(self,canvas,doc):
        w,h=canvas._pagesize
        canvas.saveState();canvas.setTitle('FlightReady - Complete Visual Codebase Guide');canvas.setAuthor('Source-derived project documentation')
        if doc.page==1:
            canvas.setFillColor(colors.HexColor('#edf3e5'));canvas.rect(0,h-118,w,118,fill=1,stroke=0)
            canvas.setFillColor(GREEN);canvas.setFont('BodyBold',10);canvas.drawString(43,h-50,'FLIGHTREADY / THE VISUAL SOURCEBOOK')
            canvas.setFont('Body',8);canvas.setFillColor(DIM);canvas.drawString(43,h-70,'Architecture, data lineage and the actual code behind the application')
        else:
            canvas.setFont('Body',7.8);canvas.setFillColor(DIM);canvas.drawString(43,h-29,'FLIGHTREADY  |  CODEBASE VISUAL GUIDE')
            canvas.drawRightString(w-43,h-29,self.chapter[:86])
            canvas.setStrokeColor(LINE);canvas.line(43,h-37,w-43,h-37)
        canvas.setStrokeColor(LINE);canvas.line(43,34,w-43,34)
        canvas.setFont('Body',7.7);canvas.setFillColor(DIM);canvas.drawString(43,22,'Source snapshot: '+SNAPSHOT+'  |  Complete offline textbook')
        canvas.drawRightString(w-43,22,str(doc.page));canvas.restoreState()
    def afterFlowable(self,flowable):
        if hasattr(flowable,'_book_heading'):
            level,title,key=flowable._book_heading
            self.canv.bookmarkPage(key)
            if level==1 and (re.match(r'^\d+\.',title) or title.startswith('Source verification')):
                self.chapter=title
                if self.page>1:
                    w,h=self.canv._pagesize
                    self.canv.saveState();self.canv.setFillColor(colors.white);self.canv.rect(43,h-36,w-86,18,fill=1,stroke=0)
                    self.canv.setFont('Body',7.8);self.canv.setFillColor(DIM);self.canv.drawString(43,h-29,'FLIGHTREADY  |  CODEBASE VISUAL GUIDE');self.canv.drawRightString(w-43,h-29,self.chapter[:86]);self.canv.restoreState()
                self.notify('TOCEntry',(0,title,self.page,key))
                self.canv.addOutlineEntry(title,key,level=0,closed=False)
            elif level==2 and self.page>2:
                try:self.canv.addOutlineEntry(title,key,level=1,closed=True)
                except ValueError:pass

def pdf_export():
    story=[];expected=[];pending_anchors=[];figure_heading=None;in_toc=False;skip_toc_list=False;list_stack=[];pending_bullet=False;diagram_layouts=[];after_figure=False
    doc=BookDoc(DOC/'CODEBASE_VISUAL_GUIDE.pdf',allowSplitting=True)
    toc=TableOfContents();toc.levelStyles=[STYLES['toc']]
    width=A4[0]-86
    def remember(text):
        if text.strip():expected.append(text.strip())
    def flush():
        while pending_anchors:story.append(Anchor(pending_anchors.pop(0)))
    def regular():
        nonlocal after_figure
        if after_figure:
            story.extend([NextPageTemplate('portrait'),PageBreak()]);after_figure=False
    i=0
    while i<len(TOKENS):
        t=TOKENS[i]
        if t.type=='html_block':
            for id in re.findall(r'<a id="([^"]+)"></a>',t.content):pending_anchors.append(id)
            summary=re.search(r'<summary>(.*?)</summary>',t.content,re.S)
            if summary:
                regular();flush();label=re.sub('<[^>]+>','',summary[1]);remember(label);story.append(Paragraph(html.escape(label),STYLES['aside']))
            i+=1;continue
        if t.type=='heading_open':
            title=TOKENS[i+1].content;level=int(t.tag[1]);key=HEADING_IDS[i];remember(plain_inline(TOKENS[i+1].children))
            if title.startswith('Figure: '):figure_heading=(title,key);i+=3;continue
            if level==1 and i>0:
                story.extend([NextPageTemplate('portrait'),PageBreak()]);after_figure=False
            else:regular()
            if title=='Table of contents':
                story.append(PageBreak());in_toc=True
            flush()
            style=STYLES['cover'] if i==0 else STYLES['h'+str(min(level,3))]
            if i==0:story.append(Spacer(1,60))
            p=Paragraph(inline_pdf(TOKENS[i+1].children),style);p._book_heading=(level,title,key);story.append(p)
            i+=3;continue
        if t.type=='bullet_list_open' and in_toc:
            depth=1;i+=1
            while i<len(TOKENS) and depth:
                if TOKENS[i].type=='bullet_list_open':depth+=1
                elif TOKENS[i].type=='bullet_list_close':depth-=1
                elif TOKENS[i].type=='inline':remember(plain_inline(TOKENS[i].children))
                i+=1
            story.append(toc);in_toc=False;continue
        if t.type in ('bullet_list_open','ordered_list_open'):
            list_stack.append({'ordered':t.type=='ordered_list_open','number':int(t.attrGet('start') or 1)});i+=1;continue
        if t.type in ('bullet_list_close','ordered_list_close'):
            if list_stack:list_stack.pop()
            i+=1;continue
        if t.type=='list_item_open':pending_bullet=True;i+=1;continue
        if t.type=='paragraph_open':
            inline=TOKENS[i+1]
            pending_anchors.extend(re.findall(r'<a id="([^"]+)"></a>',inline.content))
            if not plain_inline(inline.children).strip() and '<a id=' in inline.content:
                i+=3;continue
            remember(plain_inline(inline.children));regular();flush();style=STYLES['body'];bullet=None
            if list_stack:
                style=ParagraphStyle('list-'+str(len(list_stack)),parent=STYLES['body'],leftIndent=18*len(list_stack),firstLineIndent=0,bulletIndent=18*(len(list_stack)-1),spaceAfter=6)
                if pending_bullet:
                    current=list_stack[-1];bullet=str(current['number'])+'.' if current['ordered'] else '\u2022'
                    if current['ordered']:current['number']+=1
                    pending_bullet=False
            story.append(Paragraph(inline_pdf(inline.children),style,bulletText=bullet));i+=3;continue
        if t.type=='fence':
            if t.info.strip()=='mermaid':
                id=diagram_id(t.content);d=DIAGRAMS[id];geometry=RENDERS[id]
                title,key=figure_heading or ('Figure: '+d['title'],'figure-'+id)
                caption='';sources='';extras=[];j=i+1
                if j+2<len(TOKENS) and TOKENS[j].type=='paragraph_open':
                    caption=inline_pdf(TOKENS[j+1].children);remember(plain_inline(TOKENS[j+1].children));j+=3
                if j+2<len(TOKENS) and TOKENS[j].type=='paragraph_open':
                    sources=inline_pdf(TOKENS[j+1].children);remember(plain_inline(TOKENS[j+1].children));j+=3
                while j+2<len(TOKENS) and TOKENS[j].type=='paragraph_open' and plain_inline(TOKENS[j+1].children).startswith(('Chapter evidence:','Back to contents')):
                    extras.append(inline_pdf(TOKENS[j+1].children));remember(plain_inline(TOKENS[j+1].children));j+=3
                choices=[]
                for template,size in [('portrait',A4),('landscape',landscape(A4)),('large-portrait',A3),('large-landscape',landscape(A3))]:
                    w,h=size;fw=w-86;fh=h-97
                    title_p=Paragraph(html.escape(title),STYLES['h2']);cap_p=Paragraph(caption,STYLES['caption']);src_p=Paragraph(sources,STYLES['sources'])
                    extra_p=[Paragraph(value,STYLES['sources']) for value in extras]
                    consumed=sum(p.wrap(fw,fh)[1] for p in (title_p,cap_p,src_p,*extra_p))+43+6*len(extra_p)
                    scale=min(fw/geometry['width'],(fh-consumed)/geometry['height'])
                    minimum=geometry['minFont']*scale
                    choices.append((minimum>=8.8,w*h,template,size,scale,minimum,title_p,cap_p,src_p,extra_p))
                valid=[c for c in choices if c[0]]
                if not valid:raise ValueError('Diagram cannot remain readable at A3: '+id)
                chosen=min(valid,key=lambda c:c[1]);_,_,template,size,scale,minimum,title_p,cap_p,src_p,extra_p=chosen
                story.extend([NextPageTemplate(template),PageBreak()]);flush();title_p._book_heading=(3,title,key);story.append(title_p);story.append(Spacer(1,7))
                image=Image(str(ASSETS/(id+'.png')),width=geometry['width']*scale,height=geometry['height']*scale);image.hAlign='CENTER';story.append(image)
                story.append(cap_p);story.append(src_p)
                story.extend(extra_p)
                diagram_layouts.append({'id':id,'template':template,'minimum_estimated_label_pt':minimum,'image_width_pt':geometry['width']*scale,'image_height_pt':geometry['height']*scale})
                after_figure=True;figure_heading=None;i=j;continue
            regular();flush();remember(t.content)
            maxlen=94
            story.append(Preformatted(t.content.rstrip('\n'),STYLES['code'],dedent=0,maxLineLength=maxlen,splitChars=' /,|',newLineChars='  '));i+=1;continue
        if t.type=='table_open':
            rows=[];current=[];head=True;i+=1
            while i<len(TOKENS) and TOKENS[i].type!='table_close':
                token=TOKENS[i]
                if token.type=='tr_open':current=[]
                elif token.type in ('th_open','td_open'):
                    inline=TOKENS[i+1];remember(plain_inline(inline.children));current.append(Paragraph(inline_pdf(inline.children,cell=True),STYLES['th'] if token.type=='th_open' else STYLES['cell']))
                elif token.type=='tr_close':rows.append(current)
                i+=1
            regular();flush()
            if rows:
                columns=len(rows[0]);ratios={2:[.30,.70],3:[.25,.32,.43],4:[.20,.31,.27,.22],6:[.12,.16,.16,.20,.20,.16]}.get(columns,[1/columns]*columns)
                table=LongTable(rows,colWidths=[width*r for r in ratios],repeatRows=1,hAlign='LEFT',splitByRow=1,splitInRow=0)
                table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e9f1e3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#bfcfb5')),('LINEBELOW',(0,1),(-1,-1),.3,LINE),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f7faf3')])]))
                story.append(table);story.append(Spacer(1,12))
            i+=1;continue
        if t.type=='hr' and not after_figure:story.append(Spacer(1,8))
        i+=1
    flush()
    doc.multiBuild(story)
    result={'expected_text_blocks':len(expected),'expected_text':expected,'diagrams':diagram_layouts,'markdown_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
    (ASSETS/'export-content-manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return {'expected_text_blocks':len(expected),'diagram_pages':len(diagram_layouts),'pdf_bytes':(DOC/'CODEBASE_VISUAL_GUIDE.pdf').stat().st_size,'minimum_diagram_label_pt':min(d['minimum_estimated_label_pt'] for d in diagram_layouts)}

if __name__=='__main__':
    print('HTML '+json.dumps(html_export()),flush=True)
    print('PDF '+json.dumps(pdf_export()),flush=True)
