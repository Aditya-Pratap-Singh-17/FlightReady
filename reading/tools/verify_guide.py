"""Verify documentation completeness, local references, source immutability and PDF geometry."""
from pathlib import Path
import hashlib
import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from urllib.parse import unquote
import pymupdf
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'reading';ASSETS=DOC/'assets/architecture';QA=ASSETS/'review'
QA.mkdir(parents=True,exist_ok=True)

def canonical(text):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',text)).replace('\u00ad','').replace('\u200b','')

def verify():
    source=(DOC/'CODEBASE_VISUAL_GUIDE.md').read_text(encoding='utf-8')
    inv=json.loads((ASSETS/'source-inventory.json').read_text(encoding='utf-8'))
    manifest=json.loads((ASSETS/'guide-manifest.json').read_text(encoding='utf-8'))
    exported=json.loads((ASSETS/'export-content-manifest.json').read_text(encoding='utf-8'))
    rendered=json.loads((ASSETS/'render-manifest.json').read_text(encoding='utf-8'))
    changed=[p for p,h in inv['source_hashes'].items() if not (ROOT/p).exists() or hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    if changed:raise AssertionError('Application source changed: '+str(changed))
    links=re.findall(r'\]\(([^)]+)\)',source)
    missing=[];internal=[];source_links=0
    anchors=set(re.findall(r'<a id="([^"]+)"',source))
    for link in links:
        if link.startswith('#'):
            internal.append(link)
            if link[1:] not in anchors:missing.append(link)
        elif not link.startswith(('https:','http:','mailto:')):
            path,_,fragment=unquote(link).partition('#')
            target=(DOC/path).resolve()
            if not target.is_file():missing.append(link)
            elif fragment.startswith('L'):
                source_links+=1
                line=int(fragment[1:])
                if not 1<=line<=len(target.read_text(encoding='utf-8-sig').splitlines()):missing.append(link)
    if missing:raise AssertionError('Missing/invalid references '+str(missing[:20]))
    svg_errors=[]
    for d in rendered['diagrams']:
        path=ASSETS/(d['id']+'.svg')
        root=ET.fromstring(path.read_text(encoding='utf-8'))
        if not root.attrib.get('viewBox') or not d['parsed'] or not d['rendered']:svg_errors.append(d['id'])
        with Image.open(ASSETS/(d['id']+'.png')) as im:
            if im.width<d['width']*2.4 or im.height<d['height']*2.4:svg_errors.append(d['id']+' low resolution')
    if svg_errors:raise AssertionError(svg_errors)
    pdf=pymupdf.open(DOC/'CODEBASE_VISUAL_GUIDE.pdf')
    page_text=[p.get_text() for p in pdf]
    full=canonical('\n'.join(page_text))
    # Running furniture and repeated table heads interrupt paragraphs across
    # page boundaries in PDF extraction order. Keep a second body-only stream.
    body_parts=[]
    for page in pdf:
        for block in page.get_text('dict')['blocks']:
            if block['type']!=0:continue
            for line in block['lines']:
                for span in line['spans']:
                    repeated_head='Bold' in span['font'] and abs(span['size']-8.3)<.1
                    if span['bbox'][1]>43 and span['bbox'][3]<page.rect.height-38 and not repeated_head:
                        body_parts.append(span['text'])
    body=canonical(''.join(body_parts))
    missing_blocks=[]
    page_interrupted_blocks=[]
    for index,text in enumerate(exported['expected_text']):
        normalized=canonical(text)
        if normalized and normalized not in full:
            if normalized in body:page_interrupted_blocks.append(index)
            else:missing_blocks.append({'block':index,'text':text})
    clips=[];image_pages=[];internal_links=0;external_links=0;broken_dest=[];sparse=[]
    for n,page in enumerate(pdf):
        rect=page.rect
        if len(page_text[n].strip())<90:sparse.append(n+1)
        for b in page.get_text('dict')['blocks']:
            if b['type']!=0:continue
            for line in b['lines']:
                for span in line['spans']:
                    x0,y0,x1,y1=span['bbox']
                    if x0 < -1 or y0 < -1 or x1>rect.width+1 or y1>rect.height+1:
                        clips.append({'page':n+1,'text':span['text'][:100],'bbox':list(span['bbox'])})
        if page.get_images(full=True):
            image_pages.append(n+1)
            for item in page.get_images(full=True):
                for ir in page.get_image_rects(item[0]):
                    if not rect.contains(ir):clips.append({'page':n+1,'image':list(ir)})
        for link in page.get_links():
            if link['kind']==pymupdf.LINK_GOTO:
                internal_links+=1
                if not 0<=link.get('page',-1)<len(pdf):broken_dest.append({'page':n+1,'link':link})
            elif link['kind'] in (pymupdf.LINK_URI,pymupdf.LINK_LAUNCH,pymupdf.LINK_GOTOR):external_links+=1
    if len(image_pages)!=manifest['diagrams']:raise AssertionError(('Diagram page count',len(image_pages),manifest['diagrams']))
    if clips:raise AssertionError('Out-of-page PDF content: '+str(clips[:8]))
    if broken_dest:raise AssertionError('Broken PDF destinations')
    # Render every page. Contact sheets provide a complete visual audit index.
    thumbnails=[]
    for n,page in enumerate(pdf):
        pix=page.get_pixmap(matrix=pymupdf.Matrix(.48,.48),alpha=False)
        image=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        image.thumbnail((190,260))
        thumb=Image.new('RGB',(210,290),'#e6eade');thumb.paste(image,((210-image.width)//2,8))
        ImageDraw.Draw(thumb).text((10,272),f'Page {n+1}',fill='#284331')
        thumbnails.append(thumb)
    sheets=[]
    for start in range(0,len(thumbnails),24):
        batch=thumbnails[start:start+24];sheet=Image.new('RGB',(840,290*((len(batch)+3)//4)),'white')
        for i,thumb in enumerate(batch):sheet.paste(thumb,((i%4)*210,(i//4)*290))
        name=f'contact-{start//24+1:02}.png';sheet.save(QA/name);sheets.append(name)
    review_pages=sorted(set([0,1,2,8,len(pdf)//2,len(pdf)-1]+[n-1 for n in image_pages]))
    # Remove only obsolete generated previews, so the audit directory describes
    # this export rather than retaining pages from an earlier pagination pass.
    current_previews=set(sheets)|{f'page-{n+1:03}.png' for n in review_pages}
    for preview in QA.iterdir():
        if re.fullmatch(r'(?:page-\d{3}|contact-\d{2})\.png',preview.name) and preview.name not in current_previews:
            preview.unlink()
    for n in review_pages:
        pdf[n].get_pixmap(matrix=pymupdf.Matrix(1.3,1.3),alpha=False).save(QA/f'page-{n+1:03}.png')
    result={'markdown_sha256':hashlib.sha256((DOC/'CODEBASE_VISUAL_GUIDE.md').read_bytes()).hexdigest(),
        'source_files_hash_checked':len(inv['source_hashes']),'source_files_modified':changed,
        'markdown_internal_links_checked':len(internal),'source_line_links_checked':source_links,'missing_references':missing,
        'diagrams_parsed_and_rendered':len(rendered['diagrams']),'pdf_pages':len(pdf),'pdf_outline_entries':len(pdf.get_toc()),
        'pdf_internal_links':internal_links,'pdf_external_source_links':external_links,'invalid_pdf_destinations':broken_dest,
        'pdf_diagram_pages':image_pages,'minimum_estimated_diagram_label_pt':min(d['minimum_estimated_label_pt'] for d in exported['diagrams']),
        'expected_markdown_text_blocks':len(exported['expected_text']),'text_blocks_missing_from_pdf':missing_blocks,'blocks_verified_across_page_furniture':page_interrupted_blocks,
        'out_of_page_content':clips,'low_text_pages':sparse,'all_pages_rasterized':True,'contact_sheets':sheets,
        'html_bytes':(DOC/'CODEBASE_VISUAL_GUIDE.html').stat().st_size,'pdf_bytes':(DOC/'CODEBASE_VISUAL_GUIDE.pdf').stat().st_size}
    (ASSETS/'verification.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('pdf_diagram_pages','contact_sheets','text_blocks_missing_from_pdf')},ensure_ascii=True))
    if missing_blocks:
        print('Missing PDF blocks: '+str(len(missing_blocks)))
        for b in missing_blocks[:12]:print(json.dumps(b,ensure_ascii=True))
        raise AssertionError('PDF text accounting failed; inspect verification.json')
    return result

if __name__=='__main__':verify()
