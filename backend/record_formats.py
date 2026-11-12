"""Streaming adapters for common downloaded text/record datasets.

Adapters preserve record boundaries and source fields. Nested values are stored
as JSON text with an explicit schema note; arrays are never silently exploded.
XML entities/DTDs and spreadsheet formulas are never executed.
"""
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any,cast
import csv
import io
import json
import math
import os
import re
from numbers import Number
from decimal import Decimal
from xml.parsers import expat
import ijson
from defusedxml import ElementTree as SafeXML
from openpyxl import load_workbook
from openpyxl.worksheet.worksheet import Worksheet
from backend.ingestion import IngestionError,SelectionRequired

TEXT_SUFFIXES={'.txt','.text','.md','.rst','.log'}
RECORD_SUFFIXES={'.csv','.tsv','.tab','.psv','.data','.dat','.json','.jsonl','.ndjson','.xml','.xlsx','.parquet','.feather','.arrow'}|TEXT_SUFFIXES
MAX_RECORD=8*1024**2
MAX_FIELDS=512

@contextmanager
def json_input(path,encoding):
    """Give ijson bytes; transcode legacy/BOM encodings incrementally."""
    if encoding in ('utf-8','utf-8-sig'):
        with path.open('rb') as source:
            prefix=source.read(3)
            source.seek(3 if prefix==b'\xef\xbb\xbf' else 0)
            yield source
    else:
        class UTF8Reader:
            def __init__(self,source):self.source=source
            def read(self,size=-1):
                return self.source.read(size if size>=0 else 65536).encode('utf-8')
        with path.open('r',encoding=encoding,errors='strict') as source:yield UTF8Reader(source)

def delimiter_for(sample:str,metadata:dict,suffix='.csv'):
    declared=metadata.get('delimiter')
    if declared is not None:
        declared='\t' if declared in ('tab','\\t') else declared
        if not isinstance(declared,str) or len(declared)!=1 or declared in '\r\n\x00':
            raise IngestionError('delimiter must be a single character, for example comma, tab, semicolon or pipe')
        return declared,'user_declared'
    if suffix in ('.tsv','.tab'):return '\t','extension'
    if suffix=='.psv':return '|','extension'
    try:
        dialect=csv.Sniffer().sniff(sample,delimiters=',\t;|')
        return dialect.delimiter,'detected'
    except csv.Error:
        first=next((line for line in sample.splitlines() if line.strip()),'')
        candidates=[]
        for delimiter in ',\t;|':
            try:
                if len(next(csv.reader([first],delimiter=delimiter)))>1:candidates.append(delimiter)
            except (StopIteration,csv.Error):pass
        if len(candidates)==1:return candidates[0],'header_detected'
        return ',','default'

def _json_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise IngestionError('Duplicate JSON key: '+key)
        result[key]=value
    return result

def _invalid_constant(value):raise IngestionError('Nonstandard JSON numeric constant: '+value)

def canonical_json(value):
    if isinstance(value,Decimal):
        if not value.is_finite():raise IngestionError('Non-finite JSON number')
        return str(value)
    if isinstance(value,dict):return '{'+','.join(json.dumps(k,ensure_ascii=False)+':'+canonical_json(v) for k,v in sorted(value.items()))+'}'
    if isinstance(value,list):return '['+','.join(canonical_json(v) for v in value)+']'
    return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))

def normalize_record(value,parsing):
    record=value if isinstance(value,dict) else {'text':value} if isinstance(value,str) else {'value':value}
    if not record: record={'value':None}
    if len(record)>MAX_FIELDS:raise IngestionError('A record exceeds the 512-field limit')
    result={}
    for key,item in record.items():
        if not isinstance(key,str) or not key.strip():raise IngestionError('Record field names must be nonempty strings')
        kind='null' if item is None else 'boolean' if isinstance(item,bool) else 'number' if isinstance(item,Number) else 'string' if isinstance(item,str) else 'structured' if isinstance(item,(dict,list)) else 'other'
        types=parsing.setdefault('source_type_counts',{}).setdefault(key,{})
        types[kind]=types.get(kind,0)+1
        if isinstance(item,int) and not isinstance(item,bool) and not -(2**63)<=item<2**63:
            if key not in parsing.setdefault('exact_integer_fields',[]):parsing['exact_integer_fields'].append(key)
        if isinstance(item,(dict,list)):
            item=canonical_json(item)
            if key not in parsing['json_encoded_fields']:parsing['json_encoded_fields'].append(key)
        if isinstance(item,float) and not math.isfinite(item):raise IngestionError('Non-finite JSON values are not valid records')
        result[key]=item
    if len(json.dumps(result,ensure_ascii=False,default=str).encode('utf-8'))>MAX_RECORD:
        raise IngestionError('A structured record exceeds the 8 MiB record limit')
    return result

def _preflight_json(path,encoding,record_path=None):
    """Bound nesting and string tokens before ijson materializes a value."""
    depth=length=position=0
    quoted=escaped=False
    root=None;bounded=[];containers=[]
    record_depth=len(record_path.split('.'))+2 if record_path else None
    with path.open('r',encoding=encoding,errors='strict') as source:
        while chunk:=source.read(1024**2):
            for char in chunk:
                position+=1
                if bounded and position-bounded[0]>MAX_RECORD:raise IngestionError('A JSON record container exceeds the 8 MiB record limit')
                if quoted:
                    length+=1
                    if length>MAX_RECORD:raise IngestionError('A JSON string exceeds the 8 MiB token limit')
                    if escaped:escaped=False
                    elif char=='\\':escaped=True
                    elif char=='"':quoted=False
                elif char=='"':quoted=True;length=0
                elif char in '[{':
                    if root is None:root=char
                    depth+=1
                    if depth>64:raise IngestionError('JSON nesting exceeds 64 levels')
                    limit_from=record_depth or (2 if root=='[' else 3)
                    limited=depth>=limit_from
                    containers.append(limited)
                    if limited:bounded.append(position)
                elif char in ']}':
                    if containers and containers.pop():bounded.pop()
                    depth-=1

def json_keys(path,encoding='utf-8'):
    arrays=[];root=None;objects=[]
    with json_input(path,encoding) as source:
        for prefix,event,value in ijson.parse(source):
            if root is None:root=event
            if event=='start_map':objects.append(set())
            elif event=='end_map':objects.pop()
            elif event=='map_key':
                if value in objects[-1]:raise IngestionError('Duplicate JSON key: '+value)
                objects[-1].add(value)
                if len(objects[-1])>512:raise IngestionError('JSON object exceeds 512 fields')
            if event=='start_array' and (prefix=='' or '.' not in prefix):arrays.append(prefix)
    return root,arrays

def _json_records(path,metadata,parsing):
    encoding=parsing['encoding']
    _preflight_json(path,encoding,metadata.get('record_path'))
    root,arrays=json_keys(path,encoding)
    selected=metadata.get('record_path')
    if selected is None:
        if root=='start_array':selected=''
        elif len(arrays)==1:selected=arrays[0]
        elif len(arrays)>1:
            raise SelectionRequired('Choose the JSON array containing dataset records',[
                {'label':'JSON records: '+name,'options':{'record_path':name}} for name in arrays])
    if selected is not None:
        if not isinstance(selected,str):raise IngestionError('record_path must be a JSON field path')
        prefix=(selected+'.item') if selected else 'item'
        parsing['record_path']=selected
        def records():
            with json_input(path,encoding) as source:
                for item in ijson.items(source,prefix):yield normalize_record(item,parsing)
        return records()
    # A single JSON object is one record, with an explicit bounded read.
    if path.stat().st_size>MAX_RECORD:raise IngestionError('Large JSON objects need a record_path pointing to an array, or JSONL records')
    with path.open('r',encoding=encoding,errors='strict') as source:
        value=json.load(source,object_pairs_hook=_json_object,parse_constant=_invalid_constant)
    return iter([normalize_record(value,parsing)])

def local_tag(tag):return tag.rsplit('}',1)[-1]

def preflight_xml(path,selected=None):
    """SAX-sized validation bounds text/record growth before tree parsing."""
    parser=expat.ParserCreate(namespace_separator='}');stack:list[tuple[str,int]]=[];active=None;record_bytes=0
    def forbidden(*args):raise IngestionError('XML DTDs, entity declarations and external entities are not allowed')
    parser.StartDoctypeDeclHandler=forbidden
    parser.EntityDeclHandler=forbidden
    parser.ExternalEntityRefHandler=forbidden
    def start(name,attrs):
        nonlocal active,record_bytes
        if len(stack)>=64:raise IngestionError('XML nesting exceeds 64 levels')
        stack.append((name,0))
        size=len(name)+sum(len(k)+len(v) for k,v in attrs.items())+5
        current='/'.join(local_tag(frame[0]) for frame in stack)
        if selected and (current==selected or '/' not in selected and local_tag(name)==selected):
            if active is not None:raise IngestionError('Nested matching XML records are ambiguous; provide an exact record_path')
            active=len(stack);record_bytes=0
        if active is not None:
            record_bytes+=size
            if record_bytes>MAX_RECORD:raise IngestionError('An XML record exceeds 8 MiB')
    def text(value):
        nonlocal record_bytes
        size=len(value.encode('utf-8'))
        if stack:
            stack[-1]=(stack[-1][0],stack[-1][1]+size)
            if stack[-1][1]>MAX_RECORD:raise IngestionError('An XML text value exceeds 8 MiB')
        if active is not None:
            record_bytes+=size
            if record_bytes>MAX_RECORD:raise IngestionError('An XML record exceeds 8 MiB')
    def end(name):
        nonlocal active
        if active==len(stack):active=None
        stack.pop()
    parser.StartElementHandler=start;parser.EndElementHandler=end;parser.CharacterDataHandler=text
    with path.open('rb') as source:
        while chunk:=source.read(1024**2):parser.Parse(chunk,False)
        parser.Parse(b'',True)

def xml_candidates(path):
    preflight_xml(path)
    counts=Counter();stack=[]
    for event,element in SafeXML.iterparse(path,events=('start','end'),forbid_dtd=True,forbid_entities=True,forbid_external=True):
        if event=='start':
            if len(stack)>=64:raise IngestionError('XML nesting exceeds 64 levels')
            stack.append({'tag':local_tag(element.tag),'element':element,'has_children':False,'complex_child':False})
            if len(stack)>1:stack[-2]['has_children']=True
        else:
            current=stack[-1]
            if (current['has_children'] and not current['complex_child']) or element.attrib:
                counts['/'.join(x['tag'] for x in stack)]+=1
                if len(counts)>512:raise IngestionError('XML has too many distinct record paths; provide a simpler export')
            stack.pop()
            if stack:
                if current['has_children']:stack[-1]['complex_child']=True
                stack[-1]['element'].remove(element)
            element.clear()
    repeated=[name for name,count in counts.items() if count>1]
    return repeated or list(counts)

def xml_value(element):
    children=list(element)
    if not children and not element.attrib:return element.text.strip() if element.text and element.text.strip() else None
    result={'@'+local_tag(k):v for k,v in element.attrib.items()}
    if element.text and element.text.strip():result['#text']=element.text.strip()
    for child in children:
        key=local_tag(child.tag);value=xml_value(child)
        if key in result:
            if not isinstance(result[key],list):result[key]=[result[key]]
            result[key].append(value)
        else:result[key]=value
    return result

def _xml_records(path,metadata,parsing):
    selected=metadata.get('record_path')
    if not selected:
        choices=xml_candidates(path)
        if len(choices)>1:raise SelectionRequired('Choose the XML element representing one record',[
            {'label':'XML records: '+name,'options':{'record_path':name}} for name in choices])
        if not choices:raise IngestionError('XML contains no record elements; provide record_path or a tabular export')
        selected=choices[0]
    if not isinstance(selected,str):raise IngestionError('record_path must be an XML tag or slash-separated path')
    preflight_xml(path,selected)
    parsing['record_path']=selected
    def records():
        stack=[];active=None
        for event,element in SafeXML.iterparse(path,events=('start','end'),forbid_dtd=True,forbid_entities=True,forbid_external=True):
            if event=='start':
                stack.append(element)
                if len(stack)>64:raise IngestionError('XML nesting exceeds 64 levels')
                current='/'.join(local_tag(x.tag) for x in stack)
                if current==selected or ('/' not in selected and local_tag(element.tag)==selected):active=len(stack)
            else:
                if active==len(stack):
                    value=xml_value(element)
                    yield normalize_record(value,parsing)
                    active=None
                stack.pop()
                if active is None:
                    if stack:stack[-1].remove(element)
                    element.clear()
    return records()

def prepare_records(path:Path,metadata:dict):
    """Return a record iterator and its mutable, evidence-backed parsing notes."""
    from backend.large_ingestion import detect_encoding,BoundedLines,csv_batches
    suffix=path.suffix.lower()
    if metadata.get('record_path') is not None and not isinstance(metadata['record_path'],str):raise IngestionError('record_path must be text')
    parsing={'format':suffix.lstrip('.'),'warnings':[],'json_encoded_fields':[]}
    auto_delimiter=False
    if suffix not in ('.xlsx','.feather','.arrow'):
        parsing.update(detect_encoding(path,metadata))
    if suffix in TEXT_SUFFIXES and metadata.get('delimiter') is None and not metadata.get('text_unit') and metadata.get('format')!='text':
        with path.open('r',encoding=parsing['encoding'],errors='strict') as source:sample=source.read(65536)
        if '\t' in sample:
            try:
                rows=list(csv.reader(io.StringIO(sample),delimiter='\t'))[:20]
                if len(rows)>=2 and len(rows[0])>1 and all(len(row)==len(rows[0]) for row in rows if row):
                    metadata={**metadata,'delimiter':'\t'}
                    auto_delimiter=True
            except csv.Error:pass
    if suffix in ('.csv','.tsv','.tab','.psv','.data','.dat') or (suffix in TEXT_SUFFIXES and metadata.get('delimiter') is not None):
        generator=csv_batches(path,metadata)
        columns,info=next(generator);parsing.update(info)
        if auto_delimiter:parsing['delimiter_source']='detected_tabular_text'
        parsing['format']='delimited'
        def delimited_rows():
            try:
                for batch in generator:
                    for values in batch:yield dict(zip(columns,values))
            finally:generator.close()
        return delimited_rows(),parsing
    if suffix in TEXT_SUFFIXES:
        parsing.update(format='plain_text',record_unit=metadata.get('text_unit','line'))
        if parsing['record_unit']=='document':
            if path.stat().st_size>MAX_RECORD:raise IngestionError('A single document exceeds 8 MiB; use line records or an explicit preprocessing step')
            text=path.read_text(encoding=parsing['encoding'])
            return iter([{'text':text or None}]),parsing
        if parsing['record_unit']!='line':raise IngestionError('text_unit must be line or document')
        def text_lines():
            with path.open('r',encoding=parsing['encoding'],errors='strict',newline='') as source:
                bounded=BoundedLines(source)
                for line in bounded:
                    bounded.record_size=0
                    yield {'text':line.rstrip('\r\n') or None}
        return text_lines(),parsing
    if suffix in ('.jsonl','.ndjson'):
        parsing['format']='jsonl'
        def json_lines():
            with path.open('r',encoding=parsing['encoding'],errors='strict') as source:
                bounded=BoundedLines(source)
                for number,line in enumerate(bounded,1):
                    bounded.record_size=0
                    if not line.strip():raise IngestionError(f'JSONL line {number} is blank; remove it or explicitly preprocess the file')
                    try:value=json.loads(line,object_pairs_hook=_json_object,parse_constant=_invalid_constant)
                    except json.JSONDecodeError as exc:raise IngestionError(f'Invalid JSONL at line {number}: {exc.msg}') from exc
                    yield normalize_record(value,parsing)
        return json_lines(),parsing
    if suffix=='.json':return _json_records(path,metadata,parsing),parsing
    if suffix=='.xml':return _xml_records(path,metadata,parsing),parsing
    if suffix=='.xlsx':
        if path.stat().st_size>int(os.getenv('MAX_XLSX_MB','256'))*1024**2:
            raise IngestionError('Spreadsheet exceeds MAX_XLSX_MB; export very large sheets as CSV or Parquet')
        # defusedxml is installed so openpyxl's XML reader rejects entity attacks.
        workbook=load_workbook(path,read_only=True,data_only=False,keep_links=False)
        sheets=[sheet.title for sheet in workbook.worksheets]
        if not sheets:workbook.close();raise IngestionError('Spreadsheet has no tabular worksheets')
        selected=metadata.get('sheet_name')
        if selected is None and len(sheets)>1:
            workbook.close()
            raise SelectionRequired('Choose the spreadsheet sheet to assess',[{'label':'Sheet: '+s,'options':{'sheet_name':s}} for s in sheets])
        selected=selected or sheets[0]
        if selected not in sheets:workbook.close();raise IngestionError('Requested sheet_name is not present')
        parsing.update(sheet_name=selected,warnings=['Spreadsheet formulas are preserved as text and are never executed.'])
        def excel_rows():
            try:
                # Worksheet names exclude chartsheets; read-only worksheets
                # expose the same iter_rows API as a normal Worksheet.
                iterator=cast(Worksheet,workbook[selected]).iter_rows(values_only=True)
                first=next(iterator,None)
                if not first:raise IngestionError('Spreadsheet sheet is empty')
                header=metadata.get('has_header',True)
                if not isinstance(header,bool):raise IngestionError('has_header must be true or false')
                columns=[str(v) if v is not None else '' for v in first] if header else [f'column_{i+1}' for i in range(len(first))]
                if any(not c.strip() for c in columns) or len(set(columns))!=len(columns):raise IngestionError('Spreadsheet headers must be nonempty and unique')
                if not header:yield normalize_record(dict(zip(columns,first)),parsing)
                for row in iterator:yield normalize_record(dict(zip(columns,row)),parsing)
            finally:workbook.close()
        return excel_rows(),parsing
    if suffix in ('.feather','.arrow'):
        import pyarrow as pa
        def arrow_rows():
            with pa.memory_map(str(path),'r') as source:
                try:
                    reader=pa.ipc.open_file(source)
                    batches=(reader.get_batch(i) for i in range(reader.num_record_batches))
                except pa.ArrowInvalid:
                    source.seek(0);batches=iter(pa.ipc.open_stream(source))
                for batch in batches:
                    for row in batch.to_pylist():yield normalize_record(row,parsing)
        return arrow_rows(),parsing
    raise IngestionError('Unsupported record format: '+suffix)
