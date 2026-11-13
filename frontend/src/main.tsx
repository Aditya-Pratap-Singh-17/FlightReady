import React, { useEffect, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';
import { useAssessment, hasReport, type Job } from './useAssessment';
import { Workbench } from './Workbench';
import {ExperimentalScore,type WeightedScore} from './ExperimentalScore';

type Goal = { task_type:string; task_description:string; prediction_target:string; required_inputs:string[]; optional_inputs:string[]; required_classes:string[]; [key:string]:unknown };
type Context = { original_text:string; selected_tags:Record<string,string[]>; goal:Goal; clarification_questions:string[]; confirmed:boolean; provider:string; [key:string]:unknown };
type Metric = {metric_id:string; dimension:string; state:string; assessment_status?:string; value:unknown; reason:string; evidence:unknown};
type Finding = {requirement:{requirement_id:string;description:string;severity:string};state:string;reason:string;result:Metric};
type Certificate = {overall_status:string;executive_summary:string;disclaimer:string;findings:Finding[];metrics:Metric[];additional_benefits:string[];limitations:string[];recommendations:{description:string}[];reproducibility:unknown;learning_views?:Record<string,unknown>;experimental_score?:WeightedScore|null};
const TAGS:Record<string,string[]> = {
  Task:['Track objects','Detect objects','Classify images','Classify text','Predict a number','Detect anomalies','Analyze flight data'],
  'Data type':['Images','Text','Tabular data','GPS','IMU','Multisensor data'],
  'Operating environment':['Clear','Cloudy','Foggy','Rainy','Snowy','Nadir','Oblique','Daytime','Nighttime','Urban','Rural','High altitude','Low altitude','Indoor','Outdoor'],
  Priorities:['High accuracy','Minimize missed detections','Rare-event detection','Reliable across locations','Low annotation cost'],
  'Data concerns':['Missing values','Incorrect labels','Duplicate records','Imbalanced classes','Poor coverage','Uncertain data quality']
};
const DIMENSIONS = ['integrity','validity','duplication','label_quality','coverage','representativeness','leakage','evidence_reliability'];
const UPLOAD_TYPES='.csv,.tsv,.tab,.psv,.txt,.text,.data,.dat,.json,.jsonl,.ndjson,.xml,.xlsx,.parquet,.feather,.arrow,.md,.rst,.log,.zip,.tar,.tgz,.gz,.bz2,.xz,.tbz2,.txz,.jpg,.jpeg,.png,.webp,.bmp,.tif,.tiff,.gif,.jfif,.ppm,.pgm,.pnm,.avif';
type DatasetChoice={label:string;options:Record<string,unknown>};
const label = (s:string) => s.replaceAll('_',' ').replace(/^./,c=>c.toUpperCase());

async function api<T>(path:string, options:RequestInit = {}):Promise<T> {
  const response = await fetch('/api'+path, options);
  if (!response.ok) {
    const data = await response.json().catch(()=>({detail:'Server unavailable'}));
    throw Object.assign(new Error(typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail)),{status:response.status});
  }
  return response.json() as Promise<T>;
}
const jsonPost = (value:unknown):RequestInit => ({method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(value)});

function ListInput({values,onCommit}:{values:string[];onCommit:(values:string[])=>void}){
  // Keep the raw draft while typing so a trailing comma is not immediately
  // removed by the controlled input. Commit the parsed list when leaving it.
  const [draft,setDraft]=useState(values.join(', '));
  const serialized=JSON.stringify(values);
  useEffect(()=>setDraft(values.join(', ')),[serialized]);
  return <input value={draft} onChange={e=>setDraft(e.target.value)} onBlur={()=>onCommit(draft.split(',').map(x=>x.trim()).filter(Boolean))}/>;
}

export function App(){
  const [text,setText] = useState('');
  const [tags,setTags] = useState<Record<string,string[]>>({});
  const [context,setContext] = useState<Context|null>(null);
  const [interpretation,setInterpretation] = useState<string>();
  const [advanced,setAdvanced] = useState('');
  const [metadata,setMetadata] = useState('{}');
  const [parseOptions,setParseOptions]=useState<Record<string,string>>({});
  const [selectedChoice,setSelectedChoice]=useState('0');
  const [dataset,setDataset] = useState<{id:string;name:string;profile:Record<string,unknown>}|null>(null);
  const [reference,setReference] = useState<string>();
  const {job,setJob,certificate,setCertificate,reportLoading,reportError,progressError,retryReport,retryProgress}=useAssessment<Certificate>(api);
  const [busy,setBusy] = useState('');
  const [error,setError] = useState('');
  const [dimension,setDimension] = useState('all');
  const [uploadPercent,setUploadPercent] = useState<number|null>(null);
  const [limits,setLimits] = useState({max_upload_bytes:2*1024**3,max_image_upload_bytes:2*1024**3});
  const activeUpload=useRef<XMLHttpRequest|null>(null);
  useEffect(()=>{api<typeof limits>('/config').then(setLimits).catch(()=>{});return ()=>activeUpload.current?.abort();},[]);

  const resetConfirmation = (next:Context) => {setContext({...next,confirmed:false}); setCertificate(null);};
  function updateGoal(field:string,value:unknown){if(context) resetConfirmation({...context,goal:{...context.goal,[field]:value}});}
  async function interpret(){
    setBusy('interpret');setError('');setJob(null);setCertificate(null);
    try {
      const value = await api<{id:string;context:Context}>('/context/interpret',jsonPost({text,tags}));
      setContext(value.context);setInterpretation(value.id);setAdvanced(JSON.stringify(value.context.goal,null,2));
    }catch(e){setError((e as Error).message);}finally{setBusy('');}
  }
  async function upload(file:File|undefined,isReference=false){
    if(!file)return;
    const maximum=file.name.toLowerCase().endsWith('.zip')?Math.min(limits.max_image_upload_bytes,limits.max_upload_bytes):limits.max_upload_bytes;
    if(file.size>maximum){setError('This file exceeds the server limit of '+(maximum/1024**2).toLocaleString()+' MiB for this format.');return;}
    setBusy('upload');setError('');setCertificate(null);setJob(null);
    setUploadPercent(0);
    try{
      JSON.parse(metadata);
      const declared=JSON.parse(metadata);
      if(!declared||typeof declared!=='object'||Array.isArray(declared))throw new Error('Metadata must be a JSON object');
      const options:Record<string,unknown>={...declared};
      for(const [key,value] of Object.entries(parseOptions))if(value)options[key]=key==='has_header'?value==='true':key==='voc_coordinate_base'?Number(value):value;
      const form = new FormData();form.append('file',file);form.append('metadata',JSON.stringify(options));
      // XHR exposes bytes actually transferred. Reaching 100% means the upload
      // finished; the server can still be ingesting/profile-checking the file.
      const value=await new Promise<{id:string;name:string;profile:Record<string,unknown>}>((resolve,reject)=>{
        const request=new XMLHttpRequest();activeUpload.current=request;
        request.open('POST','/api/datasets/upload');
        request.upload.onprogress=e=>{if(e.lengthComputable)setUploadPercent(Math.floor(100*e.loaded/e.total));};
        request.upload.onload=()=>setUploadPercent(100);
        request.onload=()=>{
          try{
            const body=JSON.parse(request.responseText);
            if(request.status>=200&&request.status<300)resolve(body);
            else reject(new Error(typeof body.detail==='string'?body.detail:JSON.stringify(body.detail)));
          }catch{reject(new Error('The server returned an invalid response. Check that the backend is running.'));}
        };
        request.onerror=()=>reject(new Error('Upload connection failed. Retry after checking the backend.'));
        request.onabort=()=>reject(new Error('Upload cancelled. The server may finish a file already received; restart the local server to clear abandoned session data.'));
        request.send(form);
      });
      if(isReference&&value.profile.needs_selection)throw new Error('Reference archive needs a selection. Upload its individual reference file, then select it as the reference dataset.');
      if(isReference)setReference(value.id);else{setDataset(value);setSelectedChoice('0');}
    }catch(e){setError((e as Error).message);}finally{setBusy('');setUploadPercent(null);activeUpload.current=null;}
  }
  async function start(){
    if(!dataset || !context)return;
    setBusy('start');setError('');setCertificate(null);
    try{
      const next=await api<Job>('/assessments',jsonPost({dataset_id:dataset.id,context,interpretation_id:interpretation,reference_dataset_id:reference}));
      setJob(next);
    }
    catch(e){setError((e as Error).message);}finally{setBusy('');}
  }
  async function chooseDataset(){
    if(!dataset)return;
    const choices=dataset.profile.choices as DatasetChoice[]|undefined;
    const choice=choices?.[Number(selectedChoice)];
    if(!choice)return;
    setBusy('selection');setError('');
    try{
      const overrides:Record<string,unknown>={};
      for(const [key,value] of Object.entries(parseOptions))overrides[key]=value?(key==='has_header'?value==='true':key==='voc_coordinate_base'?Number(value):value):null;
      const loaded=await api<{id:string;name:string;profile:Record<string,unknown>}>('/datasets/'+dataset.id+'/select',jsonPost({options:{...overrides,...choice.options}}));
      setDataset(loaded);setSelectedChoice('0');
    }catch(e){setError((e as Error).message);}finally{setBusy('');}
  }
  const running = !!job && ['queued','running'].includes(job.status);
  return <div className="app">
    <aside className="sidebar">
      <a className="brand" href="/" aria-label="FlightReady home"><span className="brand-icon">↗</span> FlightReady<span className="version">research</span></a>
      <div className="workspace-label">RESEARCH WORKSPACE</div>
      <div className="nav active"><span>◈</span> Dataset assessment</div>
      <a className="nav" href="http://127.0.0.1:8000/docs" target="_blank" rel="noreferrer"><span>⌘</span> API documentation</a>
      <div className="sidebar-note"><span className="dot"/> Offline dataset analysis<p>Evidence for your mission.<br/>Requirements before scores.</p></div>
      <div className="sidebar-bottom">UAV DATA READINESS<br/><span>Evidence-aware research edition</span></div>
    </aside>
    <main>
      <header><span>Workspace <span className="slash">/</span> New assessment</span><span className="private">◉ Local session</span></header>
      <div className="content">
        <div className="eyebrow">CONTEXT-AWARE READINESS ASSESSMENT</div>
        <h1>Ready for <em>your</em> mission?</h1>
        <p className="intro">Tell us what you want to build. We’ll assess your dataset against the requirements that matter for your goal.</p>
        <div className="steps"><span className="current">01 <b>Define your goal</b></span><i/><span className={context?'current':''}>02 <b>Confirm requirements</b></span><i/><span className={certificate?'current':''}>03 <b>Review evidence</b></span></div>

        <section className="card">
          <div className="section-top"><h2><span className="step-number">1</span> What would you like to accomplish?</h2><span className="muted">YOUR MISSION</span></div>
          <label htmlFor="goal-text" className="sr-only">Describe your machine-learning objective</label>
          <textarea id="goal-text" className="chat" value={text} disabled={running} onChange={e=>{setText(e.target.value);if(context)resetConfirmation(context);}} placeholder="I want to detect cars and pedestrians in UAV images, in urban areas during daytime, between 20 and 80 meters…"/>
          <div className="hint">Include your target, operating conditions, and what matters most.</div>
          <div className="tag-groups">{Object.entries(TAGS).map(([category,values])=><div className="tag-row" key={category}><span>{category}</span><div>{values.map(tag=><button type="button" key={tag} disabled={running} aria-pressed={tags[category]?.includes(tag)??false} className={'chip '+(tags[category]?.includes(tag)?'selected':'')} onClick={()=>{setTags({...tags,[category]:tags[category]?.includes(tag)?tags[category].filter(t=>t!==tag):[...(tags[category]||[]),tag]});if(context)resetConfirmation(context);}}>{tag}</button>)}</div></div>)}</div>
          <div className="card-footer"><small>Numbers are used only when you provide them.</small><button className="primary" disabled={!!busy||running||(!text.trim()&&!Object.keys(tags).length)} onClick={interpret}>{busy==='interpret'?'Interpreting…':'Interpret my goal'} <span>↗</span></button></div>
        </section>

        {context&&<section className="card" id="requirements">
          <div className="section-top"><h2><span className="step-number">2</span> Review your requirements</h2><span className="badge">{context.provider}</span></div>
          <p className="subtle">Correct the interpretation before confirming. Dataset column and class names must match exactly.</p>
          <fieldset disabled={running} className="editor-fields">
            <label>Task type<select value={context.goal.task_type} onChange={e=>updateGoal('task_type',e.target.value)}>{['unspecified','classification','regression','object_detection','tracking','anomaly_detection','telemetry'].map(t=><option key={t} value={t}>{label(t)}</option>)}</select></label>
            <label>Prediction target / target column<input list="dataset-columns" value={context.goal.prediction_target} onChange={e=>updateGoal('prediction_target',e.target.value)}/><datalist id="dataset-columns">{(dataset?.profile.columns as string[]|undefined)?.map(column=><option key={column} value={column}/>)}</datalist></label>
            {(['required_inputs','optional_inputs','required_classes'] as const).map(field=><label key={field}>{label(field)} (comma separated)<ListInput values={context.goal[field]} onCommit={values=>{if(JSON.stringify(values)!==JSON.stringify(context.goal[field]))updateGoal(field,values);}}/></label>)}
          </fieldset>
          {context.clarification_questions.length>0&&<div className="notice"><strong>Clarification needed</strong><ul>{context.clarification_questions.map(q=><li key={q}>{q}</li>)}</ul><button disabled={running} onClick={()=>resetConfirmation({...context,clarification_questions:[]})}>I have corrected these requirements in the editor</button></div>}
          <details onToggle={e=>{if(e.currentTarget.open)setAdvanced(JSON.stringify(context.goal,null,2));}}><summary>Advanced requirement editor · conditions, thresholds, provenance, hard and preferred checks</summary><p className="subtle">Edit the complete Goal JSON. Apply changes, then confirm below. See README for the schema and examples.</p><textarea aria-label="Full goal JSON" className="code-editor" value={advanced} onChange={e=>setAdvanced(e.target.value)} disabled={running}/><button disabled={running} onClick={()=>{try{const parsed=JSON.parse(advanced);if(typeof parsed!=='object'||!parsed||Array.isArray(parsed))throw new Error('Goal must be an object');resetConfirmation({...context,goal:parsed});setError('');}catch(e){setError((e as Error).message);}}}>Apply JSON changes</button></details>
          <details><summary>Current interpreted requirements</summary><pre>{JSON.stringify(context.goal,null,2)}</pre></details>
          <label className="confirm"><input type="checkbox" checked={context.confirmed} disabled={running||context.clarification_questions.length>0||context.goal.task_type==='unspecified'} onChange={e=>setContext({...context,confirmed:e.target.checked})}/>I reviewed these requirements and confirm they describe my goal.</label>
        </section>}

        <section className="card">
          <div className="section-top"><h2><span className="step-number">3</span> Add your dataset</h2><span className="muted">STATIC FILES ONLY</span></div>
          <details className="metadata"><summary>Optional dataset evidence and reference data</summary><label>Metadata JSON (apply before uploading)<textarea aria-label="Dataset metadata JSON" value={metadata} onChange={e=>setMetadata(e.target.value)} className="code-editor small" disabled={running}/></label><p className="hint">Example: {JSON.stringify({dataset_version:'synthetic-v1',source:'synthetic demonstration',units:{altitude_meters:'m'}})}</p><label>Reference dataset<input type="file" accept={UPLOAD_TYPES} disabled={!!busy||running} onChange={e=>upload(e.target.files?.[0],true)}/></label>{reference&&<p>Reference dataset uploaded.</p>}</details>
          <details><summary>Parsing options · use when the automatic layout needs clarification</summary><div className="editor-fields">
            {[
              ['format','File format',[['','Automatic'],['csv','Delimited CSV'],['tsv','Tab-separated table'],['text','Plain text'],['json','JSON records'],['jsonl','JSON Lines'],['xml','XML records']]],
              ['delimiter','Delimiter',[['','Automatic'],[',','Comma'],['tab','Tab'],[';','Semicolon'],['|','Pipe']]],
              ['has_header','Header row',[['','Default for format'],['true','First row contains names'],['false','No header row']]],
              ['text_unit','Plain-text record unit',[['','One line per record'],['line','One line per record'],['document','Whole file is one document']]],
              ['voc_coordinate_base','Pascal VOC coordinates',[['','Standard: one-based'],['1','One-based inclusive'],['0','Zero-based, nonstandard export']]],
            ].map(([key,title,values])=><label key={key as string}>{title as string}<select aria-label={title as string} disabled={!!busy||running} value={parseOptions[key as string]??''} onChange={e=>setParseOptions({...parseOptions,[key as string]:e.target.value})}>{(values as string[][]).map(([value,caption])=><option key={value} value={value}>{caption}</option>)}</select></label>)}
            <label>JSON/XML record path (optional)<input value={parseOptions.record_path??''} onChange={e=>setParseOptions({...parseOptions,record_path:e.target.value})} disabled={!!busy||running}/></label>
          </div><p className="hint">Options apply to the next upload. Archives with multiple datasets and spreadsheets with multiple sheets offer a selection afterward.</p></details>
          <label className="upload"><span className="upload-icon">↑</span><strong>{dataset?dataset.name:'Choose a dataset to assess'}</strong><span>Text, tables and images · ZIP / TAR collections · up to {(limits.max_upload_bytes/1024**3).toLocaleString()} GiB</span><input type="file" aria-label="Dataset file" accept={UPLOAD_TYPES} disabled={!!busy||running} onChange={e=>upload(e.target.files?.[0])}/></label>
          {busy==='upload'&&<div role="status"><p>{uploadPercent===100?'Upload received. Validating and profiling the complete dataset on the server…':'Uploading dataset: '+(uploadPercent??0)+'%'}</p><progress aria-label="Upload transfer" value={uploadPercent??0} max={100}/><p className="hint">Large CSV and Parquet files are processed on disk. Profiling may take several minutes after transfer completes.</p>{uploadPercent!==100&&<button onClick={()=>activeUpload.current?.abort()}>Cancel upload</button>}</div>}
          {!!dataset?.profile.needs_selection&&<div className="notice"><h3>Choose the dataset to assess</h3><p>{String(dataset.profile.selection_message)}</p><label>Dataset / record layout<select aria-label="Dataset selection" value={selectedChoice} onChange={e=>setSelectedChoice(e.target.value)} disabled={!!busy}>{(dataset.profile.choices as DatasetChoice[]).map((choice,index)=><option key={index} value={index}>{choice.label}</option>)}</select></label><button disabled={!!busy} onClick={chooseDataset}>{busy==='selection'?'Loading the selected dataset…':'Load selected dataset'}</button><p className="hint">The upload is already staged. Selecting a dataset does not upload it again.</p></div>}
          {dataset&&!dataset.profile.needs_selection&&<details><summary>Dataset validated · {String(dataset.profile.row_count)} observations · inspect detected schema</summary><pre>{JSON.stringify(dataset.profile,null,2)}</pre></details>}
          {!!dataset?.profile.suggested_goal&&context&&!dataset.profile.needs_selection&&<div className="notice"><p>Detected {String(dataset.profile.dataset_kind||dataset.profile.annotation_format||'record dataset')}. Review the schema, target column and labels before confirming your goal.</p>{!!dataset.profile.label_source&&<p className="hint">Label source: {String(dataset.profile.label_source)}. Verify that these labels match your intended task.</p>}<button disabled={running} onClick={()=>resetConfirmation({...context,goal:{...context.goal,...dataset.profile.suggested_goal as Partial<Goal>}})}>Use detected dataset setup</button><p className="hint">This updates the editable task fields. You must review and confirm them before assessment.</p></div>}
          {dataset?.profile.annotation_format==='visdrone_det'&&<div className="notice"><strong>Native VisDrone DET recognized</strong><p>Ignored regions and “others” are excluded from target counts. “Pedestrian” and “people” remain separate classes. Image decoding and annotation validity findings are available in the profile and final report.</p></div>}
          <div className="card-footer"><small>Data stays local. Large tables use temporary disk storage.</small><button className="primary" onClick={start} disabled={!!busy||running||!dataset||!!dataset.profile.needs_selection||!context?.confirmed}>Run assessment <span>→</span></button></div>
        </section>

        {error&&<div role="alert" className="error">{error}</div>}
        {job&&<section className="card" aria-live="polite">
          <div className="section-top"><h2>Assessment · {label(job.status)}</h2>{running&&<button onClick={()=>api('/assessments/'+job.id+'/cancel',{method:'POST'}).catch(e=>setError(e.message))}>Cancel</button>}</div>
          <ol className="progress">{job.stages.map((s,i)=><li key={i}><span>✓</span>{s}</li>)}</ol>
          {running&&<p>Backend checks are running…</p>}
          {job.status==='restoring'&&<p>Loading your saved assessment…</p>}
          {progressError&&<div role="alert" className="error"><p>{progressError}</p><button onClick={retryProgress}>Retry connection</button></div>}
          {job.error&&!progressError&&<p role="alert" className="error">{job.error}</p>}
          {['error','cancelled'].includes(job.status)&&<button onClick={start} disabled={!dataset||!context?.confirmed}>Retry assessment</button>}
          {hasReport(job.status)&&<div className="report-access">
            <h3>Your assessment report is ready</h3>
            {reportLoading&&<p role="status">Loading the results and supporting evidence…</p>}
            {reportError&&<div role="alert" className="error"><p>The assessment finished, but its results could not be displayed: {reportError}</p><button onClick={retryReport}>Retry loading results</button></div>}
            {certificate&&<p><strong>{label(certificate.overall_status)}</strong> · Review the findings and recommendations below.</p>}
            <div className="report-actions">
              {certificate?<a className="report-primary" href="#assessment-results">View results</a>:<button onClick={retryReport} disabled={reportLoading}>Load results</button>}
              <a href={'/api/assessments/'+job.id+'/export?format=html'} download>Download HTML report</a>
              <a href={'/api/assessments/'+job.id+'/export?format=json'} download>Download JSON</a>
            </div>
            <p className="hint">Downloads are available directly from the completed assessment, including NOT_READY and insufficient-evidence results.</p>
          </div>}
        </section>}

        {certificate&&<section className="results" id="assessment-results" tabIndex={-1}>
          <ExperimentalScore score={certificate.experimental_score}/>
          <div className="result-banner"><div className="eyebrow">ASSESSMENT COMPLETE</div><h2>{label(certificate.overall_status)}</h2><p>{certificate.executive_summary}</p><div className="export"><a href={'/api/assessments/'+job?.id+'/export?format=json'}>↓ Export JSON</a><a href={'/api/assessments/'+job?.id+'/export?format=html'}>↓ Download report</a></div></div>
          <div className="card"><h2>Your requirement findings</h2>{certificate.findings.map(f=><details key={f.requirement.requirement_id} className="finding"><summary><span className={'status '+f.state}>{label(f.state)}</span><span>{f.requirement.description}<small>{f.requirement.severity} requirement</small></span></summary><p>{f.reason}</p><pre>{JSON.stringify(f.result.evidence,null,2)}</pre></details>)}</div>
          <div className="card"><h2>Eight dimensions of evidence</h2><label className="filter">View dimension<select value={dimension} onChange={e=>setDimension(e.target.value)}><option value="all">All dimensions</option>{DIMENSIONS.map(d=><option key={d} value={d}>{label(d)}</option>)}</select></label>{DIMENSIONS.filter(d=>dimension==='all'||dimension===d).map(d=><div key={d} className="dimension"><h3>{label(d)}</h3>{certificate.metrics.filter(m=>m.dimension===d).map(m=><details key={m.metric_id}><summary><span className={'status '+m.state}>{label(m.assessment_status||m.state)}</span>{label(m.metric_id)}</summary><p>{m.reason}</p><pre>{JSON.stringify({value:m.value,evidence:m.evidence},null,2)}</pre></details>)}</div>)}</div>
          <div className="result-columns"><div className="card"><h2>Supported needs & benefits</h2><ul>{certificate.additional_benefits.map((x,i)=><li key={i}>{x}</li>)}</ul></div><div className="card"><h2>Recommended next steps</h2><ul>{certificate.recommendations.map((x,i)=><li key={i}>{x.description}</li>)}</ul>{certificate.recommendations.length===0&&<p>No unmet measurable requirements. Review the limitations below.</p>}</div></div>
          {certificate.learning_views&&Object.keys(certificate.learning_views).length>0&&<div className="card"><h2>Learning-potential diagnostics</h2><p>These views describe available evidence. Generalization and model performance require independent experiments.</p>{Object.entries(certificate.learning_views).map(([name,view])=><details key={name}><summary>{label(name)}</summary><pre>{JSON.stringify(view,null,2)}</pre></details>)}</div>}
          <div className="card"><h2>Limitations & unverified evidence</h2><ul>{certificate.limitations.map((x,i)=><li key={i}>{x}</li>)}</ul><details><summary>Reproducibility information</summary><pre>{JSON.stringify(certificate.reproducibility,null,2)}</pre></details></div>
          <p className="disclaimer">{certificate.disclaimer}</p>
        </section>}
        <footer><span>◈ Evidence over assumptions</span><span>Research assessment · No guarantee of model performance</span></footer>
      </div>
    </main>
  </div>;
}
const container=document.getElementById('root');
if(container)createRoot(container).render(location.search.includes('legacy=1')||new URLSearchParams(location.search).has('assessment')?<App/>:<Workbench/>);
