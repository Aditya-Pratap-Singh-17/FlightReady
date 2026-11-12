import React,{useState} from 'react';
import {BarChart,Bar,XAxis,YAxis,Tooltip,ResponsiveContainer} from 'recharts';
import {fmt,type Certificate} from './api';

type Plot={title:string;unit:string;denominator:number|string|null;status:string;rows:{label:string;value:number}[]};
type Row={label:string;value:number|null;unit:string;denominator:number|string|null;status:string};
export type EvidenceSection={id:string;title:string;metric_id:string;status:string;evidence_type:string;sample_size:number|null;coverage:number|null;denominator:number|string|null;version:string;provenance:unknown;limitations:string[];plots:Plot[];rows:Row[];methodology?:unknown};
export type ResearchEvidenceData={version:string;notice:string;sections:EvidenceSection[]};

export function ResearchEvidence({certificate}:{certificate:Certificate}){
  const evidence=certificate.quality_measurements?.research_evidence;
  const [active,setActive]=useState('object_scale');
  const section=evidence?.sections.find(s=>s.id===active)||evidence?.sections[0];
  return <section className="wb-panel" aria-label="Research measurement evidence"><span className="wb-eyebrow">MEASUREMENTS AND THEIR LIMITS</span><h2>Inspect the evidence behind your dataset</h2>
    <p>These observations describe the supplied data. They do not predict detection accuracy or guarantee visibility. New proxies keep the existing scoring policy unchanged.</p>
    {!section?<p>Enhanced evidence is unavailable in this historical certificate. Run a new assessment to collect it; saved results remain unchanged.</p>:<>
      <label className="wb-field"><span>Evidence view</span><select value={section.id} onChange={e=>setActive(e.target.value)}>{evidence!.sections.map(s=><option key={s.id} value={s.id}>{s.title}</option>)}</select></label>
      <EvidenceView key={section.id} section={section}/>
    </>}
  </section>;
}

function EvidenceView({section:s}:{section:EvidenceSection}){
  const [selected,setSelected]=useState(0);
  const plot=s.plots[selected];
  return <div><h3>{s.title}</h3><p><strong>{s.metric_id} · {s.status.replaceAll('_',' ')}</strong> · {s.evidence_type}</p>
    <p>Sample size: {fmt(s.sample_size)} · Coverage: {s.coverage==null?'Unavailable':`${(s.coverage*100).toFixed(1)}%`} · Population/denominator: {fmt(s.denominator)}</p>
    {s.plots.length>0&&<><label className="wb-field"><span>Measured quantity</span><select aria-label="Research evidence chart" value={selected} onChange={e=>setSelected(Number(e.target.value))}>{s.plots.map((p,i)=><option value={i} key={i}>{p.title}</option>)}</select></label>
      <p>{plot.title} · Units: {plot.unit} · Denominator: {fmt(plot.denominator)} · {plot.status.replaceAll('_',' ')}</p>
      {plot.rows.length>0?<><div className="wb-chart"><ResponsiveContainer width="100%" height={260}><BarChart data={plot.rows}><XAxis dataKey="label" tick={{fontSize:10}}/><YAxis allowDecimals={false}/><Tooltip/><Bar dataKey="value" fill="#398870" isAnimationActive={false}/></BarChart></ResponsiveContainer></div><div className="wb-table-scroll"><table><thead><tr><th>Category / interval</th><th>{plot.unit}</th></tr></thead><tbody>{plot.rows.map((r,i)=><tr key={i}><td>{r.label}</td><td>{fmt(r.value)}</td></tr>)}</tbody></table></div></>:<p role="status">Unavailable: no supporting measurements were collected for this quantity. This does not imply zero.</p>}
    </>}
    {s.rows.length>0&&<details><summary>Measurements, units and individual denominators ({s.rows.length})</summary><div className="wb-table-scroll"><table><thead><tr><th>Measurement</th><th>Value</th><th>Units</th><th>Denominator</th></tr></thead><tbody>{s.rows.map((r,i)=><tr key={i}><td>{r.label}</td><td>{fmt(r.value)}</td><td>{r.unit}</td><td>{fmt(r.denominator)}</td></tr>)}</tbody></table></div></details>}
    <ul>{s.limitations.map((v,i)=><li key={i}>{v}</li>)}</ul><details><summary>Source, method and measurement version</summary><pre>{JSON.stringify({version:s.version,provenance:s.provenance,methodology:s.methodology},null,2)}</pre></details>
  </div>;
}
