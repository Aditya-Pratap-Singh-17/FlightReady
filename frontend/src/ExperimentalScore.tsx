import React, {useEffect, useState} from 'react';

export type ScoreComponent = {
  id:string;label:string;weight:number;score_percent:number|null;effective_weight_percent:number;
  contribution_points:number|null;status:string;reason:string;raw_value:unknown;
};
export type WeightedScore = {
  experimental:boolean;version?:string;status:string;score_percent:number|null;evidence_coverage_percent:number;
  score_range_percent:[number,number]|null;components:ScoreComponent[];notice:string;reason?:string;
  hard_failure_ids?:string[];profile_sha256?:string;
};

const number = (value:number|null|undefined) => value === null || value === undefined ? 'Unavailable' : value.toFixed(2);

export function ExperimentalScore({score,onPreview}: {
  score?:WeightedScore|null;onPreview?:(weights:Record<string,number>)=>Promise<WeightedScore>;
}) {
  const [preview,setPreview] = useState<WeightedScore|null>(null);
  const [weights,setWeights] = useState<Record<string,string>>({});
  const [error,setError] = useState('');
  const [busy,setBusy] = useState(false);
  useEffect(()=>{setPreview(null);setWeights({});setError('');},[score]);
  const current = preview || score;
  if(current?.status === 'disabled') return null;
  if(!current && !onPreview) return null;
  const run = async () => {
    setError('');
    const selected:Record<string,number> = {};
    for(const component of current?.components || []) {
      const raw = weights[component.id] ?? String(component.weight);
      const value = Number(raw);
      if(!raw.trim() || !Number.isFinite(value) || value < 0 || value > 1000000) {
        setError('Enter finite weights between 0 and 1000000. Use 0 to exclude a component.');return;
      }
      selected[component.id] = value;
    }
    setBusy(true);
    try {setPreview(await onPreview!(selected));}
    catch(e) {setError(e instanceof Error ? e.message : String(e));}
    finally {setBusy(false);}
  };
  const download = () => {
    const url = URL.createObjectURL(new Blob([JSON.stringify(current,null,2)],{type:'application/json'}));
    const link = document.createElement('a');link.href = url;link.download = 'experimental-score-preview.json';link.click();
    setTimeout(()=>URL.revokeObjectURL(url),0);
  };
  return <section className="wb-panel card" aria-label="Experimental weighted quality score">
    <h2>Experimental weighted quality score</h2>
    <p>Temporary scoring preset. Edit importance weights below; 0 excludes a component. This score is not predicted model accuracy.</p>
    {current ? <>
      <p><strong>{number(current.score_percent)}{current.score_percent !== null ? ' / 100' : ''}</strong> · {current.status.replaceAll('_',' ')}{preview ? ' · preview' : ''}</p>
      <p>Evidence coverage: <strong>{number(current.evidence_coverage_percent)}%</strong> of applicable weight.</p>
      {current.status === 'partial' && <p>Missing evidence is excluded from the displayed average. With unknown components spanning 0–100, the full score could range from {number(current.score_range_percent?.[0])} to {number(current.score_range_percent?.[1])}. This is a missing-evidence range, not a confidence interval.</p>}
      {current.hard_failure_ids?.length ? <p role="note">Hard requirements failed. A high weighted score does not override the readiness decision.</p> : null}
      {current.reason && <p>{current.reason}</p>}
      {current.components.length > 0 && <details><summary>Edit weights and inspect contributions</summary>
        <div className="wb-table-scroll" style={{overflowX:'auto'}}><table><thead><tr><th>Component</th><th>Score / 100</th><th>Importance weight</th><th>Contribution points</th><th>Evidence</th></tr></thead>
          <tbody>{current.components.map(c=><tr key={c.id}><td>{c.label}</td><td>{number(c.score_percent)}</td><td>{onPreview ? <input style={{width:'6rem'}} aria-label={c.label+' weight'} type="number" min="0" max="1000000" step="any" value={weights[c.id] ?? c.weight} onChange={e=>setWeights({...weights,[c.id]:e.target.value})}/> : c.weight}</td><td>{number(c.contribution_points)}</td><td>{c.status.replaceAll('_',' ')}<small>{c.reason}</small></td></tr>)}</tbody></table></div>
        {onPreview && <button disabled={busy} onClick={()=>void run()}>{busy ? 'Calculating…' : 'Preview weights'}</button>}
        <p>Preview changes are local to this view. Saved assessments and model runs retain their original score profile. Change config/experimental_score.json to set defaults for future assessments.</p>
      </details>}
      <details><summary>Score profile and raw component evidence</summary><pre>{JSON.stringify(current,null,2)}</pre></details>
      {preview && <div className="wb-actions"><button onClick={()=>{setPreview(null);setWeights({});setError('');}}>Reset preview</button><button onClick={download}>Download score preview</button></div>}
    </> : <button disabled={busy} onClick={()=>void run()}>{busy ? 'Calculating…' : 'Preview experimental score for this report'}</button>}
    {error && <p role="alert">{error}</p>}
  </section>;
}
