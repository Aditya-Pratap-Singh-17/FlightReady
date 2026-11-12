import React from 'react';
import {render,screen,waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {QueryClient,QueryClientProvider} from '@tanstack/react-query';
import {beforeEach,afterEach,it,expect,vi} from 'vitest';
import {QualityAnalytics,type QualityProfile} from './QualityAnalytics';
import {Workbench} from './Workbench';
import type {Certificate} from './api';

const profile={task:'object_detection',scenario:'default',readiness_status:'blocked',requires_measurement:false,
  dimensions:[{id:'annotation_quality',name:'Annotation quality',score:0,status:'warning',evidence_coverage:1,effective_weight:.6,metrics:[],deficiencies:['No valid annotations'],recommendations:['Repair labels']},
  {id:'integrity',name:'Data integrity',score:null,status:'insufficient_evidence',evidence_coverage:.5,effective_weight:.4,metrics:[],deficiencies:['Missing independent evidence'],recommendations:[]}],
  critical_requirements:[{id:'ANNOTATIONS',severity:'block',status:'fail',kind:'technical',reason:'Observed invalid labels'}],
  profile_version:'test',normalization_version:'test',aggregation_version:'test',profile_sha256:'test',calculation_sha256:'test',weight_explanations:['Test policy provenance'],formula:'Geometric test formula',limitations:[],warning_below:70,
  profile:{dimensions:{annotation_quality:'Annotation quality',integrity:'Data integrity'},tasks:{object_detection:{},classification:{}},scenarios:{default:{tasks:['object_detection','classification'],explanation:'Default'},nighttime_detection:{tasks:['object_detection'],explanation:'Opt-in provisional policy'}}}} satisfies QualityProfile;
const certificate={quality_profile:profile,confirmed_context:{original_text:'Fixture',goal:{task_type:'object_detection'}}} as unknown as Certificate;
const requests:{url:string;body:Record<string,unknown>}[]=[];
beforeEach(()=>{
  requests.length=0;location.hash='';localStorage.clear();sessionStorage.clear();
  vi.stubGlobal('ResizeObserver',class {observe(){}unobserve(){}disconnect(){}});
  vi.stubGlobal('fetch',vi.fn(async(input:RequestInfo|URL,options?:RequestInit)=>{
    const url=String(input);const body=options?.body?JSON.parse(String(options.body)):{};requests.push({url,body});
    let value:unknown=[];
    if(url.endsWith('/quality/aggregate'))value={...profile,task:body.task||profile.task,scenario:body.scenario||'default',requires_measurement:body.task==='classification',aggregation_id:body.persist?'saved-version':undefined};
    if(url.endsWith('/compute'))value={cuda_available:false,selected_device:'cpu',devices:['cpu']};
    if(url.endsWith('/overview'))value={datasets:0,evaluations:0,training_experiments:0,comparisons:0,recent_activity:[]};
    return new Response(JSON.stringify(value),{status:200,headers:{'Content-Type':'application/json'}});
  }));
});
afterEach(()=>vi.unstubAllGlobals());
function show(){return render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><QualityAnalytics certificate={certificate} readinessId="readiness-fixture" onReassess={async()=>{}}/></QueryClientProvider>);}

it('keeps genuine zero quality distinct from missing evidence and shows independent gates',()=>{
  show();expect(screen.getByRole('heading',{name:'Multi-dimensional readiness profile'})).toBeTruthy();
  expect(screen.getByText('0.0')).toBeTruthy();expect(screen.getByText('Not scored')).toBeTruthy();
  expect(screen.getByText('Observed invalid labels')).toBeTruthy();
});

it('reaggregates on task change and requests review when task-specific evidence is missing',async()=>{
  show();await userEvent.selectOptions(screen.getByLabelText('Aggregation task'),'classification');
  expect(await screen.findByText('Task preview needs new evidence.')).toBeTruthy();
  expect(requests.find(r=>r.url.endsWith('/quality/aggregate'))?.body.task).toBe('classification');
});

it('requires explicit priority confirmation and persists the exact submitted multipliers',async()=>{
  show();const input=screen.getByLabelText('Annotation quality priority multiplier');
  await userEvent.clear(input);await userEvent.type(input,'2');
  expect(screen.getByRole('button',{name:'Save aggregation version'})).toHaveProperty('disabled',true);
  await userEvent.click(screen.getByRole('checkbox',{name:'I confirm these priority overrides'}));
  await userEvent.click(screen.getByRole('button',{name:'Save aggregation version'}));
  expect(await screen.findByRole('status')).toHaveProperty('textContent','Aggregation version saved: saved-version');
  const request=requests.find(r=>r.url.endsWith('/quality/aggregate'))!;
  expect(request.body.dimension_weights).toEqual({annotation_quality:2});expect(request.body.overrides_confirmed).toBe(true);expect(request.body.persist).toBe(true);
});

it('offers the Model Validation Lab as independent navigation with six internal sections',async()=>{
  location.hash='#Advanced%20Lab';
  render(<Workbench/>);
  expect(await screen.findByRole('heading',{name:'Does measured readiness predict model performance?'})).toBeTruthy();
  expect(screen.getByRole('navigation',{name:'Model Validation Lab sections'}).querySelectorAll('button')).toHaveLength(6);
  await userEvent.click(screen.getByRole('button',{name:'Experiment Builder'}));
  expect(await screen.findByRole('heading',{name:'Experiment Builder'})).toBeTruthy();
  expect(screen.getByLabelText('Candidate dataset version')).toBeTruthy();expect(screen.getByLabelText('Lab independent test dataset')).toBeTruthy();
});
