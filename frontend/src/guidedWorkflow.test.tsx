import React from 'react';
import {render,screen,waitFor,within} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {beforeEach,afterEach,expect,it,vi} from 'vitest';
import {Workbench} from './Workbench';

const dataset={id:'guided-data',kind:'dataset',created_at:'2026-11-02',payload:{name:'Drone images',sha256:'abc',version:'v1',modality:'image',profile:{image_count:10,class_counts:{drone:10}},training:{eligible:true,reasons:[]},provenance:{}}};
const context={original_text:'Detect drones in images',goal:{task_type:'object_detection',task_description:'Detect drones in images',required_classes:['drone'],input_modalities:['image']},clarification_questions:[],confirmed:false,provider:'gemini:fixture'};
let project:any;
let requests:{path:string;body:any}[];
beforeEach(()=>{
  location.hash='';localStorage.clear();sessionStorage.clear();requests=[];
  project={id:'guided-project',kind:'workflow',created_at:'2026-11-02',payload:{dataset_id:dataset.id,messages:[],context:null,context_id:null,confirmed_context_id:null,readiness_id:null,split_id:null,validation_id:null}};
  vi.stubGlobal('ResizeObserver',class{observe(){}unobserve(){}disconnect(){}});
  vi.stubGlobal('fetch',vi.fn(async(input:RequestInfo|URL,options?:RequestInit)=>{
    const path=String(input);const body=options?.body?JSON.parse(String(options.body)):null;requests.push({path,body});let result:unknown=[];
    if(path.endsWith('/datasets'))result=[dataset];
    if(path.endsWith('/compute'))result={cuda_available:false,devices:['cpu'],selected_device:'cpu'};
    if(path.endsWith('/context/provider'))result={configured:true,name:'Gemini'};
    if(path.endsWith('/workflows'))result=options?.method==='POST'?project:[];
    if(path.endsWith('/workflows/guided-project'))result=project;
    if(path.endsWith('/goal')){project={...project,payload:{...project.payload,messages:[{role:'user',text:body.message}],context,context_id:'interpreted-goal'}};result=project;}
    if(path.endsWith('/confirm')){project={...project,payload:{...project.payload,context:body.context,context_id:'confirmed-goal',confirmed_context_id:'confirmed-goal'}};result=project;}
    return new Response(JSON.stringify(result),{status:200,headers:{'Content-Type':'application/json'}});
  }));
});
afterEach(()=>vi.unstubAllGlobals());

async function openGoal(){render(<Workbench/>);await userEvent.click(await screen.findByText(/Choose from your dataset library/));await userEvent.click(screen.getByRole('button',{name:/Drone images/}));await screen.findByLabelText('Describe your goal');}

it('opens on upload with later workflow stages locked and technical tools collapsed',async()=>{
  render(<Workbench/>);
  expect(await screen.findByRole('heading',{name:'Good decisions start with your data.'})).toBeTruthy();
  const stages=within(screen.getByRole('navigation',{name:'Project workflow'})).getAllByRole('button');
  expect(stages[1]).toHaveProperty('disabled',true);expect(stages[2]).toHaveProperty('disabled',false);
  expect(screen.queryByRole('button',{name:'Analyze dataset'})).toBeNull();
  expect(document.querySelector('.wb-workspace-tools')).toHaveProperty('open',false);
});

it('keeps interpretation separate from confirmation and saves the reviewed goal',async()=>{
  await openGoal();await userEvent.type(screen.getByLabelText('Describe your goal'),'Detect drones in images');
  await userEvent.click(screen.getByRole('button',{name:'Interpret goal'}));
  expect(await screen.findByRole('heading',{name:'Review the goal before analysis'})).toBeTruthy();
  expect(project.payload.confirmed_context_id).toBeNull();
  expect(requests.find(r=>r.path.endsWith('/goal'))?.body.enable_remote).toBe(true);
  const target=screen.getByLabelText('Goal prediction target');
  await userEvent.clear(target);await userEvent.type(target,'drone, bird');
  expect(target).toHaveProperty('value','drone, bird');
  await userEvent.click(screen.getByRole('button',{name:'Confirm & save goal'}));
  expect(await screen.findByRole('heading',{name:'Goal confirmed'})).toBeTruthy();
  const saved=requests.find(r=>r.path.endsWith('/confirm'))!.body.context;
  expect(saved.confirmed).toBe(true);expect(saved.goal.required_classes).toEqual(['drone','bird']);
  await userEvent.click(screen.getByRole('button',{name:'Continue to analysis'}));
  expect(await screen.findByRole('button',{name:'Analyze dataset'})).toBeTruthy();
  expect(screen.queryByRole('button',{name:'Start training & comparison'})).toBeNull();
});

it('allows model import and prediction preview before a readiness report exists',async()=>{
  await openGoal();await userEvent.click(screen.getByRole('button',{name:'Model Validation Lab'}));
  expect(await screen.findByRole('heading',{name:'Import a model to test your data.'})).toBeTruthy();
  expect(screen.getByRole('heading',{name:'Run a model on your images.'})).toBeTruthy();
  expect(screen.queryByRole('button',{name:'Prepare comparison'})).toBeNull();
  await waitFor(()=>expect(requests.filter(r=>r.path.includes('/validation/')).length).toBe(0));
});

it('shows the safe Gemini failure reason alongside the local draft',async()=>{
  const fetch=globalThis.fetch;
  vi.stubGlobal('fetch',vi.fn(async(input:RequestInfo|URL,options?:RequestInit)=>{
    if(String(input).endsWith('/goal')){
      project={...project,payload:{...project.payload,context:{...context,
        provider:'deterministic (provider unavailable or invalid output)',
        provider_error_code:'quota_exceeded',provider_error:'Gemini quota or rate limit reached (HTTP 429).'},context_id:'local-draft'}};
      return new Response(JSON.stringify(project),{status:200});
    }
    return fetch(input,options);
  }));
  await openGoal();await userEvent.type(screen.getByLabelText('Describe your goal'),'Detect drones in images');
  await userEvent.click(screen.getByRole('button',{name:'Interpret goal'}));
  expect(await screen.findByText('Gemini quota or rate limit reached (HTTP 429).')).toBeTruthy();
  expect(screen.getByRole('button',{name:'Confirm & save goal'})).toHaveProperty('disabled',false);
});
