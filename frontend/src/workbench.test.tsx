import React from 'react';
import {render,screen,waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {beforeEach,afterEach,expect,it,vi} from 'vitest';
import {Workbench} from './Workbench';

const dataset={id:'dataset1',kind:'dataset',created_at:'2026-01-01',payload:{name:'UAV fixture',sha256:'abc',version:'abc',modality:'image',profile:{image_count:10,object_count:8,class_counts:{drone:8},annotation_format:'yolo'},training:{eligible:true,reasons:[]},provenance:{}}};
const response=(body:unknown,status=200)=>new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json'}});
let imported=false;
let override:((url:string,options?:RequestInit)=>Response|undefined)|undefined;
const requests:{url:string;options?:RequestInit}[]=[];
beforeEach(()=>{
  localStorage.clear();imported=false;override=undefined;requests.length=0;
  vi.stubGlobal('ResizeObserver',class {observe(){}unobserve(){}disconnect(){}});
  vi.stubGlobal('fetch',vi.fn(async(input:RequestInfo|URL,options?:RequestInit)=>{
    const url=String(input);requests.push({url,options});const replaced=override?.(url,options);if(replaced)return replaced;
    if(url.endsWith('/compute'))return response({torch_installed:true,detector_installed:true,cuda_available:false,devices:['cpu'],selected_device:'cpu'});
    if(url.endsWith('/overview'))return response({datasets:3,evaluations:2,training_experiments:1,comparisons:0,recent_activity:[]});
    if(url.endsWith('/datasets'))return response(imported?[dataset]:[]);
    if(url.endsWith('/jobs')||url.includes('/splits'))return response([]);
    if(url.includes('/samples'))return response({total:0,samples:[],modality:'image'});
    if(url.endsWith('/datasets/register')){imported=true;return response(dataset,201);}
    return response({});
  }));
});
afterEach(()=>vi.unstubAllGlobals());

it('shows real backend overview counts and meaningful empty activity',async()=>{
  location.hash='#Overview';
  render(<Workbench/>);
  await screen.findByText('Your first experiment starts here');
  expect(await screen.findByText('3')).toBeTruthy();
  expect(screen.getByText('Backend connected')).toBeTruthy();
  await userEvent.click(screen.getByRole('button',{name:'Datasets'}));
  expect(await screen.findByText('No datasets imported')).toBeTruthy();
});

it('registers a local dataset and refreshes the library',async()=>{
  render(<Workbench/>);
  await userEvent.click(await screen.findByRole('button',{name:'Datasets'}));
  await userEvent.type(screen.getByLabelText('Local dataset directory'),'C:\\Datasets\\drone');
  await userEvent.click(screen.getByRole('button',{name:/Register directory/}));
  await screen.findByText('Local directory registered without copying images');
  expect((await screen.findAllByText('UAV fixture')).length).toBeGreaterThan(0);
  const request=requests.find(r=>r.url.endsWith('/datasets/register'));
  expect(JSON.parse(String(request?.options?.body)).path).toBe('C:\\Datasets\\drone');
});

it('surfaces backend import errors without a success state',async()=>{
  override=(url)=>url.endsWith('/datasets/register')?response({detail:'Unsupported dataset layout'},422):undefined;
  render(<Workbench/>);
  await userEvent.click(await screen.findByRole('button',{name:'Datasets'}));
  await userEvent.type(screen.getByLabelText('Local dataset directory'),'C:\\Invalid');
  await userEvent.click(screen.getByRole('button',{name:/Register directory/}));
  expect(await screen.findByRole('alert')).toHaveProperty('textContent','Unsupported dataset layout');
  expect(screen.queryByText('Local directory registered without copying images')).toBeNull();
});

it('keeps a newly registered dataset selected when an existing library is cached',async()=>{
  imported=true;
  const second={...dataset,id:'dataset2',payload:{...dataset.payload,name:'Second UAV dataset'}};
  let registered=false;
  override=(url)=>{
    if(url.endsWith('/datasets/register')){registered=true;return response(second,201);}
    if(url.endsWith('/datasets'))return response(registered?[dataset,second]:[dataset]);
    return undefined;
  };
  render(<Workbench/>);
  await userEvent.click(await screen.findByRole('button',{name:'Datasets'}));
  await userEvent.type(screen.getByLabelText('Local dataset directory'),'C:\\Datasets\\second');
  await userEvent.click(screen.getByRole('button',{name:/Register directory/}));
  expect(await screen.findByRole('heading',{name:'Second UAV dataset'})).toBeTruthy();
});

it('requires context review and explicit opt-in before remote interpretation',async()=>{
  imported=true;
  override=(url)=>url.endsWith('/context/interpret')?response({context:{original_text:'Detect drones',goal:{task_type:'object_detection'},clarification_questions:[],confirmed:false,provider:'deterministic'}}):undefined;
  render(<Workbench/>);
  await userEvent.click(await screen.findByRole('button',{name:'Dataset Evaluation'}));
  await userEvent.type(await screen.findByLabelText('Intended use'),'Detect drones');
  await userEvent.click(screen.getByRole('button',{name:'Interpret context'}));
  const run=await screen.findByRole('button',{name:/Run existing readiness pipeline/});
  expect(run).toHaveProperty('disabled',true);
  expect(JSON.parse(String(requests.find(r=>r.url.endsWith('/context/interpret'))?.options?.body)).enable_remote).toBe(false);
  await userEvent.click(screen.getByRole('checkbox',{name:'I confirm this context and its explicit requirements'}));
  await waitFor(()=>expect(run).toHaveProperty('disabled',false));
});

it('keeps invalid context edits unconfirmed and displays schema errors',async()=>{
  imported=true;
  override=(url)=>url.endsWith('/context/interpret')?response({context:{original_text:'Detect drones',goal:{task_type:'object_detection'},clarification_questions:[],confirmed:false,provider:'deterministic'}}):url.endsWith('/context/validate')?response({detail:'Context requires goal and original_text'},422):undefined;
  render(<Workbench/>);
  await userEvent.click(await screen.findByRole('button',{name:'Dataset Evaluation'}));
  await userEvent.type(await screen.findByLabelText('Intended use'),'Detect drones');
  await userEvent.click(screen.getByRole('button',{name:'Interpret context'}));
  const editor=await screen.findByLabelText('Structured context JSON');
  await userEvent.clear(editor);await userEvent.type(editor,'{{}');
  await userEvent.click(screen.getByRole('button',{name:'Apply context edits'}));
  expect(await screen.findByRole('alert')).toHaveProperty('textContent','Context requires goal and original_text');
  expect(screen.getByRole('checkbox',{name:'I confirm this context and its explicit requirements'})).toHaveProperty('disabled',true);
});
