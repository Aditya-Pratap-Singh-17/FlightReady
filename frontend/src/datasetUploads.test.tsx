import React from 'react';
import { render,screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach,beforeEach,expect,it,vi } from 'vitest';
import { App } from './main';

const pending={id:'c'.repeat(32),name:'repository.zip',profile:{needs_selection:true,
  selection_message:'Choose train or test records.',choices:[
    {label:'Records: train.json',options:{dataset_member:'train.json'}},
    {label:'Records: test.json',options:{dataset_member:'test.json'}},
  ],row_count:0,columns:[]}};
const loaded={id:pending.id,name:'test.json',profile:{row_count:1,columns:['text','label'],parsing:{format:'json'}}};
const uploads:FormData[]=[];
let uploadedResponse:unknown=pending;

class UploadXHR{
  upload:{onload:(()=>void)|null;onprogress:unknown}={onload:null,onprogress:null};
  onload:(()=>void)|null=null;onabort:(()=>void)|null=null;onerror:unknown=null;
  status=200;responseText='';
  open(){}
  send(body:FormData){
    uploads.push(body);this.responseText=JSON.stringify(uploadedResponse);
    queueMicrotask(()=>{this.upload.onload?.();this.onload?.();});
  }
  abort(){this.onabort?.();}
}
beforeEach(()=>{uploads.length=0;uploadedResponse=pending;vi.stubGlobal('XMLHttpRequest',UploadXHR);});
afterEach(()=>vi.unstubAllGlobals());
const json=(body:unknown)=>new Response(JSON.stringify(body),{headers:{'Content-Type':'application/json'}});

it('selects a file from an already staged archive without a second upload',async()=>{
  const fetcher=vi.fn(async(path:RequestInfo|URL)=>json(String(path).endsWith('/select')?loaded:{max_upload_bytes:2*1024**3,max_image_upload_bytes:2*1024**3}));
  vi.stubGlobal('fetch',fetcher);
  render(<App/>);
  await userEvent.upload(screen.getByLabelText('Dataset file'),new File(['zip fixture'],'repository.zip',{type:'application/zip'}));
  await screen.findByRole('heading',{name:'Choose the dataset to assess'});
  await userEvent.selectOptions(screen.getByLabelText('Dataset selection'),'1');
  await userEvent.click(screen.getByRole('button',{name:'Load selected dataset'}));
  await screen.findByText(/Dataset validated · 1 observations/);
  expect(uploads).toHaveLength(1);
  expect(fetcher.mock.calls.some(([path])=>String(path)==='/api/datasets/'+pending.id+'/select')).toBe(true);
  expect(screen.queryByRole('heading',{name:'Choose the dataset to assess'})).toBeNull();
});

it('sends explicitly chosen delimiter and header settings as typed metadata',async()=>{
  uploadedResponse=loaded;
  vi.stubGlobal('fetch',vi.fn(async()=>json({max_upload_bytes:2*1024**3,max_image_upload_bytes:2*1024**3})));
  render(<App/>);
  await userEvent.selectOptions(screen.getByLabelText('Delimiter'),'|');
  await userEvent.selectOptions(screen.getByLabelText('Header row'),'false');
  await userEvent.upload(screen.getByLabelText('Dataset file'),new File(['hello|yes'],'records.txt',{type:'text/plain'}));
  await screen.findByText(/Dataset validated · 1 observations/);
  const metadata=JSON.parse(String(uploads[0].get('metadata')));
  expect(metadata.delimiter).toBe('|');
  expect(metadata.has_header).toBe(false);
});

it('requires fresh confirmation after the user applies a detected image setup',async()=>{
  const initial={id:'interpretation',context:{original_text:'Classify images',selected_tags:{},goal:{
    task_type:'classification',task_description:'Classify images',prediction_target:'label',required_inputs:[],optional_inputs:[],required_classes:[]},
    clarification_questions:[],confirmed:false,provider:'deterministic'}};
  uploadedResponse={id:pending.id,name:'images.zip',profile:{row_count:2,columns:['image','label'],dataset_kind:'image_classification',
    suggested_goal:{task_type:'classification',prediction_target:'label',input_modalities:['image']}}};
  vi.stubGlobal('fetch',vi.fn(async(path:RequestInfo|URL)=>json(String(path).endsWith('/interpret')?initial:{max_upload_bytes:2*1024**3,max_image_upload_bytes:2*1024**3})));
  render(<App/>);
  await userEvent.type(screen.getByLabelText('Describe your machine-learning objective'),'Classify images');
  await userEvent.click(screen.getByRole('button',{name:/Interpret my goal/}));
  const confirmation=await screen.findByLabelText('I reviewed these requirements and confirm they describe my goal.');
  await userEvent.click(confirmation);
  expect((confirmation as HTMLInputElement).checked).toBe(true);
  await userEvent.upload(screen.getByLabelText('Dataset file'),new File(['fixture'],'images.zip',{type:'application/zip'}));
  await userEvent.click(await screen.findByRole('button',{name:'Use detected dataset setup'}));
  expect((confirmation as HTMLInputElement).checked).toBe(false);
  expect((screen.getByRole('button',{name:/Run assessment/}) as HTMLButtonElement).disabled).toBe(true);
});
