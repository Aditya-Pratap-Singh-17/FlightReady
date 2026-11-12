import React from 'react';
import {render,screen,waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {QueryClient,QueryClientProvider} from '@tanstack/react-query';
import {afterEach,expect,it,vi} from 'vitest';
import {ModelLibrary} from './ModelLibrary';
import type {Job,RecordData} from './api';
const spec={id:'yolos-tiny-coco',name:'YOLOS Tiny',repository:'hustvl/yolos-tiny',revision:'b'.repeat(40),architecture:'YolosForObjectDetection',license:'apache-2.0',source_url:'https://huggingface.co/hustvl/yolos-tiny',download_size_bytes_approximate:26000000,pretrained_classes:['car'],operations:['inference'],limitations:'Only declared labels are supported.'};
afterEach(()=>vi.unstubAllGlobals());

it('inspects a supplied repository and acquires its frozen source after consent',async()=>{
  const calls:{path:string;body:any}[]=[];
  vi.stubGlobal('fetch',vi.fn(async(input:RequestInfo|URL,options?:RequestInit)=>{
    const path=String(input),body=options?.body?JSON.parse(String(options.body)):null;calls.push({path,body});
    const data=path.endsWith('/model-registry')?[spec]:path.endsWith('/inference-runtime')?{available:true}:path.endsWith('/model-sources/inspect')?{id:'inspected-source',kind:'model_source',payload:{...spec,name:'My DETR detector',repository:'researcher/my-detector',architecture:'DetrForObjectDetection'}}:[];
    return new Response(JSON.stringify(data),{status:200});
  }));
  render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}><ModelLibrary jobs={[]} renderMonitor={()=>null}/></QueryClientProvider>);
  const input=await screen.findByLabelText('Hugging Face model ID');await userEvent.clear(input);await userEvent.type(input,'researcher/my-detector');
  await userEvent.click(screen.getByRole('button',{name:'Inspect Hugging Face model'}));
  expect(await screen.findByRole('heading',{name:'My DETR detector'})).toBeTruthy();
  expect(screen.getByRole('button',{name:'Download & import model'})).toHaveProperty('disabled',true);
  await userEvent.click(screen.getByLabelText('I consent to downloading these model weights from Hugging Face.'));
  await userEvent.click(screen.getByLabelText('I have reviewed and accept the declared model license.'));
  await userEvent.click(screen.getByRole('button',{name:'Download & import model'}));
  await waitFor(()=>expect(calls.find(c=>c.path.endsWith('/model-registry/acquire'))?.body).toEqual({source_id:'inspected-source',download_consent:true,license_accepted:true,local_only:false}));
});

it('refreshes the cached models when an acquisition job completes',async()=>{
  let ready=false;
  vi.stubGlobal('fetch',vi.fn(async(input:RequestInfo|URL)=>{
    const path=String(input);const data=path.endsWith('/model-registry')?[spec]:path.endsWith('/inference-runtime')?{available:true}:path.endsWith('/acquired-models')&&ready?[{id:'loaded',payload:{name:'Imported detector',classes:['car'],architecture:'YolosForObjectDetection'}}]:[];
    return new Response(JSON.stringify(data),{status:200});
  }));
  const client=new QueryClient({defaultOptions:{queries:{retry:false}}});
  const view=(jobs:RecordData<Job>[])=><QueryClientProvider client={client}><ModelLibrary jobs={jobs} renderMonitor={()=>null}/></QueryClientProvider>;
  const {rerender}=render(view([]));await screen.findByText(/No models imported yet/);ready=true;
  rerender(view([{id:'download',kind:'job',created_at:'2026-11-10',payload:{task:'model_acquisition',status:'completed',stage:'Completed',config:{},result:null,error:null}}]));
  expect(await screen.findByText('Imported detector')).toBeTruthy();
});
