import React from 'react';
import { act, render, renderHook, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { App } from './main';
import { ASSESSMENT_CACHE_KEY, useAssessment, type Job } from './useAssessment';

const ID='a'.repeat(32);
const SECOND_ID='b'.repeat(32);
const completed:Job={id:ID,status:'completed',stages:['Dataset validated','Certificate ready'],error:null};
const certificate={
  overall_status:'NOT_READY',
  executive_summary:'Synthetic test assessment: one annotation requirement failed.',
  disclaimer:'Synthetic software-test report only.',
  findings:[{requirement:{requirement_id:'R1',description:'Target boxes have positive area',severity:'hard'},state:'fail',
    reason:'Synthetic car annotation has zero height.',result:{metric_id:'bbox_invalid_rate',dimension:'validity',state:'pass',value:.1,reason:'One invalid box',evidence:{method:'Bounding-box validation'}}}],
  metrics:[{metric_id:'bbox_invalid_rate',dimension:'validity',state:'pass',value:.1,reason:'One synthetic invalid annotation',evidence:{method:'Bounding-box validation'}}],
  additional_benefits:['Required classes are present.'],limitations:['Annotation correctness is not established.'],
  recommendations:[{description:'Repair the zero-height annotation.'}],reproducibility:{engine_version:'test'},
};
const response=(body:unknown,status=200)=>new Response(JSON.stringify(body),{status,headers:{'Content-Type':'application/json'}});
function deferred<T>(){
  let resolve!:(value:T)=>void;
  const promise=new Promise<T>(done=>{resolve=done;});
  return {promise,resolve};
}
function remember(){sessionStorage.setItem(ASSESSMENT_CACHE_KEY,ID);}
function stubFetch(handler:(path:string)=>Promise<Response>){
  const fetcher=vi.fn((path:RequestInfo|URL)=>handler(String(path)));
  vi.stubGlobal('fetch',fetcher);
  return fetcher;
}
afterEach(()=>{vi.unstubAllGlobals();});

describe('completed assessment reports',()=>{
  it('shows UAV measurement status and derived learning views',async()=>{
    remember();
    const uav={...certificate,
      metrics:[{metric_id:'UAV-09',dimension:'coverage',state:'warning',assessment_status:'estimated',value:{count:2},reason:'Optional local model estimate',evidence:{sample_size:2}}],
      learning_views:{invariance_support:{interpretation:'Independent visibility review required'}}};
    stubFetch(async path=>response(path.endsWith('/certificate')?uav:path.endsWith('/progress')?completed:{}));
    render(<App/>);
    await screen.findByRole('heading',{name:'Learning-potential diagnostics'});
    expect(screen.getByText('Estimated')).toBeTruthy();
    expect(screen.getByText('Invariance support')).toBeTruthy();
  });
  it('renders a delayed certificate after polling changes running to completed',async()=>{
    remember();
    const pending=deferred<Response>();
    let polls=0;
    stubFetch(async path=>{
      if(path==='/api/config')return response({max_upload_bytes:2*1024**3,max_image_upload_bytes:2*1024**3});
      if(path.endsWith('/progress'))return response(++polls<3?{...completed,status:'running'}:completed);
      if(path.endsWith('/certificate'))return pending.promise;
      throw new Error('Unexpected request: '+path);
    });
    render(<App/>);
    await screen.findByRole('heading',{name:'Your assessment report is ready'});
    // Downloads must be visible even while React is still loading the report.
    expect(screen.getByRole('link',{name:'Download HTML report'}).getAttribute('href')).toBe('/api/assessments/'+ID+'/export?format=html');
    expect(screen.getByRole('link',{name:'Download JSON'}).getAttribute('href')).toBe('/api/assessments/'+ID+'/export?format=json');
    await screen.findByText('Loading the results and supporting evidence…');
    await act(async()=>{pending.resolve(response(certificate));});
    await screen.findByRole('heading',{name:'Your requirement findings'});
    expect(screen.getByText('Synthetic car annotation has zero height.')).toBeTruthy();
    expect(screen.getByRole('link',{name:'View results'}).getAttribute('href')).toBe('#assessment-results');
    expect(document.getElementById('assessment-results')).not.toBeNull();
  });

  it('handles an assessment that is already complete on first retrieval',async()=>{
    remember();
    stubFetch(async path=>response(path.endsWith('/certificate')?certificate:path.endsWith('/progress')?completed:{}));
    render(<App/>);
    await screen.findByRole('heading',{name:'Your requirement findings'});
    expect(screen.getByRole('link',{name:'Download HTML report'})).toBeTruthy();
  });

  it('keeps downloads visible on certificate failure and supports retry',async()=>{
    remember();let attempts=0;
    stubFetch(async path=>{
      if(path.endsWith('/certificate'))return ++attempts===1?response({detail:'Temporary report error'},503):response(certificate);
      return response(path.endsWith('/progress')?completed:{});
    });
    render(<App/>);
    await screen.findByText(/Temporary report error/);
    expect(screen.getByRole('link',{name:'Download HTML report'})).toBeTruthy();
    await userEvent.click(screen.getByRole('button',{name:'Retry loading results'}));
    await screen.findByRole('heading',{name:'Your requirement findings'});
    expect(attempts).toBe(2);
    expect(screen.queryByText(/Temporary report error/)).toBeNull();
  });

  it('restores an assessment after a frontend reload without uploading again',async()=>{
    window.history.replaceState(null,'','/?assessment='+ID);
    const fetcher=stubFetch(async path=>response(path.endsWith('/certificate')?certificate:path.endsWith('/progress')?completed:{}));
    const first=render(<App/>);
    await screen.findByRole('heading',{name:'Your requirement findings'});
    expect(sessionStorage.getItem(ASSESSMENT_CACHE_KEY)).toBe(ID);
    first.unmount();
    // Even if the query string is absent, the current tab remembers its job.
    window.history.replaceState(null,'','/');
    render(<App/>);
    await screen.findByRole('heading',{name:'Your requirement findings'});
    expect(new URL(window.location.href).searchParams.get('assessment')).toBe(ID);
    expect(fetcher.mock.calls.every(([path])=>!String(path).includes('/datasets/upload'))).toBe(true);
  });

  it('offers reports for assessment errors, not only READY outcomes',async()=>{
    remember();
    stubFetch(async path=>response(path.endsWith('/certificate')?{...certificate,overall_status:'ASSESSMENT_ERROR'}:path.endsWith('/progress')?{...completed,status:'assessment_error'}:{}));
    render(<App/>);
    await screen.findByRole('heading',{name:'Your requirement findings'});
    expect(screen.getByRole('link',{name:'Download JSON'})).toBeTruthy();
  });

  it('explains an expired assessment instead of silently hiding the report',async()=>{
    remember();
    stubFetch(async path=>path.endsWith('/progress')?response({detail:'Assessment not found or expired'},404):response({}));
    render(<App/>);
    await screen.findByText(/Assessment unavailable/);
    expect(screen.getByRole('button',{name:'Retry connection'})).toBeTruthy();
    expect(screen.queryByRole('link',{name:'Download HTML report'})).toBeNull();
  });

  it('discards a slow previous report after switching to a different assessment',async()=>{
    const old=deferred<unknown>();
    const api=async<T,>(path:string):Promise<T>=>{
      if(path.includes(ID)&&path.endsWith('/certificate'))return old.promise as Promise<T>;
      if(path.endsWith('/progress'))return {...completed,id:SECOND_ID} as T;
      return {id:SECOND_ID} as T;
    };
    const hook=renderHook(()=>useAssessment<{id:string}>(api));
    act(()=>hook.result.current.setJob(completed));
    await waitFor(()=>expect(hook.result.current.reportLoading).toBe(true));
    act(()=>hook.result.current.setJob({...completed,id:SECOND_ID,status:'running'}));
    await waitFor(()=>expect(hook.result.current.certificate?.id).toBe(SECOND_ID));
    await act(async()=>old.resolve({id:ID}));
    expect(hook.result.current.certificate?.id).toBe(SECOND_ID);
  });
});
