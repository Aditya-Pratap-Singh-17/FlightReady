import { useEffect, useState } from 'react';

export type Job = {id:string;status:string;stages:string[];error:string|null};
type ApiClient = <T>(path:string, options?:RequestInit)=>Promise<T>;
export const ASSESSMENT_CACHE_KEY='flightready:last-assessment';
export const hasReport=(status:string)=>['completed','assessment_error'].includes(status);
const isPending=(status:string)=>['restoring','queued','running'].includes(status);
const message=(error:unknown)=>error instanceof Error?error.message:'The request failed. Please retry.';

function savedJob():Job|null{
  // Keep a reload from losing the only route to a report after a large upload.
  // This stores an ID, never the dataset or its certificate in browser storage.
  try{
    const id=new URL(window.location.href).searchParams.get('assessment')||sessionStorage.getItem(ASSESSMENT_CACHE_KEY);
    if(id&&/^[a-f0-9]{32}$/i.test(id))return {id,status:'restoring',stages:[],error:null};
  }catch{/* Browser storage can be disabled; normal assessment still works. */}
  return null;
}

export function useAssessment<TCertificate>(api:ApiClient){
  const [job,setJob]=useState<Job|null>(savedJob);
  const [certificate,setCertificate]=useState<TCertificate|null>(null);
  const [progressError,setProgressError]=useState('');
  const [reportError,setReportError]=useState('');
  const [reportLoading,setReportLoading]=useState(false);
  const [reportRetry,setReportRetry]=useState(0);
  const [progressRetry,setProgressRetry]=useState(0);
  const id=job?.id;
  const status=job?.status;

  useEffect(()=>{
    try{
      const url=new URL(window.location.href);
      if(id){sessionStorage.setItem(ASSESSMENT_CACHE_KEY,id);url.searchParams.set('assessment',id);}
      else{sessionStorage.removeItem(ASSESSMENT_CACHE_KEY);url.searchParams.delete('assessment');}
      window.history.replaceState(null,'',url);
    }catch{/* Do not block results if storage/history is unavailable. */}
  },[id]);

  useEffect(()=>{
    setCertificate(null);setReportError('');setProgressError('');
  },[id]);

  useEffect(()=>{
    if(!id||!status||!isPending(status))return;
    let disposed=false;
    let timer:ReturnType<typeof setTimeout>|undefined;
    const controller=new AbortController();
    async function poll(){
      try{
        const next=await api<Job>('/assessments/'+id+'/progress',{signal:controller.signal});
        if(disposed)return;
        setProgressError('');setJob(next);
        if(isPending(next.status))timer=setTimeout(poll,600);
      }catch(error){
        if(disposed)return;
        if((error as {status?:number})?.status===404){
          const reason='Assessment unavailable. The backend may have restarted or this session may have expired.';
          setProgressError(reason);
          setJob(previous=>previous&&previous.id===id?{...previous,status:'unavailable',error:reason}:previous);
        }else{
          setProgressError(message(error));
          timer=setTimeout(poll,2000);
        }
      }
    }
    // One request at a time prevents late running responses from replacing a
    // newer completed state. Timers represent polling, not fabricated progress.
    void poll();
    return ()=>{disposed=true;controller.abort();if(timer)clearTimeout(timer);};
  },[api,id,status,progressRetry]);

  useEffect(()=>{
    if(!id||!status||!hasReport(status)){setReportLoading(false);return;}
    let disposed=false;
    const controller=new AbortController();
    setReportLoading(true);setReportError('');
    async function load(){
      try{
        const report=await api<TCertificate>('/assessments/'+id+'/certificate',{signal:controller.signal});
        if(!disposed)setCertificate(report);
      }catch(error){
        if(!disposed)setReportError(message(error));
      }finally{
        if(!disposed)setReportLoading(false);
      }
    }
    // This effect belongs to the completed job. Cleaning up the progress poll
    // cannot discard the certificate response, which caused the missing report.
    void load();
    return ()=>{disposed=true;controller.abort();};
  },[api,id,status,reportRetry]);

  function retryProgress(){
    setJob(previous=>previous?{...previous,status:'restoring'}:previous);
    setProgressRetry(n=>n+1);
  }
  return {job,setJob,certificate,setCertificate,reportLoading,reportError,progressError,
    retryReport:()=>setReportRetry(n=>n+1),retryProgress};
}
