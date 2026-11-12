import React from 'react';
import {render,screen,waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {QueryClient,QueryClientProvider} from '@tanstack/react-query';
import {afterEach,expect,it,vi} from 'vitest';
import {CompatibilityPreview,FinalReadiness,type FinalReport} from './FinalReadiness';

const wrap=(content:React.ReactNode)=>render(<QueryClientProvider client={new QueryClient({defaultOptions:{queries:{retry:false}}})}>{content}</QueryClientProvider>);
afterEach(()=>vi.unstubAllGlobals());

it('shows the early unsupported drone-label finding and preserves unknown nighttime evidence',async()=>{
  vi.stubGlobal('fetch',vi.fn(async()=>new Response(JSON.stringify({status:'incompatible',summary:'Drone-captured imagery does not provide drone-object annotations.',annotated_classes:['car','pedestrian'],dataset_role:'val',training_partition_available:false,
    checks:[{state:'insufficient_evidence',reason:'Nighttime metadata is unavailable; presence and absence remain unverified.',requirement:{description:'Nighttime coverage'}}],limitations:[]}),{status:200})));
  wrap(<CompatibilityPreview datasetId="visdrone-val" context={{original_text:'Detect drones at night',goal:{task_type:'object_detection'},confirmed:true,provider:'deterministic',clarification_questions:[]}}/>);
  expect(await screen.findByText('Drone-captured imagery does not provide drone-object annotations.')).toBeTruthy();
  await userEvent.click(screen.getByText('Inspect label and scenario evidence'));
  expect(screen.getByText(/presence and absence remain unverified/)).toBeTruthy();
});

it('leads with the conservative final verdict and links immutable exports plus separate outcomes',async()=>{
  vi.stubGlobal('fetch',vi.fn(async()=>new Response(JSON.stringify({status:'not_yet_validated',interpretation:'Original analytical evidence is unchanged.',inference_benchmarks:[],evaluations:[],comparisons:[]}),{status:200})));
  const report={report_version:'final-readiness-1.0',report_sha256:'report-hash',decision:'INDETERMINATE',conclusion:'Required evidence is missing.',dataset:{name:'Example',sha256:'immutable-version'},compatibility:{status:'unverified',summary:'Nighttime evidence is unavailable.',annotated_classes:['car'],dataset_role:'val',training_partition_available:false,checks:[],limitations:[]},technical_requirement_status:'READY',evidence_coverage:.42,critical_blockers:0,
    criterion_attainment_index:{reason:'No justified overall index is defined.'},top_problems:[{priority:'mandatory evidence gap',finding_id:'night',reason:'Missing night metadata',action:'Supply verified capture metadata.'}],recommendations:[],dimensions:[{id:'annotation',name:'Annotation quality',score:null,status:'insufficient_evidence',evidence_coverage:.5,effective_weight:1,bounds:[0,100]}],validation:{status:'blocked',reason:'Resolve compatibility first.',inference_allowed:false,training_allowed:false},limitations:['Coverage is not confidence.'],coverage_definition:'Weighted criterion coverage.'} as FinalReport;
  const inspect=vi.fn();wrap(<FinalReadiness report={report} readinessId="assessment-1" onEvidence={inspect}/>);
  expect(screen.getByRole('heading',{name:'INDETERMINATE'})).toBeTruthy();
  expect(screen.getByText('42.0%')).toBeTruthy();
  expect(screen.getByRole('link',{name:'PDF'}).getAttribute('href')).toBe('/api/v1/reports/assessment-1?format=pdf');
  await userEvent.click(screen.getByRole('button',{name:'Inspect supporting evidence'}));expect(inspect).toHaveBeenCalledOnce();
  await userEvent.click(screen.getByLabelText('Research detail'));
  expect(screen.getByText(/Index bounds 0.0–100.0; not a confidence interval/)).toBeTruthy();
  await waitFor(()=>expect(screen.getByText('Original analytical evidence is unchanged.')).toBeTruthy());
});
