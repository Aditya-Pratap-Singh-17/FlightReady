import React from 'react';
import {render,screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {expect,it,vi} from 'vitest';
import {ResearchEvidence,type EvidenceSection} from './ResearchEvidence';
import type {Certificate} from './api';

vi.mock('recharts',()=>({ResponsiveContainer:({children}:{children:React.ReactNode})=><div>{children}</div>,BarChart:({data}:{data:unknown})=><div data-testid="actual-chart-data">{JSON.stringify(data)}</div>,Bar:()=>null,XAxis:()=>null,YAxis:()=>null,Tooltip:()=>null}));
const section:EvidenceSection={id:'object_scale',title:'Object-scale distributions',metric_id:'UAV-08',status:'measured',evidence_type:'observed annotation geometry',sample_size:3,coverage:.5,denominator:'valid annotated objects',version:'test-only',provenance:{dataset_sha256:'fixture'},limitations:['Geometry does not establish detectability.'],plots:[{title:'Native area bins',unit:'objects/bin',denominator:3,status:'measured',rows:[{label:'small',value:0},{label:'large',value:3}]}],rows:[]};
function cert(sections:EvidenceSection[]){return {quality_measurements:{distributions:{},research_evidence:{version:'test',notice:'Synthetic fixture',sections}}} as unknown as Certificate;}

it('shows recorded zeros with units, denominator, sample and coverage',()=>{
  render(<ResearchEvidence certificate={cert([section])}/>);
  expect(screen.getByTestId('actual-chart-data').textContent).toContain('"value":0');
  expect(screen.getByText(/Sample size: 3/).textContent).toContain('50.0%');
  expect(screen.getByText(/Units: objects\/bin/).textContent).toContain('Denominator: 3');
  expect(screen.getByText('Geometry does not establish detectability.')).toBeTruthy();
});

it('keeps unsupported evidence visible without drawing a zero chart',async()=>{
  const unknown={...section,id:'annotation_audit',title:'Independent annotation audit',status:'insufficient_evidence',sample_size:null,coverage:null,plots:[{title:'Audit errors',unit:'errors',denominator:null,status:'insufficient_evidence',rows:[]}]};
  render(<ResearchEvidence certificate={cert([section,unknown])}/>);
  await userEvent.selectOptions(screen.getByLabelText('Evidence view'),'annotation_audit');
  expect(screen.getByRole('status').textContent).toContain('This does not imply zero');
  expect(screen.queryByTestId('actual-chart-data')).toBeNull();
});

it('preserves historical certificates as historical evidence',()=>{
  render(<ResearchEvidence certificate={{} as Certificate}/>);
  expect(screen.getByText(/unavailable in this historical certificate/)).toBeTruthy();
});
