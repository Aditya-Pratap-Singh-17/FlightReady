import React from 'react';
import {render,screen,waitFor} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {expect,it,vi} from 'vitest';
import {ExperimentalScore,type WeightedScore} from './ExperimentalScore';

const fixture:WeightedScore={experimental:true,status:'partial',score_percent:75,evidence_coverage_percent:80,
  score_range_percent:[60,80],notice:'Experimental',hard_failure_ids:['hard-failure'],components:[
    {id:'validity',label:'Annotation validity',weight:3,score_percent:100,effective_weight_percent:75,
      contribution_points:75,status:'measured',reason:'Known evidence',raw_value:0},
    {id:'clarity',label:'Image clarity',weight:1,score_percent:0,effective_weight_percent:25,
      contribution_points:0,status:'measured',reason:'Known evidence',raw_value:0},
    {id:'coverage',label:'Scenario coverage',weight:1,score_percent:null,effective_weight_percent:0,
      contribution_points:null,status:'insufficient_evidence',reason:'No scenarios configured',raw_value:null}]};

it('shows coverage, unknown evidence and hard failures separately from the score',()=>{
  render(<ExperimentalScore score={fixture}/>);
  expect(screen.getByText('75.00 / 100')).toBeTruthy();
  expect(screen.getByText('80.00%')).toBeTruthy();
  expect(screen.getByText(/full score could range from 60.00 to 80.00/)).toBeTruthy();
  expect(screen.getByText(/Hard requirements failed/)).toBeTruthy();
  expect(screen.getByText('No scenarios configured')).toBeTruthy();
  expect(screen.getAllByText('Unavailable').length).toBeGreaterThan(0);
});

it('previews changed weights through the server and resets to saved evidence',async()=>{
  const onPreview=vi.fn(async()=>({...fixture,score_percent:25}));
  render(<ExperimentalScore score={fixture} onPreview={onPreview}/>);
  await userEvent.click(screen.getByText('Edit weights and inspect contributions'));
  const weight=screen.getByLabelText('Annotation validity weight');
  await userEvent.clear(weight);await userEvent.type(weight,'1');
  await userEvent.click(screen.getByRole('button',{name:'Preview weights'}));
  await screen.findByText('25.00 / 100');
  expect(onPreview).toHaveBeenCalledWith({validity:1,clarity:1,coverage:1});
  await userEvent.click(screen.getByRole('button',{name:'Reset preview'}));
  expect(screen.getByText('75.00 / 100')).toBeTruthy();
  expect(fixture.components[0].weight).toBe(3);
});

it('rejects empty weights and reports API failures',async()=>{
  const onPreview=vi.fn(async()=>{throw new Error('Invalid score profile');});
  render(<ExperimentalScore score={fixture} onPreview={onPreview}/>);
  await userEvent.click(screen.getByText('Edit weights and inspect contributions'));
  const weight=screen.getByLabelText('Annotation validity weight');
  await userEvent.clear(weight);
  await userEvent.click(screen.getByRole('button',{name:'Preview weights'}));
  expect(screen.getByRole('alert').textContent).toContain('Enter finite weights');
  expect(onPreview).not.toHaveBeenCalled();
  await userEvent.type(weight,'0');
  await userEvent.click(screen.getByRole('button',{name:'Preview weights'}));
  await waitFor(()=>expect(screen.getByRole('alert').textContent).toBe('Invalid score profile'));
});

it('can preview a historical report without a stored score',async()=>{
  const onPreview=vi.fn(async()=>fixture);
  render(<ExperimentalScore onPreview={onPreview}/>);
  await userEvent.click(screen.getByRole('button',{name:'Preview experimental score for this report'}));
  await screen.findByText('75.00 / 100');
  expect(onPreview).toHaveBeenCalledWith({});
});

it('hides a disabled experiment and never renders missing score as zero',()=>{
  const view=render(<ExperimentalScore score={{...fixture,status:'disabled'}}/>);
  expect(screen.queryByRole('heading')).toBeNull();
  view.rerender(<ExperimentalScore score={{...fixture,status:'unavailable',score_percent:null}}/>);
  expect(screen.queryByText('0.00 / 100')).toBeNull();
  expect(screen.getAllByText('Unavailable').length).toBeGreaterThan(0);
});
