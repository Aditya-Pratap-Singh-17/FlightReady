import {test,expect} from '@playwright/test';
import path from 'node:path';

test('canonical report, linked real outcomes and inference acquisition remain connected on desktop and mobile',async({page,request})=>{
  const projects=await (await request.get('/api/v1/workflows')).json();
  const project=projects.find((p:any)=>p.payload.readiness_id&&p.payload.validation_id);
  expect(project).toBeTruthy();
  await page.goto('/');
  await page.evaluate((id:string)=>localStorage.setItem('wb-project',id),project.id);
  await page.goto('/#Readiness%20report');await page.reload();
  await expect(page.getByRole('heading',{name:'INDETERMINATE',exact:true})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Validation: completed'})).toBeVisible();
  const certificate=await (await request.get('/api/v1/jobs/'+project.payload.readiness_id)).json();
  const originalHash=certificate.payload.result.final_report.report_sha256;
  await page.getByLabel('Research detail').check();
  await expect(page.getByText(/not a confidence interval/).first()).toBeVisible();
  const shots=path.resolve('../.runtime/flightready-verification');
  await page.screenshot({path:path.join(shots,'canonical-desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
  await page.screenshot({path:path.join(shots,'canonical-mobile.png'),fullPage:true});
  await page.getByRole('link',{name:'Validate this finding'}).click();
  await page.getByRole('button',{name:'Inference-only benchmark',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Run a model on your images.'})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Import a model to test your data.'})).toBeVisible();
  await expect(page.getByText('apache-2.0',{exact:false})).toBeVisible();
  await expect(page.getByRole('button',{name:'Download & import model'})).toBeDisabled();
  if(await page.getByLabel('Cached inference model').inputValue()==='')await expect(page.getByRole('button',{name:'Run prediction preview'})).toBeDisabled();
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
  await page.screenshot({path:path.join(shots,'inference-mobile.png'),fullPage:true});
  const after=await (await request.get('/api/v1/jobs/'+project.payload.readiness_id)).json();
  expect(after.payload.result.final_report.report_sha256).toBe(originalHash);
});

test('VisDrone validation imagery fails drone-target compatibility before expensive analysis',async({page})=>{
  await page.goto('/');
  await page.getByLabel('Upload dataset',{exact:true}).setInputFiles('../.runtime/flightready-verification/visdrone-val.zip');
  await page.getByLabel('Describe your goal').fill('I want to detect drones at night.');
  await page.getByRole('button',{name:'Interpret goal',exact:true}).click();
  const preview=page.getByRole('region',{name:'Immediate task compatibility'});
  await expect(preview.getByText(/does not provide drone-object annotations/)).toBeVisible();
  await expect(preview.getByText(/Dataset role: val/)).toBeVisible();
  await preview.getByText('Inspect label and scenario evidence').click();
  await expect(preview.getByText(/insufficient evidence:/).first()).toBeVisible();
  await expect(page.getByRole('button',{name:'Analyze dataset',exact:true})).toHaveCount(0);
  await page.setViewportSize({width:390,height:844});
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
  await page.screenshot({path:'../.runtime/flightready-verification/visdrone-compatibility-mobile.png',fullPage:true});
});
