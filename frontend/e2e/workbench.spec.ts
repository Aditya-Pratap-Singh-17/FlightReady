import {test,expect} from '@playwright/test';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {existsSync} from 'node:fs';

test('real import, context, readiness, exports, responsive navigation',async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/#Overview');
  await expect(page.getByRole('heading',{name:'Overview',exact:true})).toBeVisible();
  await expect(page.getByText('Backend connected')).toBeVisible();
  await page.getByText('Workspace tools',{exact:true}).first().click();
  await page.getByRole('button',{name:'Datasets',exact:true}).click();
  const importedResponse=page.waitForResponse(r=>r.url().endsWith('/api/v1/datasets/import')&&r.request().method()==='POST');
  const localPython=path.resolve('../.venv/Scripts/python.exe');
  const python=process.env.PYTHON|| (existsSync(localPython)?localPython:'python');
  const fixture=execFileSync(python,['-c',[
    'import io,sys',
    'from pathlib import Path',
    'from tempfile import TemporaryDirectory',
    'from contextlib import redirect_stdout',
    'from backend.scripts.generate_demos import generate',
    'with TemporaryDirectory(prefix="flightready-e2e-",dir=Path.cwd()/".work") as temporary:',
    ' with redirect_stdout(io.StringIO()): generate(temporary)',
    ' sys.stdout.buffer.write((Path(temporary)/"detection_valid.zip").read_bytes())',
  ].join('\n')],{cwd:path.resolve('..')});
  await page.getByLabel('Upload dataset',{exact:true}).setInputFiles({name:'detection_valid.zip',mimeType:'application/zip',buffer:fixture});
  const imported=(await (await importedResponse).json());
  await expect(page.getByText('Dataset imported and inspected',{exact:true})).toBeVisible();
  await expect(page.locator('.wb-gallery img').first()).toBeVisible();
  await expect(page.locator('.wb-gallery svg rect').first()).toBeAttached();
  await page.getByRole('button',{name:'Evaluate dataset'}).click();
  await page.getByLabel('Intended use').fill('Detect car and pedestrian objects in synthetic UAV images.');
  await page.getByRole('button',{name:'Interpret context',exact:true}).click();
  await expect(page.getByLabel('Structured context JSON')).toBeVisible();
  const context=JSON.parse(await page.getByLabel('Structured context JSON').inputValue());
  context.goal.task_type='object_detection';context.goal.required_classes=['car','pedestrian'];context.clarification_questions=[];
  await page.getByLabel('Structured context JSON').fill(JSON.stringify(context));
  await page.getByRole('button',{name:'Apply context edits'}).click();
  await page.getByLabel('I confirm this context and its explicit requirements').check();
  await page.getByRole('button',{name:'Run existing readiness pipeline'}).click();
  await expect(page.getByText('CONTEXT-AWARE READINESS OUTCOME')).toBeVisible({timeout:90000});
  await expect(page.getByRole('heading',{name:'Every measured quality dimension'})).toBeVisible();
  const href=await page.locator('.wb-result a').filter({hasText:'JSON'}).getAttribute('href');
  const response=await page.request.get(href!);expect(response.status()).toBe(200);
  const report=await response.json();
  expect(report.payload.result.metrics.length).toBeGreaterThan(20);
  expect(report.payload.config.dataset_id).toBe(imported.id);
  await page.getByRole('button',{name:'Reports',exact:true}).click();
  await expect(page.getByRole('link',{name:'Complete research JSON'})).toBeVisible();
  await page.getByRole('button',{name:'Toggle color theme'}).click();
  await expect(page.locator('html')).toHaveAttribute('data-theme','dark');
  await page.screenshot({path:'../.runtime/browser-results/reports-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await page.getByRole('button',{name:'Overview',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Overview',exact:true})).toBeVisible();
  expect(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth)).toBe(true);
  await page.screenshot({path:'../.runtime/browser-results/overview-mobile.png',fullPage:true});
  expect(errors).toEqual([]);
});

test('real training controls, independent testing, comparisons and research views',async({page})=>{
  const existingModels=await (await page.request.get('/api/v1/models')).json();
  test.skip(existingModels.length<2,'Point WORKBENCH_URL at a backend using a completed Python smoke storage root.');
  const candidate=existingModels.find((m:{payload:{configuration:{model_type:string}}})=>m.payload.configuration.model_type==='candidate');
  const reference=existingModels.find((m:{payload:{configuration:{model_type:string}}})=>m.payload.configuration.model_type==='reference');
  const existingEvaluations=await (await page.request.get('/api/v1/evaluations')).json();
  const testDataset=existingEvaluations[0].payload.dataset_id;
  const splits=await (await page.request.get('/api/v1/splits?dataset_id='+candidate.payload.dataset_id)).json();
  const jobs=await (await page.request.get('/api/v1/jobs')).json();
  const readiness=jobs.find((j:{payload:{task:string;status:string;config:{split_id:string}}})=>j.payload.task==='readiness'&&j.payload.status==='completed'&&j.payload.config.split_id===splits[0].id);
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/#Model%20Training');
  await page.getByText('Workspace tools',{exact:true}).first().click();
  await page.getByLabel('Working dataset').selectOption(candidate.payload.dataset_id);
  await page.getByLabel('Split manifest',{exact:true}).selectOption(splits[0].id);
  await page.getByLabel('Training-only readiness report').selectOption(readiness.id);
  await page.getByLabel('Epochs',{exact:true}).fill('1');
  await page.getByLabel('Batch size',{exact:true}).fill('2');
  await page.getByLabel('Image size (multiple of 32)').fill('64');
  await page.getByLabel('Compute device').selectOption('cpu');
  const trainingResponse=page.waitForResponse(r=>r.url().endsWith('/api/v1/training')&&r.request().method()==='POST');
  await page.getByRole('button',{name:'Start training',exact:true}).click();
  const training=await (await trainingResponse).json();
  expect(training.id).toBeTruthy();
  await expect.poll(async()=>(await (await page.request.get('/api/v1/jobs/'+training.id)).json()).payload.status,{timeout:90000}).toBe('completed');
  const trained=(await (await page.request.get('/api/v1/jobs/'+training.id)).json()).payload.result.model_id;
  await page.getByRole('button',{name:'Model Comparison',exact:true}).click();
  await page.getByRole('checkbox',{name:new RegExp(trained.slice(0,8))}).check();
  await page.getByRole('checkbox',{name:new RegExp(reference.id.slice(0,8))}).check();
  await page.getByLabel('Independent test dataset').selectOption(testDataset);
  const evaluationResponse=page.waitForResponse(r=>r.url().endsWith('/api/v1/evaluations')&&r.request().method()==='POST');
  await page.getByRole('button',{name:'Evaluate selected models'}).click();
  const evaluation=await (await evaluationResponse).json();
  expect(evaluation.id).toBeTruthy();
  await expect.poll(async()=>(await (await page.request.get('/api/v1/jobs/'+evaluation.id)).json()).payload.status,{timeout:90000}).toBe('completed');
  const evaluated=(await (await page.request.get('/api/v1/jobs/'+evaluation.id)).json()).payload.result.evaluation_ids;
  await page.getByLabel('Model A evaluation').selectOption(evaluated[0]);
  await page.getByLabel('Model B evaluation').selectOption(evaluated[1]);
  await expect(page.getByText('Ground truth',{exact:true})).toBeVisible();
  await expect(page.locator('.wb-predictions img')).toHaveCount(3);
  await page.getByRole('button',{name:'Audit comparison conditions'}).click();
  await expect(page.getByText('Comparison recorded',{exact:true})).toBeVisible();
  await page.screenshot({path:'../.runtime/browser-results/comparison-desktop.png',fullPage:true});
  await page.getByRole('button',{name:'Research Analytics',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Cross-domain performance matrix'})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Evidence-based research summary'})).toBeVisible();
  await page.getByRole('button',{name:'Experiments',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Experiment registry'})).toBeVisible();
  await page.getByRole('button',{name:'Settings',exact:true}).click();
  await expect(page.getByRole('heading',{name:'Compute environment'})).toBeVisible();
  expect(errors).toEqual([]);
});
