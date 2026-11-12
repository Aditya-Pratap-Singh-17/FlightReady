import {test,expect} from '@playwright/test';

test('imports a local pretrained transformer, auto-selects it and runs preview and annotation metrics without readiness',async({page})=>{
  test.setTimeout(120000);
  await page.goto('/');
  await page.getByLabel('Upload dataset',{exact:true}).setInputFiles('../.runtime/flightready-verification/visdrone-val.zip');
  await expect(page.getByRole('heading',{name:'What do you want to achieve?'})).toBeVisible();
  await page.getByRole('navigation',{name:'Main navigation'}).getByRole('button',{name:'Model Validation Lab'}).click();
  await expect(page.getByRole('heading',{name:'Import a model to test your data.'})).toBeVisible();
  await expect(page.getByRole('heading',{name:'Run a model on your images.'})).toBeVisible();
  await page.getByRole('button',{name:'Local model folder',exact:true}).click();
  await page.getByLabel('Local Transformers model folder').fill(process.env.IMPORT_MODEL_FOLDER||'C:/DataReadiness/.runtime/workbench/hubcache/models--hustvl--yolos-tiny/snapshots/95a90f3c189fbfca3bcfc6d7315b9e84d95dc2de');
  await page.getByLabel('I have reviewed the local model terms and have permission to use it.').check();
  await page.getByRole('button',{name:'Import local Transformers model',exact:true}).click();
  await expect(page.getByLabel('Cached inference model')).not.toHaveValue('');
  await page.getByLabel('Maximum preview images').fill('1');
  await page.getByRole('button',{name:'Run prediction preview',exact:true}).click();
  const preview=page.locator('summary').filter({hasText:/Prediction preview .*quality not scored/});
  await expect(preview).toBeVisible({timeout:60000});await preview.click();
  await expect(page.getByText('Detection quality is not scored in this preview.',{exact:false})).toBeVisible();
  await page.getByLabel('Inference purpose').selectOption('benchmark');
  await page.getByRole('button',{name:'Run real inference & evaluation',exact:true}).click();
  await expect(page.locator('summary').filter({hasText:/Annotation benchmark .*mAP50:95/})).toBeVisible({timeout:60000});
  await page.screenshot({path:'../.runtime/model-import-verification/model-import-desktop.png',fullPage:true});
  await page.setViewportSize({width:390,height:844});
  await expect.poll(()=>page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth)).toBe(true);
  await page.screenshot({path:'../.runtime/model-import-verification/model-import-mobile.png',fullPage:true});
});
