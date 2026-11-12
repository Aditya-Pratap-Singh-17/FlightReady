import {defineConfig} from '@playwright/test';
export default defineConfig({
  testDir:'./e2e',timeout:120000,workers:1,
  use:{baseURL:process.env.WORKBENCH_URL||'http://127.0.0.1:8017',headless:true,channel:'msedge',viewport:{width:1440,height:1000}},
  outputDir:'../.runtime/browser-results',
});
