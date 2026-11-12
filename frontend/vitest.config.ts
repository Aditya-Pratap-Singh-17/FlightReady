import { defineConfig } from 'vitest/config';

export default defineConfig({test:{
  exclude:['e2e/**','node_modules/**'],
  environment:'jsdom',
  environmentOptions:{jsdom:{url:'http://localhost:5173/'}},
  setupFiles:['./src/testSetup.ts'],
  restoreMocks:true,
}});
