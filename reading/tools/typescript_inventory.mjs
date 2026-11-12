import ts from '../../frontend/node_modules/typescript/lib/typescript.js';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const output=[];
for(const folder of ['frontend/src','frontend/e2e']){
 for(const file of fs.readdirSync(path.join(root,folder)).filter(f=>/\.tsx?$/.test(f)).sort()){
  const rel=folder+'/'+file,text=fs.readFileSync(path.join(root,rel),'utf8');
  const source=ts.createSourceFile(rel,text,ts.ScriptTarget.Latest,true,file.endsWith('tsx')?ts.ScriptKind.TSX:ts.ScriptKind.TS);
  const symbols=[],imports=[],calls=[];
  function walk(node,owner=''){
   if(ts.isImportDeclaration(node))imports.push(node.moduleSpecifier.text);
   if(ts.isCallExpression(node))calls.push({call:node.expression.getText(source),line:source.getLineAndCharacterOfPosition(node.getStart()).line+1});
   const named=ts.isFunctionDeclaration(node)||ts.isClassDeclaration(node)||ts.isInterfaceDeclaration(node)||ts.isTypeAliasDeclaration(node);
   const variable=ts.isVariableDeclaration(node)&&node.initializer&&(ts.isArrowFunction(node.initializer)||ts.isFunctionExpression(node.initializer));
   let next=owner;
   if((named&&node.name)||variable){
    const name=node.name.getText(source);next=owner?owner+'.'+name:name;
    const target=variable?node.initializer:node;
    symbols.push({name:next,kind:variable?'callback':ts.SyntaxKind[node.kind],line:source.getLineAndCharacterOfPosition(node.getStart()).line+1,
      signature:target.parameters?'('+target.parameters.map(p=>p.getText(source)).join(', ')+')':ts.isTypeAliasDeclaration(node)?node.type.getText(source):'',returns:target.type?.getText(source)||'inferred by TypeScript'});
   }
   ts.forEachChild(node,n=>walk(n,next));
  }
  walk(source);
  output.push({path:rel,lines:text.split('\n').length,imports,symbols,calls});
 }
}
fs.writeFileSync(path.join(root,'reading/assets/architecture/typescript-inventory.json'),JSON.stringify(output,null,2));
console.log(JSON.stringify({files:output.length,symbols:output.reduce((n,f)=>n+f.symbols.length,0)}));
