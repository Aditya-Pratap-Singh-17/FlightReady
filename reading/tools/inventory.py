"""Read-only repository analysis for the visual guide. Never imports the app."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'reading/assets/architecture'


def sources():
    paths = list(ROOT.glob('*.py'))
    for folder in ('backend', 'metrics', 'sql', 'tests'):
        paths += list((ROOT/folder).rglob('*.py'))
    return sorted(p for p in paths if '__pycache__' not in p.parts)


def expr(node):
    return ast.unparse(node) if node is not None else ''


def inventory():
    files = []
    for path in sources():
        text = path.read_text(encoding='utf-8-sig')
        tree = ast.parse(text, filename=str(path))
        symbols, imports, routes = [], [], []
        def visit(items, owner=''):
            for node in items:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    qualified = owner+'.'+node.name if owner else node.name
                    calls = sorted({expr(n.func) for n in ast.walk(node) if isinstance(n, ast.Call)})
                    fields = [{ 'name': expr(n.target), 'type': expr(n.annotation), 'default': expr(n.value)}
                              for n in node.body if isinstance(n, ast.AnnAssign)]
                    symbols.append({'name': qualified, 'kind': 'class' if isinstance(node, ast.ClassDef) else 'async function' if isinstance(node, ast.AsyncFunctionDef) else 'function',
                        'line': node.lineno, 'end': node.end_lineno, 'signature': ('('+', '.join(expr(b) for b in node.bases)+')') if isinstance(node, ast.ClassDef) else '('+ast.unparse(node.args)+')',
                        'returns': expr(getattr(node, 'returns', None)), 'doc': ast.get_docstring(node) or '',
                        'decorators': [expr(d) for d in node.decorator_list], 'calls': calls, 'fields': fields})
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and dec.func.attr in ('get','post','put','patch','delete') and dec.args and isinstance(dec.args[0], ast.Constant):
                            routes.append({'method': dec.func.attr.upper(), 'path': dec.args[0].value, 'function': qualified, 'line': node.lineno, 'signature': ast.unparse(node.args), 'response': expr(getattr(node,'returns',None)), 'decorator': expr(dec), 'calls': calls})
                    visit(node.body, qualified)
        visit(tree.body)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports += [{'module': a.name, 'names': [], 'line': node.lineno} for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports.append({'module': '.'*node.level+(node.module or ''), 'names': [a.name for a in node.names], 'line': node.lineno})
        files.append({'path': path.relative_to(ROOT).as_posix(), 'lines': len(text.splitlines()),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'doc': ast.get_docstring(tree) or '', 'imports': imports, 'symbols': symbols, 'routes': routes})
    hashes = {f['path']: f['sha256'] for f in files}
    for folder in ('frontend/src','frontend/e2e','metrics/config'):
        for p in sorted((ROOT/folder).rglob('*')):
            if p.is_file() and p.suffix in ('.tsx','.ts','.css','.json'):
                hashes[p.relative_to(ROOT).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    for filename in ('frontend/package.json','frontend/package-lock.json','frontend/vite.config.ts','frontend/vitest.config.ts','frontend/playwright.config.ts','frontend/tsconfig.json','requirements/requirements.txt','requirements/requirements-training.txt','requirements/requirements-dev.txt','pytest.ini','pyrightconfig.json'):
        p=ROOT/filename
        if p.exists(): hashes[filename]=hashlib.sha256(p.read_bytes()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'source-inventory.json').write_text(json.dumps({'python':files,'source_hashes':hashes},indent=2),encoding='utf-8')
    return files


if __name__=='__main__':
    files=inventory()
    print(json.dumps({'python_files':len(files),'symbols':sum(len(f['symbols']) for f in files),'routes':sum(len(f['routes']) for f in files)}))
