"""Literal-only settings and verified existing public source lookup."""
import ast,hashlib,importlib.util,json,sys
from pathlib import Path

def repository_root():
    return next(p for p in Path(__file__).resolve().parents if p.name=='experiments').parent

def read_settings(path):
    path=Path(path)
    if path.suffix.lower()=='.json':
        return json.loads(path.read_text(encoding='utf-8-sig'))
    if path.suffix.lower()!='.py':raise ValueError('Use a Python settings file or an external JSON file')
    tree=ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    nodes=[n for n in tree.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str))]
    if len(nodes)!=1 or not isinstance(nodes[0],ast.Assign) or len(nodes[0].targets)!=1 or not isinstance(nodes[0].targets[0],ast.Name) or nodes[0].targets[0].id!='CONFIG':
        raise ValueError('Settings must contain only a literal CONFIG assignment')
    value=ast.literal_eval(nodes[0].value)
    def resolve(x):
        if isinstance(x,dict):return {k:resolve(v) for k,v in x.items()}
        if isinstance(x,list):return [resolve(v) for v in x]
        if isinstance(x,str):return x.replace('${GARQ_REPOSITORY_ROOT}',str(repository_root()))
        return x
    return resolve(value)

def public_file(relative,expected_sha256):
    path=repository_root()/relative
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError('Public source differs from the required scientific version: '+str(path))
    return path

def public_module(relative,expected_sha256,name):
    path=public_file(relative,expected_sha256)
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module
