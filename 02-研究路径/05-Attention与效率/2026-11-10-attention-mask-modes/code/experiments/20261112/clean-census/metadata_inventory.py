"""New native-coordinate/permission census over the sealed completed graph.

This never reconstructs the old case graph, tokenizes data or loads a model.
It does not admit training or assert semantic decontamination.
"""
import array
import base64
from collections import Counter
from datetime import datetime,timezone
from functools import lru_cache
import fcntl
import json
import mmap
import os
from pathlib import Path
import re
import resource
import shutil
import signal
import sqlite3
import struct
import subprocess
import sys
import time

ROOT=Path(os.environ.get('CENSUS_ROOT','/tmp/opencode/jev-clean-census-20261112-v1'))
AUDIT=Path('/tmp/opencode/jev-06b-research-20261112-validation-audit-v2')
HOME=Path('/home/dev/assumeengage')
START=time.monotonic()
OWNED=False


def require(ok,message):
    if not ok:raise RuntimeError(message)


def utc():return datetime.now(timezone.utc).isoformat()


def read(path):return json.loads(Path(path).read_text())


def metadata(path):
    path=Path(path);s=path.stat();require(path.is_file(),'Expected file: '+str(path))
    return {'path':str(path),'resolved':str(path.resolve()),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}


def verify(files):
    for item in files:require(metadata(item['path'])==item,'Source changed: '+item['path'])


def atomic(path,value):
    temp=path.with_name(path.name+f'.{os.getpid()}.writing')
    with temp.open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def budget():
    require(not (ROOT/'NO_AUTOMATIC_RESUME').exists(),'CPU census stopped')
    require(time.monotonic()-START<2700,'45min CPU census budget exhausted')
    require(shutil.disk_usage(ROOT).free>=20*2**30,'Scratch free space below20GiB')
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<6*2**20,'CPU census RSS exceeds6GiB')


def family(source):
    value=re.sub(r'(?i)(?:[/_:.-](?:train|training|test|dev|valid|validation))$','',str(source).strip()).casefold()
    for name in ('weblinx','mind2web','websrc'):
        if value==name or value.startswith(name+'/'):return name
    return value


def main():
    global OWNED
    require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU census must hide CUDA')
    os.umask(0o077);os.setpriority(os.PRIO_PROCESS,0,19)
    cpus=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,set(cpus[-2:]))
    subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True,capture_output=True)
    require(ROOT.is_dir() and not ROOT.is_symlink(),'Private exclusive namespace required')
    lock=(ROOT/'RUN.lock').open('a+')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    OWNED=True
    require(not any((ROOT/n).exists() for n in ('SOURCE_BINDING.json','native-coordinates.sqlite3','COMPLETE.json','STATUS.json')),'Existing census assets preserved')
    def interrupted(signum,frame):raise InterruptedError('CPU census interrupted; partial preserved')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    signal.signal(signal.SIGALRM,interrupted);signal.setitimer(signal.ITIMER_REAL,2700)
    source=read(AUDIT/'SOURCE_BINDING.json');done=read(AUDIT/'COMPLETE.json');summary=read(AUDIT/'SUMMARY.json')
    require(done['complete'] and summary['complete_original_scan'] and summary['graph_nodes']==5561805,'Existing audit not complete')
    files=source['files']+done['files']+[done['source_binding'],done['code_binding'],metadata(AUDIT/'COMPLETE.json')]
    files += [metadata(HOME/'scripts/pair_data_common.py'),metadata(HOME/'config/source-domain-overrides.json')]
    unique={}
    for value in files:
        require(value['path'] not in unique or unique[value['path']]==value,'Conflicting inherited source binding: '+value['path'])
        unique[value['path']]=value
    files=[unique[k] for k in sorted(unique)];verify(files)
    expected=read(ROOT/'CODE_BINDING.json')
    actual={str(p.relative_to(ROOT/'code')):base64.b64encode(p.read_bytes()).decode() for p in sorted((ROOT/'code').glob('*.py'))}
    require(actual==expected,'Census code changed')
    atomic(ROOT/'SOURCE_BINDING.json',{'files':files,'old_audit':str(AUDIT),'purpose':'new complete native-coordinate/license census, no old graph reconstruction'})
    sys.path.insert(0,str(HOME/'scripts'))
    from pair_data_common import domain_for
    import pyarrow as pa
    import pyarrow.parquet as pq
    pa.set_cpu_count(1);pa.set_io_thread_count(1)
    overrides=read(HOME/'config/source-domain-overrides.json')
    old=sqlite3.connect((AUDIT/'case-graph.sqlite3').as_uri()+'?mode=ro',uri=True);old.execute('PRAGMA query_only=ON')
    handles=[(AUDIT/n).open('rb') for n in ('parent-int64.bin','flags-uint8.bin')]
    parents,flags=[mmap.mmap(h.fileno(),0,access=mmap.ACCESS_READ) for h in handles]
    require(sys.byteorder=='little' and len(parents)==5561805*8 and len(flags)==5561805,'Audited array layout differs')
    @lru_cache(maxsize=262144)
    def final_root(node):
        for _ in range(128):
            parent=struct.unpack_from('<q',parents,node*8)[0]
            require(0<=parent<5561805,'Parent out of bounds')
            if parent==node:return node
            node=parent
        raise RuntimeError('Old graph depth/cycle exceeds bound')
    @lru_cache(maxsize=131072)
    def lookup_group(f,group):
        key=json.dumps([f,group],ensure_ascii=False,separators=(',',':'))
        row=old.execute('SELECT node FROM keys WHERE kind=? AND value=?',('native_group',key)).fetchone()
        require(row is not None,'Source-bound native_group absent from old audit')
        root=final_root(row[0]);return root,int(flags[root])
    db=sqlite3.connect(ROOT/'native-coordinates.sqlite3')
    db.executescript('''PRAGMA journal_mode=DELETE;PRAGMA synchronous=FULL;PRAGMA temp_store=FILE;PRAGMA cache_size=-32768;
      CREATE TABLE files(file_id INTEGER PRIMARY KEY,path TEXT UNIQUE,native_file_split TEXT,expected_rows INTEGER);
      CREATE TABLE sources(source_id INTEGER PRIMARY KEY,source TEXT UNIQUE,family TEXT,domain TEXT);
      CREATE TABLE permissions(permission_id INTEGER PRIMARY KEY,source_id INTEGER,license TEXT,license_use TEXT,UNIQUE(source_id,license,license_use));
      CREATE TABLE rows(file_id INTEGER,row_index INTEGER,old_root INTEGER,old_bits INTEGER,source_id INTEGER,permission_id INTEGER,kind TEXT,declared_split TEXT,group_present INTEGER,PRIMARY KEY(file_id,row_index)) WITHOUT ROWID;
    ''')
    sources={};permissions={};source_counts=Counter();permission_counts=Counter();split_counts=Counter();kind_counts=Counter();missing=Counter();totals=Counter();first={}
    try:
        entries=[('train',p) for p in source['train_paths']]+[('validation',p) for p in source['validation_paths']]
        for file_id,(split,path) in enumerate(entries):
            budget();parquet=pq.ParquetFile(path);expected_rows=summary['native_scan_rows'][path]
            require(parquet.metadata.num_rows==expected_rows,'Source-bound row count differs')
            needed=['source','group_id','kind','split','license','license_use']
            require(set(needed)<=set(parquet.schema_arrow.names),'Native metadata schema missing')
            db.execute('INSERT INTO files VALUES(?,?,?,?)',(file_id,path,split,expected_rows));offset=0
            for batch in parquet.iter_batches(batch_size=4096,columns=needed,use_threads=False):
                records=[]
                for raw in batch.to_pylist():
                    src=raw['source'] or '';f=family(src)
                    if src not in sources:
                        sid=len(sources);sources[src]=sid;domain=overrides.get(src,domain_for(src))
                        db.execute('INSERT INTO sources VALUES(?,?,?,?)',(sid,src,f,domain))
                    sid=sources[src];lic=raw['license'];use=raw['license_use'];permission=(sid,lic,use)
                    if permission not in permissions:
                        pid=len(permissions);permissions[permission]=pid;db.execute('INSERT INTO permissions VALUES(?,?,?,?)',(pid,sid,lic,use))
                    pid=permissions[permission]
                    gid=raw['group_id'];present=bool(f) and gid is not None and not isinstance(gid,bool) and bool(str(gid).strip())
                    root,bits=lookup_group(f,str(gid)) if present else (None,16)
                    records.append((file_id,offset,root,bits,sid,pid,raw['kind'],raw['split'],int(present)))
                    source_counts[(split,sid)]+=1;permission_counts[(split,pid)]+=1;split_counts[(split,raw['split'])]+=1;kind_counts[(split,raw['kind'])]+=1
                    if not present:missing[(split,sid)]+=1
                    first.setdefault(sid,{'file':path,'row_index':offset,'native_file_split':split})
                    offset+=1
                db.executemany('INSERT INTO rows VALUES(?,?,?,?,?,?,?,?,?)',records);db.commit();budget()
                if offset%65536<len(records):atomic(ROOT/'STATUS.json',{'complete':False,'stage':'new_coordinate_census','file':path,'rows':offset,'expected_rows':expected_rows,'elapsed_seconds':time.monotonic()-START,'utc':utc()})
            require(offset==expected_rows,'Native coordinate inventory incomplete');totals[split]+=offset
        require(dict(totals)=={'train':2529342,'validation':15550},'Full coordinate census totals differ')
        budget();db.execute('CREATE INDEX rows_root ON rows(old_root)');db.execute('CREATE INDEX rows_source ON rows(source_id)');db.commit()
        require(db.execute('SELECT count(*) FROM rows').fetchone()[0]==2544892,'Actual stored coordinate count differs')
        source_table=[{'source_id':sid,'source':src,'family':family(src),'domain':overrides.get(src,domain_for(src)),
            'train_rows':source_counts[('train',sid)],'validation_rows':source_counts[('validation',sid)],
            'missing_group_train_rows':missing[('train',sid)],'missing_group_validation_rows':missing[('validation',sid)],'first_coordinate':first[sid]} for src,sid in sources.items()]
        permission_table=[{'permission_id':pid,'source_id':key[0],'license':key[1],'license_use':key[2],
            'train_rows':permission_counts[('train',pid)],'validation_rows':permission_counts[('validation',pid)]} for key,pid in permissions.items()]
        verify(files);budget()
        atomic(ROOT/'INVENTORY.json',{'complete_coordinate_inventory':True,'DATA_READY':False,'native_rows':dict(totals),
            'sources':source_table,'permissions':permission_table,'kind_counts':[{'file_split':k[0],'kind':k[1],'rows':v} for k,v in kind_counts.items()],
            'declared_split_counts':[{'file_split':k[0],'declared_split':k[1],'rows':v} for k,v in split_counts.items()],
            'old_graph_mutated':False,'semantic_input_filter_complete':False,'license_review_complete':False,
            'tokenized':False,'model_run':False,'cpu_affinity':sorted(os.sched_getaffinity(0)),
            'elapsed_seconds':time.monotonic()-START,'max_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024})
    finally:
        db.close();old.close();parents.close();flags.close()
        for h in handles:h.close()
    verify(files);budget()
    actual={str(p.relative_to(ROOT/'code')):base64.b64encode(p.read_bytes()).decode() for p in sorted((ROOT/'code').glob('*.py'))}
    require(actual==expected,'Census code changed during execution')
    result={'complete':True,'DATA_READY':False,'stage':'native_coordinate_metadata_only','native_rows':dict(totals),
        'source_count':len(sources),'permission_combinations':len(permissions),'elapsed_seconds':time.monotonic()-START,
        'files':[metadata(ROOT/n) for n in ('native-coordinates.sqlite3','INVENTORY.json','SOURCE_BINDING.json')],
        'semantic_input_filter_complete':False,'clean_training_admitted':False,'model_run':False,'utc':utc()}
    atomic(ROOT/'COMPLETE.json',result);atomic(ROOT/'STATUS.json',result)
    signal.setitimer(signal.ITIMER_REAL,0)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        import traceback
        if OWNED and ROOT.is_dir():atomic(ROOT/'FAILURE.json',{'complete':False,'DATA_READY':False,'error_type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'utc':utc()})
        raise
