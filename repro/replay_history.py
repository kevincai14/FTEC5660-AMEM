"""Offline counterfactual replay; no imports of the agent or API clients.

Reads trusted local pickle caches using inert note classes. Uses stored prompts
to recover actual direct retrievals, and validates every emitted current block.
Historical loop bodies are executed from git using inert data only.
"""
import ast
import argparse
import math
import hashlib
import json
import pickle
import re
import statistics
import subprocess
from pathlib import Path
import sys
import os
import shutil

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'diagnostics'
# Main evaluation: first 199 questions are conv-26; remaining 105 are conv-30.
# This diagnostic intentionally targets that recorded 304-question run.

class Note:
    pass

class CacheReader(pickle.Unpickler):
    def find_class(self, module, name):
        if name in ('RobustMemoryNote', 'MemoryNote'):
            return Note
        return super().find_class(module, name)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def block(n):
    return ('talk start time:' + n.timestamp + 'memory content: ' + n.content
            + 'memory context: ' + n.context + 'memory keywords: ' + str(n.keywords)
            + 'memory tags: ' + str(n.tags) + '\n')

def indices_valid(note, total):
    return [x for x in note.links if isinstance(x, int) and not isinstance(x, bool) and 0 <= x < total][:11]

def historical_function(commit):
    source = subprocess.check_output(['git', '--git-dir='+str(ROOT/'.git'),
        'show', commit+':memory_layer.py'], encoding='utf-8')
    tree = ast.parse(source)
    fn = next(n for c in tree.body if isinstance(c, ast.ClassDef)
              for n in c.body if isinstance(n, ast.FunctionDef) and n.name=='find_related_memories_raw')
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None
    ns = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), commit, 'exec'), ns)
    return ns[fn.name]

class Retriever:
    def __init__(self, indices): self.indices = indices
    def search(self, query, k): return self.indices

class Agent:
    pass

def verify_summary(path):
    """Check committed aggregates without original caches, tokenizer or model access."""
    data=json.loads(path.read_text(encoding='utf-8'))
    rows=data['per_question']
    if len(rows)!=304 or [r['question_index'] for r in rows]!=list(range(304)):
        raise ValueError('expected 304 ordered question records')
    if [r['sample_id'] for r in rows] != [0]*199+[1]*105:
        raise ValueError('unexpected conversation membership')
    for scope, selected in [('conv26',rows[:199]),('all304',rows)]:
        for commit, metrics in data['summary'][scope].items():
            for metric,value in metrics.items():
                actual=statistics.mean(r['versions'][commit][metric] for r in selected)
                if not math.isclose(actual,value,rel_tol=1e-12,abs_tol=1e-9):
                    raise ValueError(f'aggregate mismatch: {scope}/{commit}/{metric}')
    print('PASS: 304 records, conversation split and all stored means verified.')
    print('This checks arithmetic, not the unavailable original inputs or historical model outputs.')


def main():
    global ROOT, OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',type=Path,default=ROOT,help='study checkout with complete upstream Git history')
    ap.add_argument('--log',type=Path,help='original main-run log (304 questions)')
    ap.add_argument('--cache-dir',type=Path,help='original directory containing sample 0/1 memory pickles')
    ap.add_argument('--out',type=Path,help='output JSON; defaults to diagnostics/history-replay.json')
    ap.add_argument('--tokenizer-file',type=Path,help='optional local cl100k_base.tiktoken dictionary')
    ap.add_argument('--tokenizer-cache',type=Path,help='cache directory; downloads the public dictionary if absent')
    ap.add_argument('--verify-json',type=Path,help='verify aggregates only; no tokenizer, pickle or network required')
    args=ap.parse_args()
    if args.verify_json:
        verify_summary(args.verify_json)
        return
    ROOT=args.repo.resolve()
    OUT=ROOT/'diagnostics'
    sys.path.insert(0,str(ROOT/'repro'))
    from blocks import from_log
    log=args.log or ROOT/'logs/eval_robust_deepseek-flash_openai_ratio0.2_conversation_2026-11-10-15-06.log'
    cache_dir=args.cache_dir or ROOT/'cached_memories_robust_openai_deepseek-flash'
    if not log.is_file() or any(not (cache_dir/f'memory_cache_sample_{i}.pkl').is_file() for i in range(2)):
        ap.error('Full replay requires the original ignored log and memory caches; see diagnostics/README.md. Use --verify-json for the committed arithmetic check.')
    prompts = from_log(log)
    assert len(prompts)==304, len(prompts)
    functions = {c:historical_function(c) for c in ['85787ea','3542c98','5b345e6']}
    import tiktoken
    dictionary=args.tokenizer_file
    cache=args.tokenizer_cache or ROOT/'.cache/history-replay-tokenizer'
    cache.mkdir(parents=True,exist_ok=True)
    url='https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken'
    if dictionary:
        shutil.copyfile(dictionary,cache/hashlib.sha1(url.encode()).hexdigest())
    # tiktoken verifies the dictionary checksum and downloads on cache miss.
    # This is a public static asset download, not an LLM call.
    os.environ['TIKTOKEN_CACHE_DIR']=str(cache)
    # A common measuring ruler, NOT the DeepSeek API tokenizer.
    encoder = tiktoken.get_encoding('cl100k_base')
    caches=[]
    paths=[]
    for i in range(2):
        path=cache_dir/f'memory_cache_sample_{i}.pkl'
        with path.open('rb') as f: caches.append(CacheReader(f).load())
        paths.append(path)
    rows=[]
    for q, prompt in enumerate(prompts):
        memory=caches[0 if q<199 else 1]
        notes=list(memory.values())
        texts=[block(n) for n in notes]
        pos=prompt.index('talk start time:')
        context_start=pos
        direct=[]
        for _ in range(10):
            matches=[i for i,t in enumerate(texts) if prompt.startswith(t,pos)]
            assert len(matches)==1, (q,pos,len(matches))
            i=matches[0]; direct.append(i)
            pos+=len(texts[i])
            for neighbor in indices_valid(notes[i],len(notes)):
                assert prompt.startswith(texts[neighbor],pos), ('cache/log mismatch',q,i,neighbor)
                pos+=len(texts[neighbor])
        original_context=prompt[context_start:pos]
        assert 'talk start time:' not in prompt[pos:], ('unconsumed context',q)
        agent=Agent(); agent.memories=memory; agent.retriever=Retriever(direct)
        outputs={c:fn(agent,'frozen keywords',10) for c,fn in functions.items()}
        assert outputs['5b345e6']==original_context, ('December/current mismatch',q)
        row={'question_index':q,'sample_id':0 if q<199 else 1,'direct_indices':direct,'versions':{}}
        for c,context in outputs.items():
            full=prompt[:context_start]+context+prompt[pos:]
            row['versions'][c]={'blocks':context.count('talk start time:'),
                'context_chars':len(context),'context_tokens_cl100k':len(encoder.encode(context)),
                'answer_prompt_tokens_cl100k':len(encoder.encode(full))}
        rows.append(row)
    summary={}
    for scope, selected in [('conv26',rows[:199]),('all304',rows)]:
        summary[scope]={c:{metric:statistics.mean(r['versions'][c][metric] for r in selected)
                            for metric in rows[0]['versions'][c]} for c in functions}
    payload={'method':'frozen current notes + actual logged direct retrievals; historical functions from git',
       'tokenizer':'cl100k_base; diagnostic ruler, not provider billing tokenizer',
       'checks':'304 prompts exactly reconstructed; December function equals recorded context',
       'source_sha256':{p.name:digest(p) for p in [log]+paths},
       'summary':summary,'per_question':rows}
    target=args.out or OUT/'history-replay.json'
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
