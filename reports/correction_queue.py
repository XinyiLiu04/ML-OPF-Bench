"""Run explicit versioned tasks with bounded slots; preserve every attempt."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def alive(pid):
    path=Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split()[2] != 'Z'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--reserve-pid',type=int,action='append',default=[])
    a=p.parse_args()
    plan=json.loads(a.plan.read_text())
    tasks=list(plan['tasks']);active={}
    env=os.environ.copy()
    with (a.root/'logs/correction-queue-v1.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for item in plan['sources']:
            snapshot=a.root/item['snapshot']
            current_env=env|{'PYTHONPATH':str(snapshot/'src')+':'+str(snapshot)}
            actual=subprocess.check_output([sys.executable,'-c','from ml_opf_bench.io import code_version; print(code_version()["source_sha256"])'],cwd=snapshot,env=current_env,text=True).strip()
            assert actual==item['source_sha256'], (snapshot,actual)
        while tasks or active:
            for pid,(child,task) in list(active.items()):
                if child.poll() is None:continue
                folder=a.root/task['output']/task['run_id']
                done=list(folder.glob('*/completed.json'))
                failed=list(folder.glob('*/failed.json'))
                known=(task['run_id']=='ac-case300-as-cross-system-seed42' and len(failed)==1
                       and 'No validation sample shares an active set' in failed[0].read_text())
                assert (child.returncode==0 and len(done)==1 and not failed) or known,task
                print(json.dumps({'action':'finish','run_id':task['run_id'],'returncode':child.returncode}),flush=True)
                del active[pid]
            while tasks and len(active)+sum(alive(pid) for pid in a.reserve_pid)<3:
                task=tasks.pop(0)
                output=a.root/task['output']
                assert not list((output/task['run_id']).glob('*/manifest.json')),task['run_id']
                source=plan['sources'][task['source']]
                snapshot=a.root/source['snapshot']
                current_env=env|{'PYTHONPATH':str(snapshot/'src')+':'+str(snapshot)}
                cmd=[sys.executable,'-m','ml_opf_bench.cli','run',*task['args'],'--seed','42','--device','cuda','--workers','8',
                     '--data-root',str(a.root/'data'),'--output-root',str(output)]
                child=subprocess.Popen(cmd,cwd=snapshot,env=current_env)
                active[child.pid]=(child,task)
                print(json.dumps({'action':'start','pid':child.pid,'task':task,'host':os.uname().nodename,'max_concurrency':3}),flush=True)
            time.sleep(10)
        print('CORRECTION_QUEUE_FINISHED',flush=True)


if __name__=='__main__':main()
