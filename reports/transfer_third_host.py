"""Relay an authorized project bundle; launch third-host verification only after both SSH processes succeed."""
import json
import subprocess

ssh=['ssh','-i','/Users/xinyiliu/.ssh/lxy','-o','IdentitiesOnly=yes']
source='ubuntu@159.54.172.24'
target='ubuntu@146.235.234.66'
root='/home/ubuntu/lxy-west2'
command="cd /home/ubuntu && tar --transform='s|^lxy-west/|lxy-west2/|' -czf - lxy-ml-opf-env .local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu lxy-west/data lxy-west/snapshots/seed42-2c1d52c lxy-west/environment/data_manifest.json lxy-west/logs/expected-environment.json"
print('THIRD_HOST_TRANSFER_STARTED',flush=True)
sender=subprocess.Popen(ssh+[source,command],stdout=subprocess.PIPE)
receiver=subprocess.Popen(ssh+[target,'tar -xzf - -C /home/ubuntu'],stdin=sender.stdout)
sender.stdout.close()
receive_status=receiver.wait();send_status=sender.wait()
assert receive_status==0 and send_status==0,(receive_status,send_status)
print('THIRD_HOST_TRANSFER_COMPLETE',flush=True)
launch='''from pathlib import Path
import json,subprocess
root=Path('/home/ubuntu/lxy-west2')
with (root/'logs/transfer-complete.json').open('x') as f:json.dump({'sender_exit':0,'receiver_exit':0},f)
with (root/'logs/third-host-bootstrap.log').open('x') as log:
 p=subprocess.Popen(['/home/ubuntu/lxy-ml-opf-env/bin/python','-u',str(root/'logs/third_host_bootstrap.py'),'--root',str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
print('BOOTSTRAP_PID',p.pid)
'''
import shlex
subprocess.run(ssh+[target,'python3 -c '+shlex.quote(launch)],check=True)
