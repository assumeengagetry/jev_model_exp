#!/usr/bin/env python3
"""Read-only observation; notify this session on real training and completion."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--run-root', default='/tmp/opencode/jev-all-dimension-supervised-20261108')
parser.add_argument('--local-status-root')
args = parser.parse_args()
ROOT = args.run_root
SESSION = 'ses_e749dc3c2ffev4nW5qgrukOIiC'
LOCAL = Path(args.local_status_root).resolve() if args.local_status_root else Path(__file__).resolve().parents[1]
LOCK = (LOCAL / 'watch.lock').open('a+')
fcntl.flock(LOCK.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def notify(text):
    body = json.dumps({'text': text, 'description': '五维正式训练进展', 'delivery': 'steer', 'resume': True})
    subprocess.run(['opencode', 'api', 'post', f'/api/session/{SESSION}/synthetic', '--data', body],
                   check=True, text=True, capture_output=True, timeout=30)


announced = set()
deadline = time.monotonic() + 48 * 3600
while time.monotonic() < deadline:
    code = ('import json; from pathlib import Path; p=Path(' + repr(ROOT) + '); '
            'print(json.dumps({n:json.loads((p/n).read_text()) for n in '
            "['PIPELINE.json','progress.json','COMPLETE.json','DATA_READY.json'] if (p/n).is_file()}))")
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                             '-o', 'StrictHostKeyChecking=yes', '-o', 'ClearAllForwardings=yes',
                             'sh-10-gpu', 'python3 -c ' + __import__('shlex').quote(code)],
                            text=True, capture_output=True, timeout=40)
    if result.returncode == 0:
        data = json.loads(result.stdout)
        pipeline = data.get('PIPELINE.json', {})
        progress = data.get('progress.json', {})
        key = progress.get('stage')
        if progress.get('step', 0) > 0 and progress.get('actual_training_kernel_executed') and key not in announced:
            notify('五维原始标签正式训练已实际开始：' + json.dumps(progress, ensure_ascii=False)
                   + '。这是正式训练的真实优化进度，不是测试／烟测；未执行hash。远端目录：' + ROOT)
            announced.add(key)
        if pipeline.get('stage') == 'blocked':
            notify('五维正式训练阻塞：' + json.dumps(pipeline, ensure_ascii=False)
                   + '。请读取本任务pipeline.log/prepare.log/train.log处理，不修改其他实验。')
            print(json.dumps(data, ensure_ascii=False), flush=True)
            break
        if data.get('COMPLETE.json', {}).get('complete'):
            notify('五维原始标签正式训练已完成，模型已保存：'
                   + json.dumps(data['COMPLETE.json'], ensure_ascii=False)
                   + '。本任务没有自动运行benchmark、测试或hash，也没有上传；下一步按用户要求准备Private参榜。')
            print(json.dumps(data, ensure_ascii=False), flush=True)
            break
    time.sleep(180)
else:
    notify('五维训练只读监视到达48小时上限；请直接查看 ' + ROOT + '/PIPELINE.json。未自动停止训练。')
