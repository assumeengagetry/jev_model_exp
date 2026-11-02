#!/usr/bin/env python3
"""Notify the user's SCP session and wake the execution session after all phases finish."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
STATE = ROOT / 'state.json'
EXECUTION = 'ses_e749dc3c2ffev4nW5qgrukOIiC'
SCP = 'ses_e78d20bb6ffeeIq6l616oJr4f8'
REMOTE = '/home/dev/assumeengage'
PLAN = 'reports/next-experiments-plan-20261030/REPORT.md'


def emit(event, **values):
    print(json.dumps({'time': datetime.now(timezone.utc).isoformat(), 'event': event, **values},
                     ensure_ascii=False), flush=True)


def read_state():
    try:
        return json.loads(STATE.read_text())
    except FileNotFoundError:
        return {'sent': {}, 'distillation_wakeup': None, 'blocked_wakeups': []}


def save(state):
    temporary = STATE.with_suffix('.writing')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(STATE)


def api_prompt(session, text, metadata):
    payload = {'text': text, 'delivery': 'queue', 'resume': True, 'metadata': metadata}
    result = subprocess.run(['/usr/bin/opencode', 'api', 'post', '/api/session/' + session + '/prompt',
                             '--data', json.dumps(payload, ensure_ascii=False)],
                            text=True, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError('OpenCode message admission failed, exit ' + str(result.returncode))
    receipt = json.loads(result.stdout)['data']
    if not receipt.get('id'):
        raise RuntimeError('Message admission returned no receipt')
    return receipt['id']


def snapshot():
    result = subprocess.run(['ssh', '-T', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
                             '-o', 'StrictHostKeyChecking=yes', '-o', 'ClearAllForwardings=yes',
                             'sh-09-gpu', 'python3 -B ' + REMOTE + '/scripts/five-area-stage-snapshot.py'],
                            text=True, capture_output=True, timeout=45)
    if result.returncode:
        raise RuntimeError('Cluster stage snapshot failed, exit ' + str(result.returncode))
    return json.loads(result.stdout)


def cycle(state):
    current = snapshot()
    fresh = [m for m in current['milestones'] if m['key'] not in state['sent']]
    if fresh:
        text = ('用户要求：每有阶段性成果，通知本session将成果scp回来。以下是新完成的阶段；'
                '请现在用SSH别名sh-09-gpu将列出的已完成文件/目录拉回本地项目，按阶段整理。'
                '本次仅scp，不执行提交/推送，也不改集群的实验或运行进程。'
                '公共基座已有的不用重复下载；共同初始化 models/Qwen3-0.6B-common-pair-init-r32 '
                '可从 /home/dev/assumeengage/models/Qwen3-0.6B-common-pair-init-r32 拉取一次，供加载与后续分析。\n'
                + json.dumps(fresh, ensure_ascii=False, indent=2))
        receipt = api_prompt(SCP, text, {'sender': 'cluster-stage-followup',
                                       'milestones': [m['key'] for m in fresh]})
        for item in fresh:
            state['sent'][item['key']] = {'message_id': receipt, 'name': item['name']}
        save(state)
        emit('scp_stage_notification_admitted', message_id=receipt, milestones=[m['key'] for m in fresh])
    if current['five_area_complete'] and not state['distillation_wakeup']:
        text = ('用户已明确授权，并最新取消 Language 专家及其训练曝光：等四专科、四科 MIXED_ALL、五种融合和10条件统一五维评测全部结束后，'
                '接着执行 ' + PLAN + ' 的Teacher蒸馏实验。集群完成标记和控制锁现已满足该条件。'
                '请直接SSH sh-09-gpu接手，不等待第二次开训确认。先读这份报告及集群AGENTS.md，'
                '默认以已训练MIXED_ALL为共同Student起点，复用完整有效四科训练池和token缓存。'
                '务必先读 config/experiment-five-specialists.json、config/queued-distillation.json 和上一轮 ACTIVE_TRAINING_SCOPE.json；'
                '只用 knowledge、retrieval、tools、arts，缓存中的 Language 保留但不得补标或训练；'
                'PairPool 必须传入 domains=cfg["active_training_domains"]，不能使用默认五科并集。'
                'Teacher为PelaAI/KnowLine-31B-Gen1，离线补齐每pair独立匹配/不匹配概率，'
                '从同一起点执行KD_HARD、KD_SOFT、KD_HYBRID三组等量完整一轮续训和同一37项评测。'
                '不要重训专科或再次做模型融合，不套9998对或50万上限；用户不要求额外测试/哈希。'
                '新阶段建议目录 ' + REMOTE + '/runs/distillation-after-five-specialists-20261031。'
                '阶段完成后写complete.json；Teacher/标注/总完成分别写TEACHER_READY.json、'
                'TEACHER_LABELS_READY.json、DISTILLATION_COMPLETE.json（complete:true，artifacts列出需要scp的结果路径），'
                '已有阶段通知程序会继续告诉 ' + SCP + ' 拉取成果。不要将本通知仅当作提醒后结束，请实际推进。')
        receipt = api_prompt(EXECUTION, text, {'sender': 'approved-next-experiment',
                                              'plan': PLAN, 'gate': 'all-ten-active-five-area-conditions-complete'})
        state['distillation_wakeup'] = receipt
        save(state)
        emit('distillation_execution_wakeup_admitted', message_id=receipt)
    pipelines = [('five', current.get('pipeline', {})), ('distillation', current.get('distillation_status', {}))]
    for label, pipeline in pipelines:
        if pipeline.get('stage') != 'blocked':
            continue
        key = label + ':' + str(pipeline.get('updated_unix', '')) + ':' + str(pipeline.get('error', ''))
        if key in state['blocked_wakeups']:
            continue
        text = ('已授权集群实验出现阻塞，请实际SSH接手修复并从现有进度继续，不缩减当前有效四科数据和训练预算；'
                'Language 专家及其混合训练曝光已由用户取消，不得恢复；先读当前配置与 ACTIVE_TRAINING_SCOPE.json。'
                '不动不属于本轮的任务。当前状态：\n' + json.dumps(pipeline, ensure_ascii=False, indent=2))
        receipt = api_prompt(EXECUTION, text, {'sender': 'owned-experiment-blocked', 'phase': label})
        state['blocked_wakeups'].append(key)
        save(state)
        emit('blocked_execution_wakeup_admitted', phase=label, message_id=receipt)
    state['last_observed_utc'] = datetime.now(timezone.utc).isoformat()
    state['last_five_area_stage'] = current.get('pipeline', {}).get('stage')
    save(state)
    return current['distillation_complete']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    with (ROOT / 'watcher.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            emit('existing_stage_watcher_active')
            return
        state = read_state()
        while True:
            try:
                finished = cycle(state)
                if args.once or finished:
                    break
            except Exception as error:
                emit('stage_watcher_retry_pending', error=str(error))
                if args.once:
                    raise
            time.sleep(120)


if __name__ == '__main__':
    main()
