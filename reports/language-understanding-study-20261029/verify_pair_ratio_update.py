"""Verify published tables, version boundaries, arithmetic and links; no model load or scoring."""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import csv,json,math,os,re

if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise SystemExit('CPU verification only')
OUT=Path(__file__).resolve().parent
PROJECT=OUT.parents[1]


def read(path):return json.loads(Path(path).read_text())
def table(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))
def close(a,b):assert math.isclose(float(a),float(b),abs_tol=1e-12,rel_tol=1e-12),(a,b)


def main():
    update=read(OUT/'pair-ratio-update.json');review=update['review'];old=table('runs.csv');new=table('pair-ratio-runs.csv');all_runs=table('runs-through-pair-ratio.csv')
    assert len(old)==91 and len(new)==4 and len(all_runs)==95 and all_runs[:91]==old and all_runs[91:]==new
    assert len({str(Path(x['checkpoint']).resolve()) for x in all_runs})==95
    assert all(Path(x['checkpoint']).is_dir() and Path(x['source']).is_file() for x in new)
    assert len(table('pilot-dev-only-runs.csv'))==8
    native=table('pair-ratio-native-metrics.csv');summary=table('pair-ratio-summary.csv');benchmark=table('pair-ratio-public-benchmarks.csv')
    assert len(native)==40 and len(summary)==4 and len(benchmark)==36
    for x in summary:
        a=x['arm'];data=update['arms'][a];exposure=data['exposure'];natural=data['natural_validation'];public=data['official_Language'];metrics=natural['pair_metrics']
        assert data['train']['actual_new_tokens']==exposure['actual_committed_nonpadding_tokens']==19999954
        assert metrics['pairs']==67202 and sum(sum(row) for row in metrics['confusion'])==67202
        cm=metrics['confusion'];f1=[]
        for label in [0,1]:
            tp=cm[label][label];fp=cm[1-label][label];fn=cm[label][1-label]
            m=metrics['classes'][str(label)]
            for key,value in [('precision',tp/(tp+fp)),('recall',tp/(tp+fn)),('f1',2*tp/(2*tp+fp+fn))]:
                close(m[key],value);close(x[f'{key}_class{label}'],value)
            assert int(x[f'support_class{label}'])==sum(cm[label]);f1.append(m['f1'])
        close(metrics['macro_f1'],sum(f1)/2);close(x['macro_f1'],sum(f1)/2);close(x['accuracy'],(cm[0][0]+cm[1][1])/67202)
        assert public['coverage']==1 and public['counts']=={'ok':31715} and len(public['per_benchmark'])==9
        close(x['language_skill'],public['language_skill'])
        for b in [b for b in benchmark if b['arm']==a]:
            m=public['per_benchmark'][b['catalog_id']];close(b['skill'],m['skill_0_100']);close(b['raw'],m['raw_coverage_adjusted'])
        close(x['native_macro'],sum(float(n['value']) for n in native if n['arm']==a)/10)
        audit=review['actual_exposure_audit'][a]
        assert audit['every_update_task_label_rows_tokens_match_source_backed_plan'] and audit['D0_counts_match_reference_multiset']
        assert exposure['native_question_draws_incomplete']==exposure['discarded_completed_microbatch_rows']==exposure['discarded_completed_microbatch_tokens']==0
    assert review['source_group_bootstrap']['source_group_strata']=={'D0-validation-Language':2837,'D2-dev':3635}
    assert not any(update['promotion_checks_recomputed'][a]['composite_pass'] for a in ['NEG2','POS2','POS4'])
    assert update['timing_scope']['evaluation_minutes']<35 and update['timing_scope']['training_max_minutes']<60
    assert update['timing_scope']['formal_execution_minutes']<120 and not update['timing_scope']['whole_direction_preparation_to_report_within120_proven']
    assert update['timing_scope']['minutes_from_prior_schedule_completion_to_initial_report_lower_bound']>120
    receipt_rows=table('pair-ratio-exposure-by-task-label.csv');by_arm=Counter()
    for x in receipt_rows:by_arm[x['arm']]+=int(x['nonpadding_tokens'])
    assert set(by_arm.values())=={19999954} and sum(x['train']['training_rows_seen'] for x in update['arms'].values())==471110
    checked=0
    for f in [PROJECT/'reports/language-understanding-study-20261029.md',OUT/'PAIR-RATIO-ADDENDUM.md',OUT/'PILOT-AND-LENGTH-ADDENDUM.md',OUT/'CHECKPOINT-RESOURCE-ADDENDUM.md',
              PROJECT/'reports/length-label-benchmark-20261029/RATIO-RUNNING-NOTE.md']:
        for link in re.findall(r'\]\(([^)]+)\)',f.read_text()):
            if link.startswith(('http:','https:','#')):continue
            assert (f.parent/link.split('#')[0]).exists(),(f,link);checked+=1
    handoff=read(PROJECT/'reports/length-label-benchmark-20261029/PAIR-RATIO-HANDOFF.json')
    assert handoff['formal_ratio_round_state']=='completed_reviewed_released' and handoff['current_GPU_claim']['GPUs']==[]
    assert not handoff['training_launch_allowed'] and handoff['formal_ratio_per_update_task_label_rows_tokens_verified']
    result={'time':datetime.now(timezone.utc).isoformat(),'passed':True,'old_snapshot_rows_unchanged_in_combined_table':91,
        'new_endpoints':4,'cumulative_distinct_endpoints':95,'dev_only_pilot_endpoints_separate':8,
        'summary_rows':4,'native_task_rows':40,'public_benchmark_rows':36,'exposure_rows':len(receipt_rows),
        'report_links_checked':checked,'pair_confusion_derived_metrics_checked':True,'promotion_gate_recomputed':True,
        'prior_CPU_preparation_timebox_limitation_explicit':True,'GPU_claim_released_in_ledger':True,
        'no_model_load_or_new_GPU_job':True,'original91_CSV_JSON_APPENDIX_not_written':True}
    path=OUT/'verification-pair-ratio-update.json';assert not path.exists();path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
