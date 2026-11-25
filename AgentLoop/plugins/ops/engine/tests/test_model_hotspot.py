"""用临时训练仓和合成测量验证单批流程；不冒充 NPU 性能测试。"""
import difflib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import model_hotspot as flow
from model_metrics import compare, validate_metrics
from training_benchmark import parse_log


WORKER = r'''
import csv, json, os, pathlib, sys
p = pathlib.Path(os.environ['AGENTLOOP_MODEL_ROOT'])
phase = os.environ['AGENTLOOP_PHASE']
with (p/'calls.txt').open('a') as f: f.write(phase+'\n')
if phase == 'profile':
    out = pathlib.Path(os.environ['AGENTLOOP_PROFILE_DIR'])
    with (out/'kernel_details.csv').open('w') as f:
        w=csv.writer(f)
        w.writerow(['Step Id','Name','Duration(us)','Input Shapes','Input Data Types'])
        for step in range(1,4):
            for name,dur in [('hot_kernel',60),('other_kernel',25),('unknown_op',10),('aclnnAdd',5)]:
                w.writerow([step,name,dur,'4,8','FLOAT'])
else:
    if phase == 'regression' and (p/'crash').exists(): sys.exit(7)
    slow = phase == 'regression' and (p/'slow').exists()
    bad_loss = phase == 'regression' and (p/'bad_loss').exists()
    optimized = 'optimized' in (p/'kernel.py').read_text()
    samples=[{'step':n,'step_time_ms':120 if slow else (80 if optimized else 100),
              'loss':9 if bad_loss else 1/n,'tokens':1000} for n in range(1,4)]
    result={'run_id':os.environ['AGENTLOOP_RUN_ID'],'phase':phase,
            'workload_id':os.environ['AGENTLOOP_WORKLOAD_ID'],'profiling_enabled':False,
            'samples':samples}
    pathlib.Path(os.environ['AGENTLOOP_METRICS_PATH']).write_text(json.dumps(result))
'''


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.root / 'model'
        self.model.mkdir()
        self.source = self.model / 'kernel.py'
        self.source.write_text('value = "original"\n')
        self.original = self.source.read_bytes()
        self.worker = self.root / 'worker.py'
        self.worker.write_text(WORKER)
        self.index = self.root / 'Index.md'
        self.index.write_text('MANDATORY_PER_ROUND: 1\n')
        self.config = self.root / 'config.json'
        self.cfg = {'model_root':str(self.model), 'profile_command':[sys.executable,str(self.worker)],
                    'benchmark_command':[sys.executable,str(self.worker)],
                    'source_map':{'hot_kernel':{'source':'kernel.py'}}, 'min_steps':3,
                    'latency_index':str(self.index),
                    'workload':dict(model='test', checkpoint='initial',data='fixed',seed=1,
                                    precision='fp32',parallelism='1',batch_size=1,sequence_length=1000)}
        flow.write(self.config,self.cfg)

    def prepare(self):
        return flow.prepare(self.config,self.root/'runs')

    def deliver(self, run):
        state=flow.read(run/'state.json')
        task=next(t for t in state['tasks'] if t['name']=='hot_kernel')
        out=Path(task['directory'])
        old=self.original.decode()
        new=old.replace('original','optimized')
        (out/'original.py').write_text(old)
        (out/'best.py').write_text(new)
        (out/'kernel.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                                                                  fromfile='a/kernel.py',tofile='b/kernel.py')))
        for name,lat in [('baseline',2),('optimized',1)]:
            flow.write(out/f'{name}.json',{'implementation':{'avg_latency_ms':lat}})
        summary={'run_id':run.name,'op_name':'hot_kernel','status':'REACHED',
                 'perf_baseline':str(self.source),'gate_ref':'reference.py','best_code':'best.py',
                 'optimizer_rounds':1,'speedup_vs_user':2,
                 'optimizer':{'tool':'triton-latency-optimizer','points_hit':[1], 'loaded_docs':['IR.md'],
                              'points_considered':1,'scan_complete':False,'kb_preflight':{'available':False},
                              'scan_trace':[{'round':1,'scan_from':1,'hit':1,'not_hit':'[1,0]','mandatory':[1]}]},
                 'bench':{'tool':'scripts/benchmark.py','baseline_result_json':'baseline.json',
                          'optimized_result_json':'optimized.json','baseline_avg_ms':2,'optimized_avg_ms':1},
                 'patch':{'mode':'kernel_replace','path':'kernel.patch','diff_from':'original.py','diff_target':'best.py',
                          'apply_target':str(self.source),'apply_rel':'kernel.py','resverified':{'passed':1,'total':1}}}
        flow.write(out/'summary.json',summary)
        (out/'report.md').write_text('算子级归因；替换回图由外层执行。\n调优过程 triton-latency-optimizer scan_complete')
        return out,summary

    def test_one_profile_fixed_batch_one_regression(self):
        run=self.prepare()
        self.deliver(run)
        state=flow.finish(run)
        self.assertEqual(state['status'],'COMPLETED',state)
        self.assertAlmostEqual(state['comparison']['step_speedup'],1.25)
        self.assertIn('optimized',self.source.read_text())
        self.assertEqual((self.model/'calls.txt').read_text().splitlines(),['profile','baseline','regression'])
        statuses={t['name']:t['status'] for t in state['tasks']}
        self.assertEqual(statuses['other_kernel'],'SKIPPED_NO_SOURCE')
        self.assertEqual(statuses['unknown_op'],'NEEDS_REVIEW')
        self.assertEqual(statuses['hot_kernel'],'APPLIED')
        with self.assertRaises(ValueError): flow.finish(run)

    def test_accuracy_failure_rolls_back_and_reports_measured_gain(self):
        run=self.prepare(); self.deliver(run)
        (self.model/'bad_loss').touch()
        state=flow.finish(run)
        self.assertEqual(state['status'],'ROLLED_BACK')
        self.assertFalse(state['comparison']['accuracy_passed'])
        self.assertEqual(self.source.read_bytes(),self.original)

    def test_slowdown_rolls_back_without_more_optimization(self):
        run=self.prepare(); self.deliver(run)
        (self.model/'slow').touch()
        state=flow.finish(run)
        self.assertEqual(state['status'],'ROLLED_BACK')
        self.assertEqual(self.source.read_bytes(),self.original)
        self.assertEqual((self.model/'calls.txt').read_text().splitlines(),['profile','baseline','regression'])

    def test_failed_training_restores_source(self):
        run=self.prepare(); self.deliver(run)
        (self.model/'crash').touch()
        with self.assertRaises(RuntimeError): flow.finish(run)
        self.assertEqual(self.source.read_bytes(),self.original)
        self.assertEqual(flow.read(run/'state.json')['status'],'FAILED')

    def test_stale_summary_is_not_applied(self):
        run=self.prepare(); out,s=self.deliver(run)
        s['run_id']='old_run'; flow.write(out/'summary.json',s)
        state=flow.finish(run)
        self.assertEqual(state['status'],'NO_CHANGES')
        self.assertEqual(self.source.read_bytes(),self.original)
        self.assertEqual((self.model/'calls.txt').read_text().splitlines(),['profile','baseline'])

    def test_modified_source_is_preserved_and_rejected(self):
        run=self.prepare(); self.deliver(run)
        self.source.write_text('user_edit = True\n')
        with self.assertRaises(ValueError): flow.finish(run)
        self.assertEqual(self.source.read_text(),'user_edit = True\n')

    def test_modified_candidates_cannot_trigger_new_tuning(self):
        run=self.prepare(); self.deliver(run)
        flow.write(run/'analysis/candidates.json',{'candidates':[]})
        with self.assertRaises(ValueError): flow.finish(run)
        self.assertEqual(self.source.read_bytes(),self.original)

    def test_bad_patch_rejected_without_touching_other_files(self):
        run=self.prepare(); out,s=self.deliver(run)
        (out/'kernel.patch').write_text('--- a/another.py\n+++ b/another.py\n@@ -1 +1 @@\n-x\n+y\n')
        state=flow.finish(run)
        self.assertEqual(state['status'],'NO_CHANGES')
        self.assertEqual(self.source.read_bytes(),self.original)

    def test_patch_target_must_match_reverified_bytes(self):
        run=self.prepare(); out,s=self.deliver(run)
        (out/'best.py').write_text('different=1\n')
        state=flow.finish(run)
        self.assertEqual(state['status'],'NO_CHANGES')
        self.assertEqual(self.source.read_bytes(),self.original)

    def test_malformed_operator_result_is_rejected_not_batch_failure(self):
        run=self.prepare();out,s=self.deliver(run)
        flow.write(out/'summary.json',['invalid'])
        state=flow.finish(run)
        self.assertEqual(state['status'],'NO_CHANGES')
        self.assertEqual(next(t for t in state['tasks'] if t['name']=='hot_kernel')['status'],'REJECTED')

    def test_tuner_failure_does_not_block_other_candidates(self):
        self.cfg['tune_command']=[sys.executable,'-c','raise SystemExit(3)']
        flow.write(self.config,self.cfg)
        run=self.prepare(); flow.tune(run)
        state=flow.finish(run)
        self.assertEqual(state['status'],'NO_CHANGES')
        self.assertEqual(next(t for t in state['tasks'] if t['name']=='hot_kernel')['status'],'FAILED')

    def test_no_hotspots_no_post_training(self):
        self.cfg['analyzer']={'bottleneck_threshold':0.99}
        flow.write(self.config,self.cfg)
        run=self.prepare()
        self.assertEqual(flow.finish(run)['status'],'NO_CHANGES')
        self.assertEqual((self.model/'calls.txt').read_text().splitlines(),['profile','baseline'])

    def test_operator_gain_without_model_gain_rolls_back(self):
        self.worker.write_text(self.worker.read_text().replace("80 if optimized else 100", "100"))
        run=self.prepare(); self.deliver(run)
        state=flow.finish(run)
        self.assertEqual(state['status'],'ROLLED_BACK')
        self.assertEqual(state['comparison']['acceptance_reason'],'NO_E2E_GAIN')
        self.assertEqual(self.source.read_bytes(),self.original)

    def test_second_candidate_deferred_even_if_artifacts_exist(self):
        self.cfg['source_map']['other_kernel']={'source':'kernel.py'}
        flow.write(self.config,self.cfg)
        run=self.prepare();out,summary=self.deliver(run)
        second=next(t for t in flow.read(run/'state.json')['tasks'] if t['name']=='other_kernel')
        second_dir=Path(second['directory'])
        for name in ['original.py','best.py','kernel.patch','baseline.json','optimized.json','report.md']:
            (second_dir/name).write_bytes((out/name).read_bytes())
        summary['op_name']='other_kernel';flow.write(second_dir/'summary.json',summary)
        state=flow.finish(run)
        self.assertEqual(state['status'],'COMPLETED')
        self.assertEqual([t['status'] for t in state['tasks'][:2]],['APPLIED','DEFERRED'])
        self.assertIn('optimized',self.source.read_text())

    def test_explicit_target_selects_only_that_candidate(self):
        self.cfg['source_map']['other_kernel']={'source':'kernel.py'}
        self.cfg['target_operator']='other_kernel'
        flow.write(self.config,self.cfg)
        run=self.prepare()
        state=flow.read(run/'state.json')
        self.assertEqual(state['selected_operator']['name'],'other_kernel')
        self.assertEqual([t['status'] for t in state['tasks'][:2]],['DEFERRED','PENDING'])

    def test_full_cli_dispatches_tuner_and_measures_once(self):
        tuner=self.root/'tuner.py'
        tuner.write_text(
            'import json,os,pathlib,sys\nfrom types import SimpleNamespace\n'
            f'sys.path.insert(0,{str(Path(__file__).resolve().parent)!r})\n'
            'from test_model_hotspot import PipelineTests\n'
            'r=json.loads(pathlib.Path(sys.argv[1]).read_text())\n'
            'obj=SimpleNamespace(original=pathlib.Path(r["frozen_source"]).read_bytes(),source=pathlib.Path(r["source"]))\n'
            'PipelineTests.deliver(obj,pathlib.Path(os.environ["AGENTLOOP_RUN_DIR"]))\n')
        self.cfg['tune_command']=[sys.executable,str(tuner),'{request}']
        flow.write(self.config,self.cfg)
        proc=subprocess.run([sys.executable,'-B',str(flow.ROOT/'plugins/ops/engine/model_hotspot.py'),
                             'run','--config',str(self.config),'--out',str(self.root/'runs')],
                            capture_output=True,text=True)
        self.assertEqual(proc.returncode,0,proc.stdout+proc.stderr)
        run=Path(proc.stdout.splitlines()[0])
        self.assertEqual(flow.read(run/'state.json')['status'],'COMPLETED')
        self.assertEqual((self.model/'calls.txt').read_text().splitlines(),['profile','baseline','regression'])

    def test_regression_timeout_restores_source(self):
        run=self.prepare();self.deliver(run)
        with self.worker.open('a') as stream:
            stream.write("\nif phase == 'regression':\n import time\n time.sleep(5)\n")
        state=flow.read(run/'state.json');state['config']['timeout_seconds']=0.1
        flow.write(run/'state.json',state)
        with self.assertRaises(subprocess.TimeoutExpired):flow.finish(run)
        self.assertEqual(self.source.read_bytes(),self.original)
        self.assertEqual(flow.read(run/'state.json')['status'],'FAILED')

    def test_benchmark_adapter_runs_command_and_exports_real_log_samples(self):
        import os
        logline='iteration {}/10 | elapsed time per iteration (ms): 100 | lm loss: 1.1'
        command=[sys.executable,'-c',f'print({logline.format(1)!r});print({logline.format(2)!r})']
        output=self.root/'measure.json'
        env={**os.environ,'AGENTLOOP_METRICS_PATH':str(output),'AGENTLOOP_RUN_ID':'run',
             'AGENTLOOP_PHASE':'baseline','AGENTLOOP_WORKLOAD_ID':'fixed'}
        subprocess.run([sys.executable,'-B',str(flow.ROOT/'plugins/ops/engine/training_benchmark.py'),
                        '--warmup-steps','1','--',*command],env=env,check=True)
        samples=flow.read(output)['samples']
        self.assertEqual(samples,[dict(step=2,step_time_ms=100.0,loss=1.1)])


class MetricsTests(unittest.TestCase):
    def payload(self):
        return dict(run_id='run',phase='baseline',workload_id='same',profiling_enabled=False,
                    samples=[dict(step=1,step_time_ms=10,loss=0.0)])

    def test_stale_profiled_or_nonfinite_metrics_rejected(self):
        for field,value in [('run_id','old'),('workload_id','changed'),('profiling_enabled',True)]:
            data=self.payload(); data[field]=value
            with self.assertRaises(ValueError):validate_metrics(data,'run','baseline','same',1)
        data=self.payload();data['samples'][0]['loss']=float('nan')
        with self.assertRaises(ValueError):validate_metrics(data,'run','baseline','same',1)

    def test_loss_zero_uses_absolute_tolerance_and_steps_must_align(self):
        a=self.payload();b=self.payload();b['samples'][0]['loss']=1e-6
        self.assertTrue(compare(a,b,1e-5,1e-3,0)['accuracy_passed'])
        b['samples'][0]['step']=2
        with self.assertRaises(ValueError):compare(a,b,1e-5,1e-3,0)

    def test_zero_or_small_e2e_gain_is_not_accepted(self):
        a=self.payload(); b=self.payload()
        for latency in (10, 9.95, 10.1):
            b['samples'][0]['step_time_ms']=latency
            r=compare(a,b,1e-5,1e-3,5)
            self.assertFalse(r['accepted'])
            self.assertFalse(r['e2e_gain_passed'])
        b['samples'][0]['step_time_ms']=9.8
        self.assertTrue(compare(a,b,1e-5,1e-3,0)['accepted'])
        self.assertFalse(compare(a,b,1e-5,1e-3,0,3)['accepted'])
        with self.assertRaises(ValueError):compare(a,b,1e-5,1e-3,0,0)

    def test_mean_gain_without_median_gain_is_not_accepted(self):
        a=self.payload(); b=self.payload()
        a['samples']=[dict(step=i,step_time_ms=x,loss=0.0) for i,x in enumerate([10,10,30])]
        b['samples']=[dict(step=i,step_time_ms=x,loss=0.0) for i,x in enumerate([10,10,15])]
        self.assertFalse(compare(a,b,1e-5,1e-3,0)['accepted'])

    def test_training_log_and_duplicate_ranks(self):
        text='iteration 1/10 | elapsed time per iteration (ms): 120 | lm loss: 1.2E+00\n'
        text+='iteration 2/10 | elapsed time per iteration (ms): 100 | lm loss: 1.1E+00\n'
        self.assertEqual(parse_log(text,1,4096),[dict(step=2,step_time_ms=100,loss=1.1,tokens=4096)])
        with self.assertRaises(ValueError):parse_log(text+text,0)
        with self.assertRaises(ValueError):parse_log(text.replace('1.1E+00','nan'),0)


if __name__ == '__main__':
    unittest.main()
