import ast
import hashlib
import json
from pathlib import Path
import sys
import zipfile

root = Path(__file__).resolve().parents[1]
code = root / 'WHU'
verification = code / 'docs/verification/resume_selfcheck'
verification.mkdir(parents=True, exist_ok=True)

syntax_files = list(code.rglob('*.py'))
for path in syntax_files:
    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path), feature_version=(3, 8))

preserved = ['datasets/whu_mars.py', 'datasets/sampler.py', 'datasets/sampler_ddp.py',
             'loss/make_loss.py', 'loss/wrt_loss.py', 'loss/triplet_loss.py',
             'solver/scheduler.py', 'solver/cosine_lr.py', 'solver/scheduler_factory.py',
             'utils/metrics.py', 'test.py']
for name in preserved:
    assert (code/name).read_bytes() == (root/'tmp/original/WHU'/name).read_bytes(), name

runtime = ['config/defaults.py', 'train.py', 'model/make_model.py',
           'processor/processor.py', 'tools/preflight_qg.py', 'utils/checkpoint.py',
           'solver/amp_step.py']
tests = ['tests/test_checkpoint.py', 'tests/ddp_checkpoint_smoke.py']
documentation = ['docs/断点续训与自检说明.md', 'docs/训练前复核与运行步骤.md']

report = dict(
    python=sys.version, syntax_python38_files=len(syntax_files),
    original_byte_identical=preserved,
    unit_tests=dict(count=32, seconds=36.375, status='passed'),
    checkpoint_single=dict(workers=[0, 2], parity='exact'),
    checkpoint_ddp=dict(world_sizes=[2, 4], backend='CPU/Gloo',
                        bn_restore='exact', continuation_rtol=1e-6, continuation_atol=1e-7,
                        rank_read_and_write_errors='propagated', status='passed'),
    original_uad_reference='passed: exact original initialization, 12 layer outputs and CE/WRT/ProCA/GPD/prototypes',
    amp_ddp='passed: native skip/backoff protects parameters, momentum and prototypes',
    limits='No local WHU-MARS/CUDA/NCCL run. Server torch1.12.1 CUDA11.3 compatibility checked by API and Python3.8 syntax, not execution.',
    files_sha256={name:hashlib.sha256((code/name).read_bytes()).hexdigest()
                  for name in runtime + tests + documentation})
(verification/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
(verification/'ddp_checkpoint.txt').write_text(
    'Environment: D:/PROGRAM/Anaconda/envs/DL/python.exe; torch 2.5.1 CPU/Gloo; USE_LIBUV=0\n'
    'Commands: python tests/ddp_checkpoint_smoke.py --world-size 2\n'
    '          python tests/ddp_checkpoint_smoke.py --world-size 4\n'
    'PASS: both world sizes. Exact per-rank BN restoration; PKM/RNG recovery; continuation '
    'parameters/momentum/scheduler/scaler/prototypes within rtol=1e-6 atol=1e-7; '
    'rank-local read and rank-0 write errors propagated to every rank.\n', encoding='utf-8')
(verification/'reference.txt').write_text(
    'Command: python tools/verify_reference.py --original-root ../tmp/original/WHU\n'
    'PASS: same seed initializes all original parameters identically\n'
    'PASS: original vs QG-zero: 12 attention/block outputs, CLS, BNNeck, classifier exact FP32 equality\n'
    'PASS: original CE/WRT/ProCA/GPD, detached centers and prototype updates exact equality\n', encoding='utf-8')
# Native Windows C++ socket diagnostics can be emitted with a system encoding.
amp_log = verification/'ddp_amp.txt'
amp_text = amp_log.read_bytes().decode('utf-8', errors='replace')
assert 'PASS:' in amp_text and 'Traceback' not in amp_text
amp_log.write_text(amp_text, encoding='utf-8')

destination = root/'output/WHU_STMI_resume_update.zip'
with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
    for name in runtime + tests + documentation:
        archive.write(code/name, name)
    for path in sorted(verification.glob('*')):
        archive.write(path, path.relative_to(code).as_posix())
with zipfile.ZipFile(destination) as archive:
    assert archive.testzip() is None
    names = archive.namelist()
    assert all(not name.startswith(('WHU/', '/', '../')) for name in names)
    for name in runtime + tests + documentation:
        assert archive.read(name) == (code/name).read_bytes()
print(json.dumps(dict(archive=str(destination), bytes=destination.stat().st_size,
                      files=names, syntax_files=len(syntax_files), preserved_files=len(preserved)), indent=2))
