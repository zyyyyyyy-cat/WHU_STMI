from pathlib import Path
import os
import subprocess
import sys
import time

root = Path(__file__).resolve().parents[1]
out = root / 'WHU/docs/verification'
out.mkdir(parents=True, exist_ok=True)
kind = sys.argv[1]
commands = {
    'unit': [sys.executable, '-m', 'unittest', 'discover', '-s', 'WHU/tests', '-p', 'test_qg_sfm.py', '-v'],
    'reference': [sys.executable, 'WHU/tools/verify_reference.py', '--original-root', 'tmp/original/WHU'],
    'ddp': [r'D:\PROGRAM\Anaconda\envs\DL\python.exe', 'WHU/tests/ddp_qg_smoke.py'],
    'syntax': [sys.executable, '-m', 'compileall', '-q', 'WHU'],
}
environment = os.environ.copy()
environment['PYTHONIOENCODING'] = 'utf-8'
environment['USE_LIBUV'] = '0'
start = time.time()
result = subprocess.run(commands[kind], cwd=root, env=environment, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=180)
output = result.stdout.decode('utf-8', errors='replace')
summary = 'COMMAND: {}\nEXIT: {}\nELAPSED_SECONDS: {:.2f}\n\n{}'.format(
    subprocess.list2cmdline(commands[kind]), result.returncode, time.time() - start, output)
(out / (kind + '.txt')).write_text(summary, encoding='utf-8')
print(summary)
sys.exit(result.returncode)
