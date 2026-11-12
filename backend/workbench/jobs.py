"""Durable serialized subprocess jobs; one compute job at a time, Windows safe."""
from pathlib import Path
from threading import Thread, RLock
import os
import subprocess
import sys
import time


class JobManager:
    def __init__(self, store):
        self.store = store
        self.lock = RLock()
        self.thread = None
        self.process = None
        self.stopping = False

    def recover(self):
        for job in self.store.list('job'):
            if job['payload']['status'] == 'running':
                # A hard-stopped parent can leave its worker alive. Verify process
                # identity before terminating it, preventing PID-reuse mistakes.
                import psutil
                payload=job['payload']
                if payload.get('worker_pid'):
                    try:
                        process=psutil.Process(payload['worker_pid'])
                        command=process.cmdline()
                        if abs(process.create_time()-payload.get('worker_started',0))<.1 and any(module in command for module in ('backend.workbench.worker', 'workbench.worker')) and job['id'] in command:
                            process.terminate();process.wait(timeout=10)
                    except psutil.NoSuchProcess:
                        pass
                self.store.update(job['id'], status='interrupted', error='Backend stopped during execution. Artifacts/logs retained; create a new job to retry.')
                if job['payload']['task']=='experiment':
                    self.store.update(job['payload']['config']['experiment_id'],status='interrupted',error='Backend stopped during experiment; partial evidence retained. Create a new protocol to retry.')
        self.start()

    def enqueue(self, task, config):
        if sum(j['payload']['status'] in ('queued', 'running') for j in self.store.list('job')) >= 16:
            raise ValueError('Job queue is full')
        job = self.store.create('job', {'task': task, 'config': config, 'status': 'queued', 'stage': 'Waiting for worker',
                                        'error': None, 'result': None})
        self.start()
        return job

    def start(self):
        with self.lock:
            if self.thread and self.thread.is_alive():
                return
            self.stopping = False
            self.thread = Thread(target=self._run, daemon=True, name='research-queue')
            self.thread.start()

    def _run(self):
        while not self.stopping:
            with self.lock:
                queued = [j for j in reversed(self.store.list('job')) if j['payload']['status'] == 'queued']
                if not queued:
                    self.thread = None
                    return
                job = queued[0]
                id = job['id']
                directory = self.store.directory('jobs', id)
                self.store.update(id, status='running', stage='Starting isolated worker')
                environment = {**os.environ, 'WORKBENCH_ROOT': str(self.store.root), 'YOLO_OFFLINE': 'true',
                               'WANDB_DISABLED': 'true', 'COMET_MODE': 'DISABLED', 'PYTHONUNBUFFERED': '1'}
                with (directory / 'worker.log').open('ab') as output:
                    self.process = subprocess.Popen([sys.executable, '-m', 'backend.workbench.worker', id],
                        cwd=str(Path(__file__).resolve().parents[2]), env=environment, stdout=output, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    import psutil
                    self.store.update(id,worker_pid=self.process.pid,worker_started=psutil.Process(self.process.pid).create_time())
            self.process.wait()
            with self.lock:
                current = self.store.get(id)['payload']
                if current['status'] == 'running':
                    self.store.update(id, status='failed', error='Worker exited unexpectedly; inspect retained logs')
                    if current['task']=='experiment':
                        self.store.update(current['config']['experiment_id'],status='failed',error='Worker exited unexpectedly; partial evidence retained.')
                self.process = None

    def cancel(self, id):
        with self.lock:
            payload = self.store.get(id, 'job')['payload']
            if payload.get('parent_job_id') and payload['status'] in ('queued','running'):
                # Experiment steps share one isolated process. Cancel its owning
                # protocol so no child remains falsely marked as active.
                return self.cancel(payload['parent_job_id'])
            if payload['status'] in ('queued', 'running'):
                self.store.update(id, status='cancelled', stage='Cancelled by user')
                if payload['task'] == 'experiment':
                    self.store.update(payload['config']['experiment_id'], status='cancelled')
                    for child in self.store.list('job'):
                        if child['payload'].get('parent_job_id') == id and child['payload']['status'] in ('running', 'queued'):
                            self.store.update(child['id'], status='cancelled', stage='Parent experiment cancelled')
                if payload['status'] == 'running' and self.process:
                    self.process.terminate()
        return self.store.get(id)

    def close(self):
        self.stopping = True
        with self.lock:
            if self.process:
                self.process.terminate()
        if self.thread:
            self.thread.join(timeout=10)
