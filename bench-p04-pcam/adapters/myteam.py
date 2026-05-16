import json
import subprocess
import os
import atexit
import numpy as np

from adapter import Adapter


class Engine(Adapter):
    """PCAM precision agent (R backend).

    This python wrapper forwards all queries to the core intelligence layer
    implemented in R (`core.R`). The benchmark evaluation harness remains
    in Python.
    """

    def __init__(self, stored_patterns, model_params):
        self.X = np.asarray(stored_patterns, dtype=np.float64)
        _, self.N = self.X.shape

        # Resolve paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        r_script = os.path.join(current_dir, 'core.R')

        # Start R process
        self.process = subprocess.Popen(
            [r'C:\Program Files\R\R-4.6.0\bin\Rscript.exe', r_script],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        # Ensure cleanup
        atexit.register(self.cleanup)

        # Convert R to list if present
        params_copy = dict(model_params)
        if "R" in params_copy and isinstance(params_copy["R"], np.ndarray):
            params_copy["R"] = params_copy["R"].tolist()

        # Send init payload
        init_payload = {
            "type": "init",
            "data": {
                "stored_patterns": self.X.tolist(),
                "model_params": params_copy
            }
        }
        self._send_request(init_payload)

    def _send_request(self, payload):
        if self.process.poll() is not None:
            err = self.process.stderr.read()
            raise RuntimeError(f"R subprocess crashed: {err}")
            
        req_str = json.dumps(payload)
        self.process.stdin.write(req_str + "\n")
        self.process.stdin.flush()
        
        resp_str = self.process.stdout.readline()
        if not resp_str:
            err = self.process.stderr.read()
            raise RuntimeError(f"R subprocess closed unexpectedly. stderr: {err}")
            
        resp = json.loads(resp_str)
        if "error" in resp:
            raise RuntimeError(f"R process error: {resp['error']}")
        return resp

    def predict_precision(self, corrupted_query):
        q = np.asarray(corrupted_query, dtype=np.float64)
        payload = {
            "type": "predict",
            "q": q.tolist()
        }
        resp = self._send_request(payload)
        return np.array(resp["pi"], dtype=np.float64)

    def cleanup(self):
        if hasattr(self, 'process') and self.process.poll() is None:
            try:
                self.process.stdin.write(json.dumps({"type": "exit"}) + "\n")
                self.process.stdin.flush()
                self.process.communicate(timeout=1)
            except Exception:
                self.process.kill()

    def __del__(self):
        self.cleanup()
