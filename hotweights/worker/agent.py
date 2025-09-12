"""
SOTA Worker Agent using the CUDA-IPC transport layer.

This agent loop is greatly simplified by the new SOTA modules:
  1) Registers with coordinator.
  2) Fetches plan.
  3) Replicates using the unified CudaIPCTransport.
  4) Commits the update.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from ..coordinator.zmq_client import Client, Subscriber
from ..staging.cuda_ipc_agent import CudaIPCAgent
from ..transport.cuda_ipc import CudaIPCTransport
from ..telemetry.cuda_ipc_metrics import CudaIPCMetrics
from ..telemetry.prom import start_http_server


@dataclass
class WorkerConfig:
    """Configuration for the SOTA worker agent."""
    endpoint: str = "tcp://127.0.0.1:5555"
    device: str = "cuda"
    use_sub: bool = True
    pub_endpoint: Optional[str] = None
    event_token: Optional[str] = None


def _derive_pub_endpoint(endpoint: str) -> str:
    try:
        if endpoint.startswith("tcp://") and ":" in endpoint.rsplit(":", 1)[-1]:
            host, port_s = endpoint[len("tcp://") :].rsplit(":", 1)
            port = int(port_s)
            return f"tcp://{host}:{port+1}"
    except Exception:
        pass
    return "tcp://127.0.0.1:5556"


def run_worker(cfg: WorkerConfig) -> int:
    """Main loop for the SOTA worker agent."""
    # 1. Initialization
    c = Client(cfg.endpoint)
    worker_id = os.getenv("WORKER_ID", f"pid:{os.getpid()}")
    rank = int(os.getenv("RANK", "0"))

    try:
        metrics = CudaIPCMetrics(rank)
        agent = CudaIPCAgent(device=cfg.device, metrics=metrics)
        transport = CudaIPCTransport(agent=agent, metrics=metrics, coord_endpoint=cfg.endpoint)
        try:
            port = int(os.getenv("HOTWEIGHTS_METRICS_PORT", "9099"))
            start_http_server(port)  # Start Prometheus metrics server
        except Exception:
            pass
    except Exception as e:
        print(f"Fatal: Failed to initialize SOTA modules. Is PyTorch with CUDA installed? Error: {e}")
        return 1

    # Register with coordinator
    c.call("register", worker_id=worker_id, caps={"transport": "cuda_ipc", "rank": rank})
    c.call("heartbeat", worker_id=worker_id)

    # 2. Plan Acquisition
    plan = None
    if cfg.use_sub:
        print("Waiting for 'begin' event via PUB/SUB...")
        pub_ep = cfg.pub_endpoint or os.getenv("HOTWEIGHTS_COORD_PUB") or _derive_pub_endpoint(cfg.endpoint)
        sub = Subscriber(pub_ep)
        # This loop waits for the coordinator to broadcast the start of an update
        while True:
            topic, payload = sub.recv()
            if topic == "begin":
                break
            time.sleep(0.1)
    
    # Fetch the plan from the coordinator
    plan_resp = c.call("get_plan")
    plan = plan_resp.get("plan")
    if not plan:
        print("No plan available from coordinator; exiting.")
        return 0

    # 3. Replication
    print(f"Starting replication for version {plan.get('version')}...")
    transport.replicate(plan)
    print("Replication complete.")

    # 4. Precommit and Commit
    c.call("precommit", worker_id=worker_id)
    
    # Wait for the final commit signal from the coordinator
    print("Waiting for final commit signal...")
    while True:
        topic, payload = sub.recv()
        if topic == "commit" and payload.get("version") == plan.get("version"):
            if payload.get("accepted"):
                print("Commit signal received and accepted.")
                break
            else:
                print("Commit signal received but was not accepted by quorum. Aborting.")
                return 1
        time.sleep(0.1)

    c.call("heartbeat", worker_id=worker_id)
    print("Worker finished successfully.")
    return 0
