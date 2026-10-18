#!/usr/bin/env python3
"""Plan files for cbindexplan, the call that runs it, and Couchbase's index sizing."""
import os
import json
import subprocess

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CBINDEXPLAN = os.environ.get("CBINDEXPLAN_BIN", os.path.join(_HERE, "..", "bin", "cbindexplan"))
if not os.path.exists(CBINDEXPLAN) and os.path.exists("/tmp/cbindexplan"):
    CBINDEXPLAN = "/tmp/cbindexplan"
LIBPATH = os.environ.get("CBINDEXPLAN_LIBS", "")
KEYSIZE = 200
DOCKEYSIZE = 50
BYTES_PER_DOC = 370


def _index_instance(defn_id, inst_id, name, numdoc, mutation=0, scan=0, bucket="b"):
    return {"defnId": defn_id, "instId": inst_id, "name": name, "bucket": bucket,
            "isMOI": True, "avgSecKeySize": KEYSIZE, "avgDocKeySize": DOCKEYSIZE,
            "avgArrSize": 0, "avgArrKeySize": 0, "numOfDocs": numdoc,
            "mutationRate": mutation, "scanRate": scan}


def live_fields(numdoc, mem_total, scan_rate, drain_rate):
    mem = int(mem_total)
    data = int(mem_total)
    return {
        "actualMemUsage": mem,
        "actualMemOverhead": 0,
        "actualMemMin": mem,
        "actualDataSize": data,
        "actualDiskUsage": data,
        "actualScanRate": int(round(scan_rate)),
        "actualDrainRate": int(round(drain_rate)),
        "actualNumDocs": int(numdoc),
        "actualBuildPercent": 100,
        "actualResidentPercent": 100,
    }


def write_plan(path, plan):
    json.dump(plan, open(path, "w"))


def run(args, cwd=None, timeout=120):
    def _cluster_up(host="127.0.0.1", port=9000, t=0.4):
        import socket
        try:
            with socket.create_connection((host, port), timeout=t):
                return True
        except OSError:
            return False

    env = dict(os.environ)
    if not _cluster_up():
        for k in [k for k in env if k.startswith("NS_SERVER_CBAUTH") or k == "CBAUTH_REVRPC_URL"]:
            env.pop(k, None)
    env["DYLD_LIBRARY_PATH"] = LIBPATH + ":" + env.get("DYLD_LIBRARY_PATH", "")
    try:
        r = subprocess.run([CBINDEXPLAN] + args, capture_output=True, text=True,
                           env=env, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        def _s(x):
            if x is None:
                return ""
            return x.decode(errors="replace") if isinstance(x, bytes) else x
        return _s(e.stdout) + _s(e.stderr) + f"\n[cbindexplan TIMEOUT after {timeout}s]"
    return r.stdout + r.stderr


def moi_mem_total(numdocs, seckey=KEYSIZE, dockey=DOCKEYSIZE, mutation=0):
    data = (120 + seckey + dockey) * numdocs
    snapshot = mutation * 120 * (seckey + dockey + 120)
    overhead = 150 * 1024 * 1024
    overhead += 30000 * (seckey + dockey) * mutation / 500
    overhead += snapshot
    mutq = (data + snapshot) * 0.10
    overhead += mutq
    overhead += (data + snapshot + mutq) * 0.05
    return data, data + overhead
