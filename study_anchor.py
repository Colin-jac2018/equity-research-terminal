"""CC-2c: public timestamp anchors for the study chain (plan §2.3).

Each run, after the day's records (and grading) are written, the chain HEAD hash -- the latest
record's hash, which commits to every earlier record through the prev_hash links -- goes into an
append-only anchors ledger (date, seq, head_hash) and into a one-line file that OpenTimestamps
stamps (`ots stamp` -> <file>.ots). The chain, the ledger and the proofs are then committed and
pushed. OTS proofs start pending; the next run's `upgrade_pending` asks the calendars for the
Bitcoin attestation, and re-commits and pushes the upgraded proofs. `verify_anchor` checks the
head against the chain and the proof against the head and reports pending / confirmed / failed.

At freeze (before day 1), `anchor_plan` stamps the plan file's hash plus the freeze commit.

Gaps stay gaps: an anchor is only ever for a date after the ledger's last one, so a missing day is
never backfilled. Requires the OpenTimestamps client (`pip install opentimestamps-client`).

    python study_anchor.py verify [--chain PATH]          # every anchor: confirmed / attested / pending / failed
    python study_anchor.py upgrade [--chain PATH] [--no-publish]
"""
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys

import study_config as C
import study_data as D
import study_log as L

LEDGER = "anchors.jsonl"
LEDGER_FIELDS = ("date", "seq", "head_hash", "head_file", "proof_file", "stamped_at")


class AnchorError(RuntimeError):
    """An anchor could not be written or checked; nothing was appended."""


def ots_path():
    """The `ots` CLI: on PATH, else next to this venv's python. None when not installed."""
    exe = shutil.which("ots")
    if exe:
        return exe
    for name in ("ots.exe", "ots"):
        cand = os.path.join(os.path.dirname(sys.executable), name)
        if os.path.exists(cand):
            return cand
    return None


def _ots_env():
    """The client's environment. ots imports python-bitcoinlib, which ctypes-loads an OpenSSL DLL by its
    old name ('ssl' / 'libeay32') at import time; Windows has none on PATH, so every ots command dies
    before it starts. Python ships OpenSSL 3's libcrypto (DLLs/libcrypto-3.dll); a copy named
    libeay32.dll in a shim directory next to the venv, put first on PATH, satisfies the import. Stamp,
    upgrade and verify never use bitcoinlib's key code, so nothing else depends on it."""
    import ctypes.util
    env = dict(os.environ)
    if os.name != "nt" or ctypes.util.find_library("ssl") or ctypes.util.find_library("libeay32"):
        return env
    src = os.path.join(sys.base_prefix, "DLLs", "libcrypto-3.dll")
    shim = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "ots_ssl_shim")
    dll = os.path.join(shim, "libeay32.dll")
    if not os.path.exists(dll):
        if not os.path.exists(src):
            raise AnchorError(f"ots needs an OpenSSL DLL and {src} is missing")
        os.makedirs(shim, exist_ok=True)
        shutil.copyfile(src, dll)
    env["PATH"] = shim + os.pathsep + env.get("PATH", "")
    return env


def _ots(*args, timeout=120):
    exe = ots_path()
    if exe is None:
        raise AnchorError("the OpenTimestamps client is not installed: pip install opentimestamps-client")
    p = subprocess.run([exe, *args], capture_output=True, text=True, timeout=timeout, env=_ots_env())
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def anchor_dir(chain_path):
    return os.path.join(os.path.dirname(os.path.abspath(chain_path)), "anchors")


def read_ledger(adir):
    path = os.path.join(adir, LEDGER)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def chain_head(chain_path):
    """(seq, hash) of the chain's last record; (-1, GENESIS) for an empty chain."""
    seq, head = -1, L.GENESIS
    if os.path.exists(chain_path):
        for rec in L.iter_records(chain_path):
            seq, head = rec["seq"], rec["hash"]
    return seq, head


def _head_text(head_hash):
    return head_hash + "\n"


def anchor_day(chain_path, date, stamp=None):
    """Append (date, seq, head_hash) to the ledger and stamp the head file. `stamp(path)` makes
    <path>.ots (default: `ots stamp`). Refuses a date at or before the ledger's last one (no backfill)
    and a chain that doesn't verify. Returns the ledger entry."""
    if not C.STUDY_MODE:
        raise AnchorError("STUDY_MODE is off: no anchors are written")
    probs = L.verify(chain_path)
    if probs:
        raise AnchorError(f"chain does not verify; not anchoring: {probs[:3]}")
    adir = anchor_dir(chain_path)
    ledger = read_ledger(adir)
    if ledger and date <= ledger[-1]["date"]:
        raise AnchorError(f"{date} is not after the last anchor {ledger[-1]['date']} (anchors are never backfilled)")
    seq, head = chain_head(chain_path)
    if seq < 0:
        raise AnchorError("the chain is empty")
    if ledger and seq <= ledger[-1]["seq"]:
        raise AnchorError(f"chain head seq {seq} has not moved past the last anchor's seq {ledger[-1]['seq']}")
    os.makedirs(adir, exist_ok=True)
    head_file = os.path.join(adir, f"{date}.head")
    if os.path.exists(head_file) or os.path.exists(head_file + ".ots"):
        raise AnchorError(f"{head_file} already exists")
    with open(head_file, "w", encoding="utf-8", newline="\n") as f:
        f.write(_head_text(head))
    (stamp or _ots_stamp)(head_file)
    if not os.path.exists(head_file + ".ots"):
        raise AnchorError(f"stamping produced no proof for {head_file}")
    entry = {"date": date, "seq": seq, "head_hash": head, "head_file": os.path.basename(head_file),
             "proof_file": os.path.basename(head_file) + ".ots",
             "stamped_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
    with open(os.path.join(adir, LEDGER), "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(entry, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    return entry


def _ots_stamp(path):
    code, out = _ots("stamp", path)
    if code != 0:
        raise AnchorError(f"ots stamp failed: {out.strip()[:500]}")


def verify_anchor(chain_path, entry, ots_verify=None):
    """{"date", "status": confirmed | attested | pending | failed, "detail"} (see classify_verify). Local checks first (the ledger's
    head_hash is the hash of the chain record at its seq, and the head file holds exactly that hash),
    then the proof against the head file (default: `ots verify`)."""
    adir = anchor_dir(chain_path)
    out = {"date": entry["date"], "seq": entry["seq"]}
    at = next((r for r in L.iter_records(chain_path) if r.get("seq") == entry["seq"]), None)
    if at is None or at.get("hash") != entry["head_hash"] or L.record_hash(at) != entry["head_hash"]:
        return {**out, "status": "failed", "detail": "ledger head_hash is not the chain record at that seq"}
    head_file = os.path.join(adir, entry["head_file"])
    if not os.path.exists(head_file) or open(head_file, encoding="utf-8").read() != _head_text(entry["head_hash"]):
        return {**out, "status": "failed", "detail": "head file does not hold the ledger's head_hash"}
    status, detail = (ots_verify or _ots_verify)(head_file, os.path.join(adir, entry["proof_file"]))
    return {**out, "status": status, "detail": detail}


# Messages of opentimestamps-client 0.8.0 (otsclient/cmds.py). Exit codes don't separate the cases
# (pending and a digest mismatch both exit 1), so the text is what's read.
_MISMATCH = "file does not match original"
_BTC_FAILED = "bitcoin verification failed"
_CONFIRMED = re.compile(r"Success! Bitcoin block (\d+) attests existence as of (\S+)")
_PENDING = "pending confirmation in bitcoin blockchain"
_UNCHECKED = re.compile(r"check that Bitcoin block (\d+) has merkleroot ([0-9a-f]+)")
_NODE_TROUBLE = ("could not connect to local bitcoin node", "bitcoin block height", "traceback", "bitcoin.conf")


def classify_verify(out, out_no_btc=None):
    """(status, detail) from `ots verify` output, and, when the node couldn't check a Bitcoin
    attestation, from `ots --no-bitcoin verify`.
    confirmed: a Bitcoin node checked the attestation against the block header.
    attested:  the proof carries a Bitcoin attestation (block height + merkle root reported) that no
               node checked here; check the merkle root against any block explorer.
    pending:   only calendar promises so far; `ots upgrade` later.
    failed:    digest mismatch (tampered head), a failed Bitcoin check, or anything unrecognized."""
    low = out.lower()
    if _MISMATCH in low:
        return "failed", "File does not match original (head file differs from what was stamped)"
    if _BTC_FAILED in low:
        return "failed", out.strip()[:500]
    m = _CONFIRMED.search(out)
    if m:
        return "confirmed", f"Bitcoin block {m.group(1)} attests existence as of {m.group(2)}"
    if out_no_btc is not None:
        m = _UNCHECKED.search(out_no_btc)
        if m:
            return "attested", f"Bitcoin block {m.group(1)} merkleroot {m.group(2)} (not checked against a node)"
        if _PENDING in out_no_btc.lower():
            return "pending", "pending confirmation in the Bitcoin blockchain"
        return "failed", out_no_btc.strip()[:500]
    if _PENDING in low:
        return "pending", "pending confirmation in the Bitcoin blockchain"
    return "failed", out.strip()[:500]


def _ots_verify(head_file, proof):
    """`ots verify` the proof against the head file -> classify_verify. When the output shows a Bitcoin
    attestation the local node couldn't check (no node, or none reachable), re-read the proof with
    --no-bitcoin to report it as attested rather than failed."""
    if not os.path.exists(proof):
        return "failed", "no proof file"
    _, out = _ots("verify", "-f", head_file, proof)
    low = out.lower()
    if any(s in low for s in _NODE_TROUBLE) and _MISMATCH not in low and not _CONFIRMED.search(out):
        _, out2 = _ots("--no-bitcoin", "verify", "-f", head_file, proof)
        return classify_verify(out, out2)
    return classify_verify(out)


def upgrade_pending(chain_path, ots_upgrade=None):
    """`ots upgrade` every proof whose verify status is pending. Returns the proofs that changed."""
    adir, changed = anchor_dir(chain_path), []
    for e in read_ledger(adir):
        proof = os.path.join(adir, e["proof_file"])
        if verify_anchor(chain_path, e)["status"] != "pending":
            continue
        before = open(proof, "rb").read()
        (ots_upgrade or (lambda p: _ots("upgrade", p)))(proof)
        if os.path.exists(proof + ".bak"):
            os.remove(proof + ".bak")          # ots keeps the pre-upgrade proof as .bak; git holds history
        if open(proof, "rb").read() != before:
            changed.append(proof)
    return changed


def publish(paths, message, repo=None, push=True, remote="origin"):
    """Commit exactly `paths` (files or directories; nothing else that happens to be staged) and push.
    Returns the commit hash, or None when none of the paths changed. The push runs either way, so a
    re-run recovers a commit whose earlier push failed."""
    repo = repo or D.HERE
    rel = [os.path.relpath(os.path.abspath(p), repo) for p in paths]
    git = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True)
    add = git("add", "--", *rel)
    if add.returncode != 0:
        raise AnchorError(f"git add failed: {add.stderr.strip()}")
    sha = None
    if git("diff", "--cached", "--name-only", "--", *rel).stdout.strip():
        c = git("commit", "-m", message, "--", *rel)
        if c.returncode != 0:
            raise AnchorError(f"git commit failed: {(c.stdout + c.stderr).strip()[:500]}")
        sha = git("rev-parse", "HEAD").stdout.strip()
    if push:
        branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        p = git("push", remote, branch)
        if p.returncode != 0:
            raise AnchorError(f"git push failed (commit {sha and sha[:12]} is local; a re-run pushes it): {p.stderr.strip()[:500]}")
    return sha


def anchor_paths(chain_path):
    """What a run publishes: the chain and the whole anchors directory (ledger, heads, proofs), so a
    file left uncommitted by an earlier failed run goes out with the next one."""
    return [chain_path, anchor_dir(chain_path)]


def anchor_plan(plan_path, freeze_commit, out_dir, stamp=None, repo=None, push=True):
    """At freeze, before day 1: stamp a file holding the plan file's SHA-256 and the freeze commit,
    then commit and push the file and its proof. Returns (record file, commit hash)."""
    if not C.STUDY_MODE:
        raise AnchorError("STUDY_MODE is off: the plan anchor runs at freeze, with STUDY_MODE=1 set")
    os.makedirs(out_dir, exist_ok=True)
    rec = os.path.join(out_dir, "plan_freeze.txt")
    if os.path.exists(rec):
        raise AnchorError(f"{rec} already exists; the plan is anchored once")
    plan_sha = D.sha256(open(plan_path, encoding="utf-8").read())
    with open(rec, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"plan {os.path.basename(plan_path)} sha256 {plan_sha}\nfreeze_commit {freeze_commit}\n")
    (stamp or _ots_stamp)(rec)
    sha = publish([rec, rec + ".ots"], f"Edge study: plan freeze anchor ({freeze_commit[:12]})", repo=repo, push=push)
    return rec, sha


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("verify", "upgrade"))
    ap.add_argument("--chain", default=C.STUDY_LOG_PATH)
    ap.add_argument("--no-publish", action="store_true")
    a = ap.parse_args(argv)
    chain = a.chain if os.path.isabs(a.chain) else os.path.join(D.HERE, a.chain)
    if a.cmd == "verify":
        for e in read_ledger(anchor_dir(chain)):
            print(verify_anchor(chain, e))
        return
    changed = upgrade_pending(chain)
    print("upgraded:", [os.path.basename(p) for p in changed])
    if changed and not a.no_publish:
        print("published:", publish(changed, f"Edge study: upgrade {len(changed)} OTS proof(s)"))


if __name__ == "__main__":
    main()
