"""Verify the native manifests and record the four packaged worker identities."""
from base64 import b64decode
import hashlib
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from publish import ROOT, api, digest, require


def worker_info(read, package, platform, arch, project_license):
    require(read("LICENSE").replace(b"\r\n", b"\n") == project_license, f"Wrong project license in {package}")
    metadata = json.loads(read("bin/engine-build.json"))
    worker = read("bin/slashstitch-worker" + (".exe" if platform == "win32" else ""))
    licenses = json.loads(read("licenses/rust-dependencies.json"))
    require((metadata["platform"], metadata["arch"]) == (platform, arch), f"Wrong worker in {package}")
    require(metadata["sha256"] == hashlib.sha256(worker).hexdigest(), f"Wrong worker digest in {package}")
    require(licenses and all(entry.get("name") for entry in licenses), f"Missing licenses in {package}")
    result = {"package": package, "metadata": metadata, "license_entries": len(licenses)}
    if platform == "win32":
        with TemporaryDirectory() as tmp:
            exe = Path(tmp) / "worker.exe"
            exe.write_bytes(worker)
            imports = subprocess.check_output(["objdump", "-p", exe], text=True)
            result["dlls"] = re.findall(r"DLL Name:\s+(\S+)", imports, re.IGNORECASE)
            require(result["dlls"], "Missing PE imports")
    return result


def validate(packages, version, run):
    project_license = b64decode(api(f"contents/LICENSE?ref={run['head_sha']}")["content"])
    layout = json.loads((ROOT / "packages.json").read_text())
    assets = []
    for artifact, entries in layout.items():
        files = {k.replace("VERSION", version): v for k, v in entries.items()}
        manifest, = [name for name, kind in files.items() if kind == "verify"]
        expected = {name: digest(packages / name) for name, kind in files.items() if kind == "publish"}
        lines = (packages.parent / artifact / manifest).read_text().splitlines()
        require(sorted(lines) == sorted(f"{h}  {name}" for name, h in expected.items()), f"Wrong {manifest}")
        assets.extend({"name": n, "sha256": h, "bytes": (packages / n).stat().st_size} for n, h in expected.items())
    workers = []
    deb, = packages.glob("*.deb")
    with TemporaryDirectory() as tmp:
        subprocess.run(["dpkg-deb", "-x", deb, tmp], check=True)
        metadata, = Path(tmp).rglob("resources/bin/engine-build.json")
        workers.append(worker_info(lambda n: (metadata.parents[1] / n).read_bytes(), deb.name, "linux", "x64", project_license))
    for archive in sorted(packages.glob("*.zip")):
        with ZipFile(archive) as z:
            suffix = "bin/engine-build.json"
            metadata, = [name for name in z.namelist() if name.lower().endswith("resources/" + suffix)]
            platform = "win32" if "-win-" in archive.name else "darwin"
            arch = "arm64" if "-arm64." in archive.name else "x64"
            workers.append(worker_info(lambda n: z.read(metadata[:-len(suffix)] + n), archive.name, platform, arch, project_license))
    print("Verified packaged LICENSE against the build commit on all four platforms", flush=True)
    jobs = api(f"actions/runs/{run['id']}/attempts/{run['run_attempt']}/jobs")["jobs"]
    require({j["name"] for j in jobs if j["conclusion"] == "success"} ==
            {"Linux x64", "Windows x64", "macOS x64", "macOS arm64"}, "Native job matrix differs")
    info = {"version": version, "source_commit": run["head_sha"],
            "source_url": run["head_repository"]["html_url"] + "/commit/" + run["head_sha"],
            "workflow_url": run["html_url"], "workers": workers,
            "native_jobs": [{k: j[k] for k in ("id", "name", "status", "conclusion")} for j in jobs],
            "assets": sorted(assets, key=lambda a: a["name"]),
            "validation_limits": ["Package inspection and native CI checks. End-user device testing is separate."]}
    (packages / "BUILD_INFO.json").write_text(json.dumps(info, indent=2) + "\n")
