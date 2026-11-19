"""Promote the exact successful main-branch CI artifacts for a version tag."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
from zipfile import ZipFile

REPO = os.environ["GITHUB_REPOSITORY"]
TAG = os.environ["RELEASE_TAG"]
ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def gh(*args):
    return subprocess.check_output(["gh", *map(str, args)], text=True).strip()


def api(path):
    return json.loads(gh("api", f"repos/{REPO}/{path}"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tagged_commit():
    ref = api(f"git/ref/tags/{TAG}")["object"]
    while ref["type"] == "tag":
        ref = api(f"git/tags/{ref['sha']}")["object"]
    require(ref["type"] == "commit", "Tag must resolve to a commit")
    return ref["sha"]


def build_for(sha):
    for _ in range(120):
        runs = json.loads(gh("run", "list", "-R", REPO, "--workflow", "ci.yml", "--branch", "main",
                             "--commit", sha, "--event", "push", "--limit", "1", "--json", "databaseId,status"))
        if runs and runs[0]["status"] == "completed":
            run = api(f"actions/runs/{runs[0]['databaseId']}")
            require(run["head_repository"]["full_name"] == REPO and run["head_sha"] == sha
                    and run["event"] == "push" and run["head_branch"] == "main"
                    and run["path"] == ".github/workflows/ci.yml" and run["conclusion"] == "success",
                    f"Tagged main build is not a successful Verify and package run: {run['html_url']}")
            return run
        time.sleep(30)
    raise RuntimeError(f"No completed main push build for {sha}")


def collect(run, temp, version):
    layout = json.loads((ROOT / "packages.json").read_text())
    artifacts = api(f"actions/runs/{run['id']}/artifacts?per_page=100")["artifacts"]
    require(len(artifacts) == len(layout) and {a["name"] for a in artifacts} == set(layout),
            "CI artifacts differ from the expected platform matrix")
    packages = temp / "packages"
    packages.mkdir()
    for artifact in artifacts:
        mapping = {k.replace("VERSION", version): v for k, v in layout[artifact["name"]].items()}
        archive = temp / "platform.zip"
        require(not artifact["expired"], f"Expired artifact: {artifact['name']}")
        with archive.open("wb") as output:
            subprocess.run(["gh", "api", f"repos/{REPO}/actions/artifacts/{artifact['id']}/zip"], stdout=output, check=True)
        require("sha256:" + digest(archive) == artifact["digest"], f"Wrong archive digest: {artifact['name']}")
        print(f"Verified {artifact['name']} from run {run['id']}", flush=True)
        with ZipFile(archive) as z:
            names = [member.filename for member in z.infolist() if not member.is_dir()]
            require(len(names) == len(mapping) and set(names) == set(mapping), f"Unexpected files: {names}")
            for name in names:
                require(Path(name).name == name, "Unexpected nested package")
                if mapping[name] == "skip":
                    continue
                group = packages if mapping[name] == "publish" else temp / artifact["name"]
                group.mkdir(exist_ok=True)
                with z.open(name) as src, (group / name).open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                require((group / name).stat().st_size > 0, f"Empty file: {name}")
        archive.unlink()
    return packages


def release_assets(release_id):
    entries = json.loads(gh("api", "--paginate", "--slurp", f"repos/{REPO}/releases/{release_id}/assets?per_page=100"))
    return {a["name"]: a for page in entries for a in page}


def publish(packages, run, sha):
    release_list = json.loads(gh("api", "--paginate", "--slurp", f"repos/{REPO}/releases?per_page=100"))
    existing = [r for page in release_list for r in page if r["tag_name"] == TAG]
    require(len(existing) <= 1, "Ambiguous release tag")
    if existing:
        release = existing[0]
    else:
        product = next(iter(json.loads((ROOT / "packages.json").read_text()))).split("-")[0]
        request = {"tag_name": TAG, "target_commitish": sha, "draft": True, "generate_release_notes": True,
                   "name": f"{product} {TAG[1:]}"}
        release = json.loads(subprocess.check_output(["gh", "api", "-X", "POST", f"repos/{REPO}/releases", "--input", "-"],
                                                     input=json.dumps(request), text=True))
    require(release["tag_name"] == TAG, "Unexpected release")
    remote = release_assets(release["id"])
    expected = {p.name: ("sha256:" + digest(p), p.stat().st_size) for p in packages.iterdir()}
    require(set(remote) <= set(expected), "Unexpected assets already attached")
    for name, identity in expected.items():
        if name in remote:
            require((remote[name]["digest"], remote[name]["size"]) == identity, f"Existing asset differs: {name}")
        else:
            require(release["draft"], f"Published release is missing {name}")
            print(f"Uploading {name}", flush=True)
            gh("release", "upload", TAG, "-R", REPO, packages / name)
    final = release_assets(release["id"])
    require({k: (a["digest"], a["size"]) for k, a in final.items()} == expected, "Uploaded assets differ")
    require(tagged_commit() == sha, "Tag moved during publication")
    if release["draft"]:
        gh("release", "edit", TAG, "-R", REPO, "--draft=false")
    result = api(f"releases/{release['id']}")
    require(not result["draft"] and result["tag_name"] == TAG, "Release did not become public")
    print(f"Published {len(expected)} files from {run['html_url']}: {result['html_url']}")


def main():
    require(re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+", TAG), "Expected a vMAJOR.MINOR.PATCH tag")
    sha = tagged_commit()
    run = build_for(sha)
    with tempfile.TemporaryDirectory() as directory:
        temp = Path(directory)
        packages = collect(run, temp, TAG[1:])
        from validate import validate
        validate(packages, TAG[1:], run)
        checksums = "".join(f"{digest(path)}  {path.name}\n" for path in sorted(packages.iterdir()))
        (packages / "SHA256SUMS.txt").write_text(checksums)
        print(checksums, flush=True)
        if os.environ.get("VERIFY_ONLY", "").lower() != "true":
            publish(packages, run, sha)


if __name__ == "__main__":
    main()
