import { createHash, randomUUID } from "node:crypto";
import { mkdir, rename, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";

interface Snapshot { sha: string; path: string; files: number }
type Validate = (path: string) => Promise<void>;

/** Download an immutable commit, validate every file, then publish a new directory.
 * Activation is a separate atomic pointer update owned by the caller.
 */
export async function downloadLensfunSnapshot(root: string, validate: Validate, fetcher: typeof fetch = fetch, signal?: AbortSignal): Promise<Snapshot> {
  const headers = { "User-Agent": "SlashStitch/0.1", Accept: "application/vnd.github+json" };
  const get = async (url: string) => {
    const response = await fetcher(url, { headers, signal: signal ? AbortSignal.any([signal,AbortSignal.timeout(30_000)]) : AbortSignal.timeout(30_000) });
    if (!response.ok) throw new Error(`Lensfun download failed (${response.status}): ${new URL(url).pathname}`);
    return response;
  };
  const commit = await (await get("https://api.github.com/repos/lensfun/lensfun/commits/master")).json() as { sha?: string };
  if (!commit.sha || !/^[a-f0-9]{40}$/.test(commit.sha)) throw new Error("Invalid Lensfun commit identity");
  const sha = commit.sha;
  const tree = await (await get(`https://api.github.com/repos/lensfun/lensfun/git/trees/${sha}?recursive=1`)).json() as { truncated?: boolean; tree?: Array<{ path: string; type: string; sha: string; size?: number }> };
  if (tree.truncated || !Array.isArray(tree.tree)) throw new Error("Incomplete Lensfun file inventory");
  const entries = tree.tree.filter(item => item.type === "blob" && /^data\/db\/[^/]+\.xml$/.test(item.path));
  if(entries.some(item=>!/^data\/db\/[A-Za-z0-9][A-Za-z0-9_.-]*\.xml$/.test(item.path)||!/^[a-f0-9]{40}$/.test(item.sha)))throw new Error("Unsafe Lensfun inventory entry");
  if (!entries.length || entries.length > 2000) throw new Error("Invalid Lensfun XML inventory");
  await mkdir(root, { recursive: true });
  const staging = join(root, `.staging-${randomUUID()}`);
  const destination = join(root, `${sha}-${randomUUID()}`);
  await mkdir(staging, { mode: 0o700 });
  try {
    let total = 0;
    // Sequential bounded downloads ensure failures cannot leave parallel writers behind.
    for (const entry of entries) {
      signal?.throwIfAborted();
      if ((entry.size ?? 0) > 8 * 1024 * 1024) throw new Error("Lensfun XML exceeds the file budget");
      const response = await get(`https://raw.githubusercontent.com/lensfun/lensfun/${sha}/${entry.path}`);
      const reader = response.body?.getReader();
      if (!reader) throw new Error("Empty download stream");
      const chunks: Uint8Array[] = []; let size = 0;
      try {
        for (;;) {
          const next = await reader.read(); if (next.done) break;
          size += next.value.byteLength; total += next.value.byteLength;
          if (size > 8 * 1024 * 1024 || total > 64 * 1024 * 1024) throw new Error("Lensfun download exceeds the memory budget");
          chunks.push(next.value);
        }
      } finally { await reader.cancel().catch(() => {}); }
      const bytes = Buffer.concat(chunks);
      const blob = createHash("sha1").update(`blob ${bytes.length}\0`).update(bytes).digest("hex");
      if (blob !== entry.sha) throw new Error(`Lensfun integrity check failed: ${entry.path}`);
      await writeFile(join(staging, entry.path.split("/").at(-1)!), bytes, { flag: "wx", mode: 0o600 });
    }
    signal?.throwIfAborted();
    await validate(staging);
    signal?.throwIfAborted();
    await writeFile(join(staging, "snapshot.json"), JSON.stringify({ sha, files: entries.length }) + "\n");
    await rename(staging, destination);
    return { sha, path: destination, files: entries.length };
  } catch (error) {
    await rm(staging, { recursive: true, force: true });
    throw error;
  }
}
