import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";
import { createInterface } from "node:readline";
export interface WorkerReply { ok: boolean; text: string }
interface Pending { child: ChildProcessWithoutNullStreams; finish: (reply: WorkerReply) => void }

/** Multiplexed control messages. Timeout cancels waiting, never the backend job. */
export class WorkerBridge extends EventEmitter {
  private child: ChildProcessWithoutNullStreams | null = null;
  private pending = new Map<string, Pending>();
  private serial = 0;
  private unavailable: string | null = null;
  private stopping: Promise<void> | null = null;
  constructor(private command: string, private args: string[], private extraEnv: Record<string, string> = {}) { super(); }

  private start(): ChildProcessWithoutNullStreams {
    const env = { ...process.env };
    delete env.ELECTRON_RUN_AS_NODE;
    Object.assign(env, this.extraEnv);
    const child = spawn(this.command, this.args, { env, stdio: "pipe", windowsHide: true });
    const lines = createInterface({ input: child.stdout });
    lines.on("line", (line) => this.receive(child, line));
    child.stderr.on("data", (data: Buffer) => this.emit("event", "stderr", { text: data.toString("utf8") }));
    const failed = (message: string) => {
      this.unavailable = message;
      this.fail(child, message);
      this.emit("event", "worker-error", { text: message });
    };
    child.on("error", (error) => failed(`Worker could not start: ${error.message}`));
    child.stdin.on("error", (error) => failed(`Worker input failed: ${error.message}`));
    child.stdout.on("error", (error) => failed(`Worker output failed: ${error.message}`));
    child.stderr.on("error", (error) => failed(`Worker diagnostics failed: ${error.message}`));
    child.on("close", (code, signal) => {
      lines.close();
      if (this.child === child) this.child = null;
      if (!this.stopping) this.unavailable = `Worker stopped (${code ?? signal ?? "unknown"}). Restart the engine to restore the project.`;
      this.fail(child, `Worker stopped (${code ?? signal ?? "unknown"}).`);
      this.emit("event", "worker-exit", { code, signal });
    });
    return child;
  }
  private receive(child: ChildProcessWithoutNullStreams, line: string): void {
    if (!line.trim()) return;
    let msg: unknown;
    try { msg = JSON.parse(line); } catch { msg = null; }
    if (!msg || typeof msg !== "object" || Array.isArray(msg)) {
      this.emit("event", "protocol-error", { text: "Worker sent a non-object JSON message." });
      return;
    }
    const value = msg as Record<string, unknown>;
    if (typeof value.event === "string") {
      if (this.child === child) this.emit("event", value.event, value.data);
      return;
    }
    if (typeof value.id !== "string" || typeof value.ok !== "boolean") {
      this.emit("event", "protocol-error", { text: "Worker reply requires string id and boolean ok." });
      return;
    }
    const call = this.pending.get(value.id);
    if (!call || call.child !== child) return;
    call.finish(value.ok ? { ok: true, text: JSON.stringify(value.result ?? null) } : { ok: false, text: String(value.error ?? "Unknown worker error") });
  }
  private fail(child: ChildProcessWithoutNullStreams, message: string): void {
    for (const call of [...this.pending.values()]) if (call.child === child) call.finish({ ok: false, text: message });
  }
  request(action: string, payload: unknown = null, timeoutMs = 5000): Promise<WorkerReply> {
    if (this.unavailable && action !== "shutdown") return Promise.resolve({ok:false,text:this.unavailable});
    if (this.stopping && action !== "shutdown") return Promise.resolve({ ok: false, text: "Worker is shutting down." });
    if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) return Promise.resolve({ ok: false, text: "Invalid control timeout." });
    return new Promise((resolve) => {
      let id: string | undefined;
      let timer: ReturnType<typeof setTimeout> | undefined;
      const finish = (reply: WorkerReply) => {
        clearTimeout(timer);
        if (id) this.pending.delete(id);
        resolve(reply);
      };
      try {
        const child = this.child ??= this.start();
        id = `w${++this.serial}`;
        const line = JSON.stringify({ id, action, payload }) + "\n";
        this.pending.set(id, { child, finish });
        timer = setTimeout(() => finish({ ok: false, text: `Control request timed out: ${action}. Backend job was not cancelled.` }), timeoutMs);
        child.stdin.write(line, (error) => { if (error) finish({ ok: false, text: `Worker input failed: ${error.message}` }); });
      } catch (error) { finish({ ok: false, text: `Worker error: ${(error as Error).message}` }); }
    });
  }
  restart(): void {
    if (this.child) throw new Error("The previous worker has not exited; wait for it to stop before restarting.");
    this.unavailable = null;
  }
  /** Backend must finalize files, stop its children, acknowledge shutdown and exit. No forced kill. */
  stop(timeoutMs = 5000): Promise<void> {
    if (this.stopping) return this.stopping;
    const child = this.child;
    if (!child) return Promise.resolve();
    this.stopping = this.shutdown(child, timeoutMs).finally(() => { this.stopping = null; });
    return this.stopping;
  }
  private async shutdown(child: ChildProcessWithoutNullStreams, timeoutMs: number): Promise<void> {
    let timer: ReturnType<typeof setTimeout> | undefined;
    let onClose: (code: number | null) => void = () => {};
    const closed = new Promise<number | null | "timeout">((resolve) => {
      onClose = (code) => resolve(code);
      child.once("close", onClose);
      timer = setTimeout(() => resolve("timeout"), timeoutMs);
    });
    try {
      const reply = await this.request("shutdown", null, timeoutMs);
      if (!reply.ok) throw new Error(reply.text);
      const code = await closed;
      if (code !== 0) throw new Error(`Worker did not exit cleanly (${code}). Application remains open.`);
    } finally { clearTimeout(timer); child.off("close", onClose); }
  }
}
