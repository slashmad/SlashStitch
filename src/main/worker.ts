import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { EventEmitter } from "node:events";
import { createInterface } from "node:readline";
export interface WorkerReply { ok: boolean; text: string }
interface Pending { child: ChildProcessWithoutNullStreams; finish: (reply: WorkerReply) => void }
export interface StartupPolicy {
  probeAction: "engine-info";
  timeoutMs: number;
  fallbackEnv?: Record<string, string>;
  retryExitCodes: number[];
}

function exitDescription(code: number | null, signal: string | null): string {
  if (code === null) return signal ?? "unknown";
  if (code < 0 && code > -65536) return String(code); // libuv spawn errors, not NTSTATUS
  const unsigned = code >>> 0;
  if (unsigned < 0x80000000) return String(code);
  const name = unsigned === 0xc0000005 ? ", access violation" : unsigned === 0xc000001d ? ", illegal instruction" : unsigned === 0xc0000409 ? ", native fast-fail" : "";
  return `${code} / 0x${unsigned.toString(16).toUpperCase().padStart(8, "0")}${name}`;
}

/** Multiplexed control messages. Timeout cancels waiting, never the backend job. */
export class WorkerBridge extends EventEmitter {
  private child: ChildProcessWithoutNullStreams | null = null;
  private pending = new Map<string, Pending>();
  private serial = 0;
  private unavailable: string | null = null;
  private stopping: Promise<void> | null = null;
  private startupPromise: Promise<WorkerReply> | null = null;
  private initializing = false;
  private startupAborted = false;
  private lastExit: {code: number | null; signal: string | null} = {code:null,signal:null};
  private spawnFailed = false;
  private diagnostics = "";
  constructor(private command: string, private args: string[], private extraEnv: Record<string, string> = {}, private startupPolicy?: StartupPolicy) { super(); }

  private start(): ChildProcessWithoutNullStreams {
    const env = { ...process.env };
    delete env.ELECTRON_RUN_AS_NODE;
    Object.assign(env, this.extraEnv);
    this.lastExit = {code:null,signal:null}; this.spawnFailed = false; this.diagnostics = "";
    const child = spawn(this.command, this.args, { env, stdio: "pipe", windowsHide: true });
    const lines = createInterface({ input: child.stdout });
    lines.on("line", (line) => this.receive(child, line));
    child.stderr.on("data", (data: Buffer) => {
      const text = data.toString("utf8"); this.diagnostics = (this.diagnostics + text).slice(-8192);
      this.emit("event", "stderr", { text });
    });
    const failed = (message: string) => {
      this.unavailable = message;
      this.fail(child, message);
      if (!this.initializing) this.emit("event", "worker-error", { text: message });
    };
    child.on("error", (error) => { this.spawnFailed = true; failed(`Worker could not start: ${error.message}`); });
    child.stdin.on("error", (error) => failed(`Worker input failed: ${error.message}`));
    child.stdout.on("error", (error) => failed(`Worker output failed: ${error.message}`));
    child.stderr.on("error", (error) => failed(`Worker diagnostics failed: ${error.message}`));
    child.on("close", (code, signal) => {
      lines.close();
      this.lastExit = {code,signal};
      if (this.child === child) this.child = null;
      const message = `Worker stopped (${exitDescription(code, signal)}).`;
      if (!this.stopping) this.unavailable = `${message} Restart the engine to restore the project.`;
      this.fail(child, message);
      if (!this.initializing) this.emit("event", "worker-exit", { code, signal });
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
    if (this.startupPolicy && action !== "shutdown") {
      if (this.stopping || this.startupAborted) return Promise.resolve({ok:false,text:"Worker is shutting down."});
      if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) return Promise.resolve({ok:false,text:"Invalid control timeout."});
      this.startupPromise ??= this.initialize();
      return this.startupPromise.then(ready => ready.ok ? this.sendRequest(action, payload, timeoutMs) : ready);
    }
    return this.sendRequest(action, payload, timeoutMs);
  }
  private async initialize(): Promise<WorkerReply> {
    const policy = this.startupPolicy!;
    this.initializing = true;
    let result: WorkerReply = {ok:false,text:"Worker startup was cancelled."};
    try {
      result = await this.sendRequest(policy.probeAction, null, policy.timeoutMs);
      if (!result.ok) {
        const timedOut = result.text.startsWith("Control request timed out:");
        await this.endStartupChild();
        const first = `${result.text}${this.diagnostics.trim() ? `\n${this.diagnostics.trim()}` : ""}`;
        const retry = !this.startupAborted && !this.stopping && !this.spawnFailed && policy.fallbackEnv &&
          Object.entries(policy.fallbackEnv).some(([key,value]) => (this.extraEnv[key] ?? process.env[key]) !== value) &&
          (timedOut || policy.retryExitCodes.includes((this.lastExit.code ?? 0) >>> 0));
        if (retry) {
          Object.assign(this.extraEnv, policy.fallbackEnv); this.unavailable = null;
          this.emit("event", "worker-recovering", {text:"Graphics startup failed. Restarting the engine with CPU rendering."});
          result = await this.sendRequest(policy.probeAction, null, policy.timeoutMs);
          if (!result.ok) {
            await this.endStartupChild();
            result = {ok:false,text:`GPU startup failed: ${first}\nCPU startup failed: ${result.text}${this.diagnostics.trim() ? `\n${this.diagnostics.trim()}` : ""}`};
          }
        } else result = {ok:false,text:first};
      }
      if (this.startupAborted) return {ok:false,text:"Worker startup was cancelled."};
      if (result.ok) {
        this.initializing = false;
        this.emit("event", "worker-ready", JSON.parse(result.text));
      }
      else this.unavailable = result.text;
      return result;
    } catch (error) {
      this.unavailable = (error as Error).message; result = {ok:false,text:this.unavailable}; return result;
    } finally {
      this.initializing = false;
      if (!result.ok && !this.startupAborted) {
        this.emit("event", "worker-error", {text:result.text});
        this.emit("event", "worker-exit", this.lastExit);
      }
    }
  }
  // Only a read-only startup probe has been sent: no import/export or user mutation
  // can be in flight, so terminating a stuck driver here cannot interrupt a job.
  private async endStartupChild(): Promise<void> {
    const child = this.child; if (!child) return;
    await new Promise<void>((resolve,reject) => {
      const closed = () => { clearTimeout(timer); resolve(); };
      const timer = setTimeout(() => {child.off("close",closed);reject(new Error("The startup probe could not be stopped; a second worker was not started."));},2000);
      child.once("close",closed);
      child.kill();
    });
  }
  private sendRequest(action: string, payload: unknown = null, timeoutMs = 5000): Promise<WorkerReply> {
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
  restart(extraEnv: Record<string,string> = {}): void {
    if (this.child || this.initializing) throw new Error("The previous worker has not exited; wait for it to stop before restarting.");
    Object.assign(this.extraEnv, extraEnv);
    this.unavailable = null;
    this.startupPromise = null; this.startupAborted = false;
  }
  /** After startup, the backend must finalize jobs and acknowledge shutdown; never force-kill a job.
   * Only the read-only startup probe can be terminated before any user command is forwarded. */
  stop(timeoutMs = 5000): Promise<void> {
    if (this.stopping) return this.stopping;
    if (this.initializing) {
      this.startupAborted = true;
      this.stopping = this.endStartupChild().finally(() => {this.stopping = null;});
      return this.stopping;
    }
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
