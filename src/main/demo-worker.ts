// Harmless built-in worker: no hardware or files. Replace with application-specific actions.
import { createInterface } from "node:readline";
const input = createInterface({ input: process.stdin });
let job: ReturnType<typeof setInterval> | null = null;
let finalizing = false;
const send = (message: unknown) => process.stdout.write(JSON.stringify(message) + "\n");
const state = (value: string) => send({ event: "job-state", data: { state: value } });
const stopJob = async () => {
  if (job) clearInterval(job);
  job = null;
  finalizing = true;
  state("finalizing");
  await new Promise((resolve) => setTimeout(resolve, 150));
  finalizing = false;
  state("idle");
};
input.on("line", async (line) => {
  let id = "";
  try {
    const msg = JSON.parse(line);
    if (!msg || typeof msg.id !== "string" || typeof msg.action !== "string") throw new Error("Invalid request");
    id = msg.id;
    switch (msg.action) {
      case "echo": send({ id, ok: true, result: msg.payload }); break;
      case "demo-start":
        if (job || finalizing) throw new Error("A job is already active");
        job = setInterval(() => send({ event: "progress", data: { text: "Demo job running; no hardware or files are used." } }), 1000);
        state("running");
        send({ id, ok: true, result: { state: "running" } });
        break;
      case "demo-stop":
        if (finalizing) throw new Error("Already finalizing");
        await stopJob();
        send({ id, ok: true, result: { state: "idle" } });
        break;
      case "shutdown":
        if (job || finalizing) throw new Error("Stop the active job before quitting");
        send({ id, ok: true, result: null });
        input.close();
        process.stdin.destroy();
        break;
      default: throw new Error("Unknown action");
    }
  } catch (error) { send({ id, ok: false, error: (error as Error).message }); }
});
input.on("close", () => { if (job) void stopJob(); });
