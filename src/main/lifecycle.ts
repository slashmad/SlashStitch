/** Both Close and Quit use this gate; jobs are stopped explicitly before shutdown. */
export class ShutdownGate {
  allowed = false;
  stopping = false;
  private pending: Promise<boolean> | null = null;
  constructor(private isBusy: () => boolean, private stop: () => Promise<void>, private onError: (error: Error) => void) {}
  request(): Promise<boolean> {
    if (this.allowed) return Promise.resolve(true);
    if (this.pending) return this.pending;
    if (this.isBusy()) return Promise.resolve(false);
    this.stopping = true;
    this.pending = Promise.resolve().then(this.stop).then(() => {
      this.allowed = true;
      return true;
    }).catch((error: Error) => {
      this.onError(error);
      return false;
    }).finally(() => { this.stopping = false; this.pending = null; });
    return this.pending;
  }
}
