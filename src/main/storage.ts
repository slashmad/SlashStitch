import { closeSync, existsSync, fsyncSync, lstatSync, openSync, realpathSync, renameSync, rmSync, statSync, writeFileSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { basename, dirname, join, resolve } from "node:path";

export function protectSources(destination: string, sources: string[]): void {
  const target = join(realpathSync(dirname(resolve(destination))), basename(destination));
  const metadata = existsSync(destination) ? statSync(destination, { bigint: true }) : null;
  for (const source of sources) {
    const canonical = existsSync(source) ? realpathSync(source) : resolve(source);
    const sourceMetadata = metadata && existsSync(source) ? statSync(source, { bigint: true }) : null;
    if (canonical === target || (metadata && sourceMetadata && metadata.dev === sourceMetadata.dev && metadata.ino === sourceMetadata.ino)) throw new Error("The destination is a source image or a link to one.");
  }
  if (existsSync(destination) && lstatSync(destination).isSymbolicLink()) throw new Error("The destination must not be a symbolic link.");
}

export function writeJsonAtomic(destination: string, value: unknown): void {
  const temp = join(dirname(destination), `.slashstitch-${randomUUID()}.part`);
  let fd: number | null = null;
  try {
    fd = openSync(temp, "wx", 0o600);
    writeFileSync(fd, JSON.stringify(value, null, 2) + "\n");
    fsyncSync(fd); closeSync(fd); fd = null;
    renameSync(temp, destination);
  } finally {
    if (fd !== null) closeSync(fd);
    rmSync(temp, { force: true });
  }
}
