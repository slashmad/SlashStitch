/// <reference types="vite/client" />
interface Window {
  Host: {
    call(action: string, payload?: unknown): Promise<{ ok: boolean; text: string }>;
    getPathForFile(file: File): string;
    onEvent(callback: (name: string, data: unknown) => void): () => void;
  };
}
