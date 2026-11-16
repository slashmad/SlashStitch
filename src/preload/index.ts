import { contextBridge, ipcRenderer, webUtils } from "electron";

contextBridge.exposeInMainWorld("Host", {
  call: (action: string, payload: unknown = null) => ipcRenderer.invoke("host:call", action, payload),
  getPathForFile: (file: File) => webUtils.getPathForFile(file),
  onEvent: (callback: (name: string, data: unknown) => void) => {
    const listener = (_event: unknown, name: string, data: unknown) => callback(name, data);
    ipcRenderer.on("host:event", listener);
    return () => ipcRenderer.removeListener("host:event", listener);
  },
});
