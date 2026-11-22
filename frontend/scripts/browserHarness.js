import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";
import { createServer } from "vite";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

export async function withBrowserPage(verify) {
  const cacheDir = await mkdtemp(path.join(tmpdir(), "companion-browser-test-"));
  let server;
  let browser;
  try {
    server = await createServer({
      root,
      cacheDir,
      logLevel: "error",
      server: { host: "127.0.0.1", port: 0, strictPort: false },
    });
    await server.listen();
    const executablePath = process.env.CHROME_PATH || process.env.PUPPETEER_EXECUTABLE_PATH;
    browser = await puppeteer.launch({
      headless: true,
      ...(executablePath ? { executablePath } : { channel: process.env.CHROME_CHANNEL || "chrome" }),
      args: process.env.CI ? ["--no-sandbox"] : [],
    });
    const page = await browser.newPage();
    page.setDefaultTimeout(10_000);
    await page.setViewport({ width: 1365, height: 900 });
    return await verify({ page, baseUrl: server.resolvedUrls.local[0] });
  } finally {
    await browser?.close();
    await server?.close();
    await rm(cacheDir, { recursive: true, force: true });
  }
}
