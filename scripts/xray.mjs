// token-xray tasks without touching package.json:
//   node scripts/xray.mjs test     run the X-ray test suite
//   node scripts/xray.mjs report   print the frozen scenarios against the installed libraries
//   node scripts/xray.mjs export   regenerate public/xray/data.json for the story
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(import.meta.dirname, "..");
const venv = process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python";
const py = existsSync(resolve(root, venv)) ? resolve(root, venv) : process.platform === "win32" ? "python" : "python3";
const tasks = {
  test: ["-m", "pytest", "-q"],
  report: ["-m", "token_xray.cli", "report"],
  export: ["-m", "token_xray.cli", "export", resolve(root, "public/xray/data.json")],
};
const task = process.argv[2];
if (!tasks[task]) {
  console.error(`usage: node scripts/xray.mjs ${Object.keys(tasks).join("|")}`);
  process.exit(2);
}
const r = spawnSync(py, tasks[task], {
  cwd: resolve(root, "xray"),
  stdio: "inherit",
  env: { ...process.env, PYTHONPATH: resolve(root, "xray/src") },
});
process.exit(r.status ?? 1);
