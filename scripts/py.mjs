// Runs a Python script with the project's .venv when present, else the system interpreter.
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";

const venv = process.platform === "win32" ? ".venv/Scripts/python.exe" : ".venv/bin/python";
const py = existsSync(venv) ? venv : process.platform === "win32" ? "python" : "python3";
const result = spawnSync(py, process.argv.slice(2), { stdio: "inherit" });
process.exit(result.status ?? 1);
