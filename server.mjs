// Prompt2Rig — tiny zero-dependency server.
// Serves ./public and proxies fal.ai so the API key never reaches the browser.
import http from 'node:http';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PUBLIC = path.join(ROOT, 'public');

// --- .env -------------------------------------------------------------------
const envFile = path.join(ROOT, '.env');
if (fs.existsSync(envFile)) {
  for (const line of fs.readFileSync(envFile, 'utf8').split(/\r?\n/)) {
    const m = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
    if (m && !process.env[m[1]]) process.env[m[1]] = m[2];
  }
}
const FAL_KEY = process.env.FAL_KEY;
const PORT = Number(process.env.PORT || 5177);
const HQ_URL = process.env.HQ_URL || 'http://127.0.0.1:5178';
if (!FAL_KEY) console.warn('[prompt2rig] FAL_KEY is not set — generation will fail. Put it in .env');

const MODELS = {
  t2v: 'minimax/h3-max-turbo/text-to-video',
  i2v: 'minimax/h3-max-turbo/image-to-video',
  image: 'fal-ai/flux/schnell',
};

async function falRun(model, input) {
  const t0 = performance.now();
  const res = await fetch(`https://fal.run/${model}`, {
    method: 'POST',
    headers: { Authorization: `Key ${FAL_KEY}`, 'Content-Type': 'application/json' },
    body: JSON.stringify(input),
  });
  const text = await res.text();
  let json;
  try { json = JSON.parse(text); } catch { json = { raw: text }; }
  if (!res.ok) {
    const detail = json?.detail ? JSON.stringify(json.detail) : text.slice(0, 500);
    throw Object.assign(new Error(`fal ${model} → ${res.status}: ${detail}`), { status: res.status });
  }
  return { ...json, _elapsed: (performance.now() - t0) / 1000 };
}

// --- request handling ----------------------------------------------------------
function readJson(req) {
  return new Promise((resolve, reject) => {
    let size = 0; const chunks = [];
    req.on('data', (c) => { size += c.length; if (size > 25e6) reject(new Error('body too large')); else chunks.push(c); });
    req.on('end', () => { try { resolve(JSON.parse(Buffer.concat(chunks).toString() || '{}')); } catch (e) { reject(e); } });
    req.on('error', reject);
  });
}

function send(res, status, body) {
  res.writeHead(status, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(body));
}

const MIME = {
  '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css',
  '.json': 'application/json', '.glb': 'model/gltf-binary', '.png': 'image/png', '.svg': 'image/svg+xml',
};

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://localhost');
  try {
    if (req.method === 'POST' && url.pathname === '/api/video') {
      const b = await readJson(req);
      const mode = b.image_url ? 'i2v' : 't2v';
      const input = {
        prompt: String(b.prompt || '').slice(0, 4000),
        duration: Math.min(15, Math.max(1, Number(b.duration) || 5)),
        resolution: ['480P', '768P', '1080P'].includes(b.resolution) ? b.resolution : '480P',
        prompt_expansion_mode: b.prompt_expansion_mode || 'disabled',
      };
      if (b.seed != null && b.seed !== '') input.seed = Number(b.seed);
      if (mode === 'i2v') input.image_url = b.image_url;
      else input.aspect_ratio = b.aspect_ratio || '16:9';
      const out = await falRun(MODELS[mode], input);
      console.log(`[video] ${mode} ${input.duration}s ${input.resolution} in ${out._elapsed.toFixed(1)}s (gpu ${out.timings?.inference?.toFixed?.(2) ?? '?'}s)`);
      return send(res, 200, { mode, video_url: out.video?.url, timings: out.timings, elapsed: out._elapsed, expanded_prompt: out.expanded_prompt });
    }

    if (req.method === 'POST' && url.pathname === '/api/image') {
      const b = await readJson(req);
      const out = await falRun(MODELS.image, {
        prompt: String(b.prompt || '').slice(0, 4000),
        image_size: b.image_size || 'landscape_16_9',
        num_inference_steps: 4,
        enable_safety_checker: true,
      });
      console.log(`[image] in ${out._elapsed.toFixed(1)}s`);
      return send(res, 200, { image_url: out.images?.[0]?.url, elapsed: out._elapsed });
    }

    // HQ pass: forward to the local GVHMR worker (hq/worker.py).
    if (url.pathname === '/api/hq/health') {
      try {
        const r = await fetch(`${HQ_URL}/health`, { signal: AbortSignal.timeout(2000) });
        return send(res, 200, await r.json());
      } catch { return send(res, 200, { ok: false, error: 'worker offline' }); }
    }
    if (req.method === 'POST' && url.pathname === '/api/hq') {
      const b = await readJson(req);
      const r = await fetch(`${HQ_URL}/process`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_url: b.video_url }),
      });
      const text = await r.text();
      res.writeHead(r.status, { 'Content-Type': 'application/json' });
      return res.end(text);
    }

    if (req.method === 'GET') {
      let p = decodeURIComponent(url.pathname);
      if (p === '/') p = '/index.html';
      const file = path.normalize(path.join(PUBLIC, p));
      if (!file.startsWith(PUBLIC) || !fs.existsSync(file) || fs.statSync(file).isDirectory()) {
        res.writeHead(404); return res.end('not found');
      }
      res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream', 'Cache-Control': 'no-cache' });
      return fs.createReadStream(file).pipe(res);
    }

    res.writeHead(405); res.end();
  } catch (err) {
    console.error('[error]', err.message);
    send(res, err.status && err.status < 600 ? err.status : 500, { error: err.message });
  }
});

server.listen(PORT, () => console.log(`Prompt2Rig → http://localhost:${PORT}`));
