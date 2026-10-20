// Frame-exact recordings of Prompt2Rig takes, rendered in an off-screen (headless) Chrome.
// Doesn't touch your screen: each frame is seeked, posed with the GVHMR motion and screenshotted.
//
//   node tools/capture.mjs --prompts tools/prompts.txt [--character michelle] [--out captures]
//
// Needs the web server (node server.mjs) and the HQ worker running. Writes one 2560×1440 30fps
// MP4 per prompt: <out>/<nn>_<prompt>.mp4, plus <out>/takes.json.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { spawn } from 'node:child_process';
import puppeteer from 'puppeteer-core';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true]);
  return acc;
}, []));
const W = Number(args.width || 2560), H = Number(args.height || 1440), FPS = 30;
const OUT = path.resolve(args.out || 'captures');
const URL = args.url || 'http://localhost:5177';
const CHROME = args.chrome || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
// Lines may end in "| <seconds>" to set that clip's duration (FLUX 3 Max: 5–20s).
const prompts = fs.readFileSync(args.prompts || 'tools/prompts.txt', 'utf8').split(/\r?\n/).map((l) => l.trim()).filter((l) => l && !l.startsWith('#'))
  .map((l) => { const [p, d] = l.split('|').map((x) => x.trim()); return { prompt: p, duration: d ? Number(d) : Number(args.duration || 5) }; });

// App settings for the recording: FLUX 3 Max, portrait, GVHMR only (hold until HQ), static camera.
const settings = {
  character: args.character || 'michelle', videoModel: args.model || 'flux3', orientation: 'portrait',
  stream: false, chain: false, swap: false, cinematic: false, hq: true, waitHq: true, showSkel: true,
  source: 'performer', duration: 5, resolution: '480P',
  ...(args.performer ? { performerUrls: { portrait: args.performer } } : {}),
};

const log = (...m) => console.log(new Date().toLocaleTimeString(), ...m);
const slug = (s) => s.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '').slice(0, 48);

fs.mkdirSync(OUT, { recursive: true });
const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  userDataDir: fs.mkdtempSync(path.join(os.tmpdir(), 'p2r-capture-')),
  args: [`--window-size=${W},${H}`, '--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist',
    '--autoplay-policy=no-user-gesture-required', '--hide-scrollbars', '--mute-audio'],
  defaultViewport: { width: W, height: H, deviceScaleFactor: 1 },
  protocolTimeout: 900000,
});
const results = [];
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => log('page error:', e.message));
  await page.evaluateOnNewDocument((s) => localStorage.setItem('p2r', JSON.stringify(s)), settings);
  await page.goto(URL, { waitUntil: 'networkidle2' });
  await page.waitForFunction(() => window.p2r?.rig && document.querySelector('#pill-hq').classList.contains('ok')
    && document.querySelector('#pill-pose').classList.contains('ok'), { timeout: 120000 });
  log(`app ready (${settings.character}, ${settings.videoModel}); ${prompts.length} prompts`);

  for (const [i, { prompt, duration: seconds }] of prompts.entries()) {
    const name = `${String(i + 1).padStart(2, '0')}_${slug(prompt)}`;
    log(`#${i + 1} generating (${seconds}s): ${prompt}`);
    await page.evaluate((p) => { document.querySelector('#prompt').value = p; }, prompt);
    const id = await page.evaluate((p, d) => window.p2r.submit(p, { duration: d }), prompt, seconds);
    const take = await page.waitForFunction((tid) => {
      const t = window.p2r.take(tid);
      return t && (t.status === 'error' || t.hq === 'error' || t.hq === 'done') && t;
    }, { timeout: 240000, polling: 500 }, id).then((h) => h.jsonValue());
    if (take.hq !== 'done') { log(`#${i + 1} failed (${take.status}/${take.hq}) — skipped`); continue; }

    const { duration, frames } = await page.evaluate((tid) => window.p2r.beginCapture(tid), id);
    // Highlight the prompt in the textarea, as when typing it.
    await page.evaluate(() => { const ta = document.querySelector('#prompt'); ta.focus(); ta.select(); });
    const file = path.join(OUT, `${name}.mp4`);
    const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error', '-f', 'image2pipe', '-c:v', 'mjpeg', '-r', String(FPS), '-i', '-',
      '-c:v', 'libx264', '-preset', 'medium', '-crf', '16', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', file], { stdio: ['pipe', 'inherit', 'inherit'] });
    const t0 = Date.now();
    for (let f = 0; f < frames; f++) {
      await page.evaluate((t) => window.p2r.renderAt(t), f / FPS);
      const jpg = await page.screenshot({ type: 'jpeg', quality: 95, optimizeForSpeed: true });
      if (!ff.stdin.write(jpg)) await new Promise((r) => ff.stdin.once('drain', r));
    }
    ff.stdin.end();
    await new Promise((r, j) => ff.on('close', (c) => (c === 0 ? r() : j(new Error('ffmpeg ' + c)))));
    await page.evaluate(() => window.p2r.endCapture());
    log(`#${i + 1} captured ${frames} frames (${duration.toFixed(2)}s clip) in ${((Date.now() - t0) / 1000).toFixed(0)}s → ${path.basename(file)}`);
    results.push({ n: i + 1, prompt, file: path.basename(file), video: take.url, duration });
    fs.writeFileSync(path.join(OUT, 'takes.json'), JSON.stringify(results, null, 2));
  }
} finally {
  await browser.close();
}
log(`done: ${results.length}/${prompts.length} takes in ${OUT}`);
