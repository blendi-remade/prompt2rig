// Render close-ups of the character's hands at given times, for side-by-side review with the source.
//   node bench/hand_review.mjs --url <clip url> --name h3 --times 0.8,2.1,3.3,4.6 [--character xbot] [--hands mediapipe|wilor|off]
// Writes bench/out/review/<name>_<hands>_<t>_<side>.png (character's own Right/Left hand).
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import puppeteer from 'puppeteer-core';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1] && !all[i + 1].startsWith('--') ? all[i + 1] : true]);
  return acc;
}, []));
const OUT = path.resolve('bench/out/review');
const times = String(args.times || '1,2,3,4').split(',').map(Number);
const hands = args.hands || 'mediapipe';
fs.mkdirSync(OUT, { recursive: true });

const browser = await puppeteer.launch({
  executablePath: 'C:/Program Files/Google/Chrome/Application/chrome.exe', headless: true,
  userDataDir: fs.mkdtempSync(path.join(os.tmpdir(), 'p2r-review-')),
  args: ['--use-angle=d3d11', '--enable-gpu', '--ignore-gpu-blocklist', '--mute-audio', '--window-size=1100,900'],
  defaultViewport: { width: 1100, height: 900, deviceScaleFactor: 1 },
});
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.log('page error:', e.message));
  await page.evaluateOnNewDocument((s) => localStorage.setItem('p2r', JSON.stringify(s)),
    { character: args.character || 'xbot', hq: true, waitHq: true, swap: false, cinematic: false, stream: false, hands: hands !== 'off' });
  await page.goto('http://localhost:5177', { waitUntil: 'networkidle2' });
  await page.waitForFunction(() => window.p2r?.rig && document.querySelector('#pill-hq').classList.contains('ok'), { timeout: 120000 });
  const id = await page.evaluate((u, h) => window.p2r.playUrl(u, true, h).id, args.url, hands);
  await page.waitForFunction((tid) => ['done', 'error'].includes(window.p2r.take(tid)?.hq), { timeout: 180000, polling: 300 }, id);
  await page.evaluate((tid) => window.p2r.beginCapture(tid), id);
  await page.addStyleTag({ content: 'aside, footer, .vidwrap, .hud, .banner { display: none !important } main { position: fixed !important; inset: 0 !important }' });
  await new Promise((r) => setTimeout(r, 500));
  for (const t of times) {
    for (const side of ['Right', 'Left']) {
      await page.evaluate((tt) => window.p2r.renderAt(tt), t);
      await page.evaluate((s, d) => window.p2r.focusBone(`${s}Hand`, d, [0.25, 0.1, 1]), side, Number(args.dist || 0.75));
      await page.evaluate((tt) => window.p2r.renderAt(tt + 1e-3), t);
      await page.screenshot({ path: path.join(OUT, `${args.name}_${hands}_${t.toFixed(2)}_${side}.png`) });
    }
  }
  console.log('ok', times.length * 2, 'close-ups');
} finally {
  await browser.close();
}
