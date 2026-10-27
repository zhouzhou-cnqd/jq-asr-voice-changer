// 逐帧截图脚本：加载 demo_page.html 的 autoplay 动画，按固定间隔连续截图
const path = require('path');
const fs = require('fs');

const PW_DIR = 'C:/Users/AA523/.workbuddy/binaries/node/versions/22.22.2-6/node_modules/@playwright/cli/node_modules';
const { chromium } = require(path.join(PW_DIR, 'playwright'));

const OUT_DIR = path.join(__dirname, 'gif_frames');
const TOTAL_MS = 6000;   // 动画总时长
const FPS = 12;          // 12 帧/秒
const INTERVAL = 1000 / FPS;
const WIDTH = 1240;
const HEIGHT = 880;

async function main() {
  if (!fs.existsSync(OUT_DIR)) fs.mkdirSync(OUT_DIR, { recursive: true });
  // 清空旧帧
  for (const f of fs.readdirSync(OUT_DIR)) {
    if (f.endsWith('.png')) fs.unlinkSync(path.join(OUT_DIR, f));
  }

  const browser = await chromium.launch({ channel: 'msedge' });
  const page = await browser.newPage({ viewport: { width: WIDTH, height: HEIGHT } });

  const fileUrl = 'file:///' + path.join(__dirname, 'demo_page.html').replace(/\\/g, '/') + '?autoplay=1';
  await page.goto(fileUrl);
  await page.waitForTimeout(300); // 等字体/样式就绪

  const totalFrames = Math.ceil(TOTAL_MS / INTERVAL);
  for (let i = 0; i < totalFrames; i++) {
    const elapsed = i * INTERVAL;
    await page.waitForTimeout(INTERVAL);
    const fname = 'frame_' + String(i).padStart(4, '0') + '.png';
    await page.screenshot({ path: path.join(OUT_DIR, fname) });
    process.stdout.write('\r已截图 ' + (i + 1) + '/' + totalFrames + ' (' + elapsed + 'ms)');
  }

  console.log('\n截图完成，共 ' + totalFrames + ' 帧');
  await browser.close();
}

main().catch(e => { console.error(e); process.exit(1); });
