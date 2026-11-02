// Operate the owner's existing notebook UI to locate reusable training files.
// This does not run model tests or modify existing training processes/files.
const fs = require('fs');
const path = require('path');
const zlib = require('zlib');
const crypto = require('crypto');
const { execFileSync } = require('child_process');
const HOME = '/home/dev/assumeengage';
const puppeteer = require(HOME + '/tools/cache-browser/node_modules/puppeteer-core');
const chromium = require(HOME + '/tools/cache-browser/node_modules/@sparticuz/chromium');

async function main() {
  const endpoint = JSON.parse(fs.readFileSync(HOME + '/config/existing-pod-endpoint.json'));
  const base = `http://${endpoint.ip}:8888`;
  const binaries = HOME + '/tools/cache-browser/runtime';
  fs.mkdirSync(binaries, {recursive: true});
  const packageRoot = path.dirname(require.resolve(HOME + '/tools/cache-browser/node_modules/@sparticuz/chromium/package.json'));
  const bin = path.join(packageRoot, 'bin');
  const executable = path.join(binaries, 'chromium');
  if (!fs.existsSync(executable)) {
    fs.writeFileSync(executable, zlib.brotliDecompressSync(fs.readFileSync(path.join(bin, 'chromium.br'))));
    fs.chmodSync(executable, 0o755);
  }
  for (const name of ['fonts.tar.br', 'swiftshader.tar.br', fs.existsSync(path.join(bin, 'al2023.tar.br')) ? 'al2023.tar.br' : 'al2.tar.br']) {
    const source = path.join(bin, name);
    const marker = path.join(binaries, name + '.unpacked');
    if (!fs.existsSync(source) || fs.existsSync(marker)) continue;
    const archive = path.join(binaries, name.slice(0, -3));
    fs.writeFileSync(archive, zlib.brotliDecompressSync(fs.readFileSync(source)));
    execFileSync('tar', ['-xf', archive, '-C', binaries]);
    fs.unlinkSync(archive);
    fs.writeFileSync(marker, '');
  }
  const libraryDirs = new Set([binaries]);
  function libraries(dir, depth) {
    if (depth > 4) return;
    for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
      const file = path.join(dir, entry.name);
      if (entry.isDirectory()) libraries(file, depth + 1);
      else if (entry.name.includes('.so')) libraryDirs.add(dir);
    }
  }
  libraries(binaries, 0);
  const browser = await puppeteer.launch({
    executablePath: executable, headless: 'shell',
    args: [...chromium.args, '--disable-gpu', '--disable-dev-shm-usage', '--no-sandbox',
           '--unsafely-treat-insecure-origin-as-secure=' + base],
    userDataDir: HOME + '/cache/cache-browser-profile',
    env: {...process.env, LD_LIBRARY_PATH: [...libraryDirs].join(':'), TMPDIR: HOME + '/tmp'},
    defaultViewport: {width: 1440, height: 1000},
  });
  const suffix = crypto.randomUUID().replaceAll('-', '');
  const remoteFile = '/tmp/assumeengage-cache-discovery-' + suffix + '.json';
  try {
    const page = await browser.newPage();
    const sockets = [];
    const client = await page.createCDPSession();
    await client.send('Network.enable');
    client.on('Network.webSocketCreated', event => {
      const parsed = new URL(event.url);
      sockets.push(parsed.host + parsed.pathname);
    });
    await page.goto(base + '/?folder=/home/jovyan', {waitUntil: 'domcontentloaded', timeout: 60000});
    if (/\/login(?:\?|$)/.test(page.url())) throw new Error('Existing notebook requires separate authentication');
    await page.waitForSelector('.monaco-workbench .part.statusbar', {timeout: 60000});
    console.log('Notebook UI language:', await page.evaluate(() => document.documentElement.lang));
    await page.evaluate(() => document.querySelectorAll('.xterm').forEach(node => node.setAttribute('data-existing-cache-helper', 'true')));
    // Invoke the explicit New Terminal action; do not toggle or type into an existing terminal.
    await page.keyboard.down('Control');
    await page.keyboard.down('Shift');
    await page.keyboard.press('KeyP');
    await page.keyboard.up('Shift');
    await page.keyboard.up('Control');
    await page.waitForSelector('.quick-input-widget input', {visible: true, timeout: 20000});
    const language = await page.evaluate(() => document.documentElement.lang);
    await page.keyboard.type(language.startsWith('zh') ? '终端: 新建终端' : 'Terminal: Create New Terminal');
    await page.waitForFunction(() => [...document.querySelectorAll('.quick-input-widget .monaco-list-row')]
      .some(row => /Create New Terminal|新建终端/.test(row.textContent)), {timeout: 20000});
    await page.evaluate(() => [...document.querySelectorAll('.quick-input-widget .monaco-list-row')]
      .find(row => /Create New Terminal|新建终端/.test(row.textContent)).click());
    console.log('Requested a new helper terminal; management routes:', sockets);
    await page.waitForFunction(() => [...document.querySelectorAll('.xterm:not([data-existing-cache-helper])')]
      .some(node => node.getClientRects().length && node.querySelector('textarea')), {timeout: 45000});
    await page.evaluate(() => {
      const node = [...document.querySelectorAll('.xterm:not([data-existing-cache-helper])')]
        .find(node => node.getClientRects().length && node.querySelector('textarea'));
      node.setAttribute('data-owned-cache-helper', 'true');
      node.querySelector('textarea').focus();
    });
    const program = `import os,json,sys,re,subprocess
result={'python':sys.executable,'uid':os.getuid(),'cache_paths':{},'directories':[],'training_code':[]}
for key in ['HF_HOME','HF_DATASETS_CACHE','TRANSFORMERS_CACHE']:
 value=os.environ.get(key)
 if value and value.startswith('/'):result['cache_paths'][key]=value
pattern=re.compile(r'tasksource|zero.?shot|tokeniz|(^|[_-])nli([_-]|$)|pair.*cache',re.I)
for start in ['/home/jovyan','/root']:
 for parent,dirs,files in os.walk(start):
  relative=os.path.relpath(parent,start)
  depth=0 if relative=='.' else relative.count(os.sep)+1
  dirs[:]=[d for d in dirs if d not in ['.git','node_modules','envs','.venv','site-packages','__pycache__']]
  if depth>=6:dirs[:]=[]
  for name in dirs:
   if pattern.search(name):result['directories'].append(os.path.join(parent,name))
  if depth<=3:
   for name in files:
    if name.endswith('.py') and re.search('train|finetune|tokeniz|specialist',name,re.I):result['training_code'].append(os.path.join(parent,name))
try:
 result['gpus']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader'],text=True)
except Exception:pass
with open(${JSON.stringify(remoteFile)},'w') as output:json.dump(result,output)
`;
    const transport = Buffer.from(program).toString('base64');
    const command = `python -c "import base64;exec(base64.b64decode('${transport}'))"`;
    await page.keyboard.type(command, {delay: 0});
    await page.keyboard.press('Enter');
    await page.waitForFunction(async file => {
      const response = await fetch('/vscode-remote-resource?path=' + encodeURIComponent(file));
      if (!response.ok) return false;
      try { window.__cache_helper_result = await response.json(); return true; } catch { return false; }
    }, {timeout: 120000, polling: 1000}, remoteFile);
    const result = await page.evaluate(() => window.__cache_helper_result);
    fs.writeFileSync(HOME + '/config/existing-training-cache.json', JSON.stringify({...result, pod: endpoint.pod, base}, null, 2));
    console.log('Existing cache directories:', result.directories);
    console.log('Existing training code paths:', result.training_code);
    console.log('Existing GPU usage:', result.gpus || 'not available');
    await page.evaluate(() => document.querySelector('.xterm[data-owned-cache-helper] textarea').focus());
    await page.keyboard.type(`rm -f '${remoteFile}'; exit`, {delay: 0});
    await page.keyboard.press('Enter');
  } finally {
    await browser.close();
  }
}

main().catch(error => { console.error('Existing cache access stopped:', error.name, error.message); process.exitCode = 1; });
