const fs = require('fs');
const path = require('path');
const { marked } = require('/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/marked');
const sharp = require('/Users/apple/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp');
const dir = __dirname;
const assets = path.join(dir, 'assets');
const palette = {ink:'#263b41',muted:'#627278',line:'#c6d2d3',paper:'#f7f9f8',green:'#267567',greenbg:'#eaf5ee',blue:'#386b95',bluebg:'#eaf1fa',amber:'#9d681d',amberbg:'#faf1df',pad:'#e5eaeb'};
const esc = s => String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function text(x,y,s,size=23,fill=palette.ink,anchor='start',weight=400){return `<text x="${x}" y="${y}" fill="${fill}" font-size="${size}" text-anchor="${anchor}" font-weight="${weight}">${esc(s)}</text>`;}
function rect(x,y,w,h,fill,r=9,stroke='none'){return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="${r}" fill="${fill}" stroke="${stroke}"/>`;}
function line(x1,y1,x2,y2,stroke=palette.line,arrow=false){return `<path d="M ${x1} ${y1} L ${x2} ${y2}" fill="none" stroke="${stroke}" stroke-width="2"${arrow?' marker-end="url(#arrow)"':''}/>`;}
function svg(title,w,h,body){return `<svg xmlns="http://www.w3.org/2000/svg" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(title)}"><title>${esc(title)}</title><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 1 L6 4 L0 7" fill="none" stroke="${palette.muted}" stroke-width="1.4"/></marker><pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="${palette.pad}"/><path d="M-2 2 L2 -2 M0 8 L8 0 M6 10 L10 6" stroke="#c9d3d5" stroke-width="1"/></pattern></defs><rect width="${w}" height="${h}" fill="#ffffff"/><g font-family="-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif">${body}</g></svg>`;}
let b = text(24,38,'这些知识，在训练体系的什么位置？',26,palette.ink,'start',600);
b += text(24,72,'围绕一次已有模型的训练，展开三个相连的部分',19,palette.muted);
const cols=[{x:24,color:palette.green,bg:palette.greenbg,title:'怎样学',scope:'第一篇 · 学习安排',lines:['前向 → 损失 → 反向 → 更新','batch：一次参考多少条','学习率：每次调整多大']},{x:364,color:palette.blue,bg:palette.bluebg,title:'怎样执行',scope:'两篇相接 · GPU 效率',lines:['microbatch / 累积 / 检查点','分桶 → 填充 → packing','Flash / varlen / 动态微批']},{x:704,color:palette.amber,bg:palette.amberbg,title:'学得怎样',scope:'第一篇 · 评测反馈',lines:['Native：想提升的整体能力','D0：原有任务保持情况','accuracy + loss：一起看']}];
for(const c of cols){ b+=rect(c.x,105,296,210,c.bg);b+=text(c.x+20,143,c.title,27,c.color,'start',600);b+=text(c.x+20,174,c.scope,18,c.color);c.lines.forEach((s,i)=>b+=text(c.x+20,219+i*32,s,19));}
b+=line(325,205,353,205,palette.muted,true)+line(665,205,693,205,palette.muted,true);
b+=`<path d="M 852 327 V 352 H 172 V 328" fill="none" stroke="${palette.muted}" stroke-width="2" marker-end="url(#arrow)"/>`;
b+=rect(360,335,306,34,'#ffffff',0)+text(513,358,'结果再反馈给下一轮学习安排',18,palette.muted,'middle');
const diagrams={ 'training-map':svg('学习安排经过 GPU 执行，由评测结果反馈下一轮训练',1024,380,b)};

b=text(24,38,'同一次更新：128 条样本，分四趟处理',26,palette.ink,'start',600);
b+=text(24,72,'四趟都使用同一组参数；每趟做完前向和反向，先留下梯度',20,palette.muted);
for(let i=0;i<4;i++){let x=24+i*214;b+=rect(x,110,190,100,palette.bluebg);b+=text(x+95,145,`第 ${i+1} 趟`,20,palette.blue,'middle');b+=text(x+95,181,'32 条样本',25,palette.ink,'middle',600);b+=line(x+95,218,x+95,246,palette.muted,true);}
b+=rect(24,256,832,58,palette.greenbg)+text(440,293,'梯度累积：汇总这 128 条样本提供的调整信息',23,palette.green,'middle');
b+=line(440,323,440,348,palette.muted,true)+rect(291,359,298,56,palette.amberbg)+text(440,395,'参数统一更新一次',25,palette.amber,'middle',600);
b+=text(440,452,'microbatch 管每趟装多少；有效 batch 管一次更新参考多少。',20,palette.muted,'middle');
diagrams['batch-accumulation']=svg('四个32条微批累积梯度，最后只更新一次参数',880,475,b);

b=text(24,38,'一条长文本，把整批的行宽都拉长了',26,palette.ink,'start',600);
b+=text(24,73,'示例：100、200、1000 个 tokens；斜纹部分是补出来的空位',20,palette.muted);
for(let i=0;i<3;i++){let len=[100,200,1000][i], y=110+i*73; b+=text(24,y+29,['A · 100','B · 200','C · 1000'][i],20);b+=rect(160,y,700,44,'url(#hatch)',0);b+=rect(160,y,len/1000*700,44,[palette.green,palette.blue,palette.amber][i],0);}
b+=text(160,359,'三条真实文本共 1300 个 tokens，补齐后占 3000 个位置。',22,palette.ink);
b+=text(160,393,'分桶让长度相近的文本一起排队，减少这些空位。',20,palette.muted);
diagrams.padding=svg('长度100、200和1000的三条文本，补齐后每行占1000个位置',900,420,b);

b=text(24,38,'紧凑地存放，仍然是三道独立的题',26,palette.ink,'start',600);
b+=text(24,74,'示例：A 有 3 个词元，B 有 2 个，C 有 4 个',20,palette.muted);
const px=125,unit=88, colors=[palette.green,palette.blue,palette.amber],light=[palette.greenbg,palette.bluebg,palette.amberbg];
for(let i=0;i<3;i++){let y=106+i*52,n=[3,2,4][i];b+=text(24,y+28,['A','B','C'][i],23,colors[i],'start',600);for(let j=0;j<4;j++){b+=rect(px+j*unit,y,unit-4,36,j<n?light[i]:'url(#hatch)',3);if(j<n)b+=text(px+j*unit+(unit-4)/2,y+25,`${['A','B','C'][i]}${j+1}`,19,colors[i],'middle');}}
b+=text(540,166,'补齐后的存放',22)+text(540,201,'12 个位置，其中 9 个有内容',20,palette.muted);
b+=line(478,272,478,302,palette.muted,true);
let cur=0;const x0=36,u=98;for(let i=0;i<3;i++){let n=[3,2,4][i];for(let j=0;j<n;j++){b+=rect(x0+cur*u,326,u-4,43,light[i],3);b+=text(x0+cur*u+(u-4)/2,355,`${['A','B','C'][i]}${j+1}`,20,colors[i],'middle');cur++;}}
for(const n of [0,3,5,9]){b+=line(x0+n*u-2,316,x0+n*u-2,381,palette.muted);b+=text(x0+n*u-2,406,String(n),19,palette.muted,'middle');}
for(let i=0,start=0;i<3;i++){let n=[3,2,4][i],cx=x0+(start+n/2)*u-2;b+=line(cx,422,cx,447,palette.muted,true)+text(cx,478,`${['A','B','C'][i]} 的分类结果`,21,colors[i],'middle',600);start+=n;}
b+=text(478,528,'packing 管紧凑存放；varlen 按边界让 A、B、C 各算各的。',21,palette.ink,'middle');
diagrams.packing=svg('三条样本从12个位置紧凑为9个，边界为0、3、5、9，各产生一个分类结果',960,550,b);

const css=`
:root{color-scheme:light;--ink:#263b41;--muted:#617278;--green:#267567;--line:#dbe3df;--paper:#fffefa;--bg:#f1f3ee}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif;-webkit-font-smoothing:antialiased}main{max-width:1020px;margin:28px auto 64px;background:var(--paper);padding:44px 64px 58px}header{border-bottom:1px solid var(--line);padding-bottom:26px;margin-bottom:42px}.eyebrow{font-size:14px;color:var(--green);letter-spacing:.08em;margin:0 0 10px}header h1{font-size:34px;line-height:1.4;margin:0 0 14px;font-weight:650;letter-spacing:.02em}.lede{font-size:17px;color:var(--muted);line-height:1.9;margin:0}nav{display:flex;gap:12px 24px;flex-wrap:wrap;margin-top:18px}a{color:var(--green);text-decoration-thickness:1px;text-underline-offset:4px}nav a{font-size:15px}article{max-width:800px;margin:0 auto}article+article{margin-top:80px;padding-top:52px;border-top:2px solid var(--line)}article h1{font-size:30px;line-height:1.5;font-weight:650;letter-spacing:.01em;margin:0 0 24px}h2{font-size:23px;line-height:1.65;font-weight:620;margin:46px 0 18px}p{font-size:18px;line-height:2.05;margin:0 0 22px;overflow-wrap:break-word}strong{font-weight:630}figure{margin:28px -14px 32px;padding:0}figure svg{display:block;width:100%;height:auto;background:#fff}table{border-collapse:collapse;width:100%;font-size:17px;line-height:1.7;margin:28px 0 32px}th,td{padding:12px 16px;text-align:left;border-bottom:1px solid var(--line)}th{font-weight:600;color:var(--muted);font-size:15px}tbody tr:nth-child(2){background:#edf5ee}code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.87em;background:#edf1ee;padding:2px 5px;border-radius:4px}footer{max-width:800px;margin:60px auto 0;padding-top:18px;border-top:1px solid var(--line);font-size:13px;line-height:1.8;color:var(--muted)}
@media(max-width:700px){body{background:var(--paper)}main{padding:28px 22px 42px;margin:0}header{margin-bottom:32px}header h1{font-size:27px}article h1{font-size:25px}h2{font-size:21px;margin-top:36px}p{font-size:17px;line-height:2}figure{margin-left:-10px;margin-right:-10px}th,td{padding:10px 6px;font-size:15px}th{font-size:13px}article+article{margin-top:56px;padding-top:38px}}
@media print{body{background:#fff}main{margin:0;padding:0;max-width:none}header nav{display:none}p{font-size:11pt;line-height:1.9}h1,h2{break-after:avoid}figure,table{break-inside:avoid}article+article{break-before:page;border:0}footer{display:none}}
`;
async function main(){
 for(const [name,source] of Object.entries(diagrams)){
   fs.writeFileSync(path.join(assets,`${name}.svg`),source);
   await sharp(Buffer.from(source)).resize({width:1800}).png().toFile(path.join(assets,`${name}.png`));
 }
 const names=['01-模型怎样学会任务.md','02-同样的训练怎样跑得更高效.md'];
 const articles=names.map((name,i)=>{
   let md=fs.readFileSync(path.join(dir,name),'utf8');
   md=md.replace(/assets\/([a-z-]+)\.svg/g,'assets/$1.png');
   fs.writeFileSync(path.join(dir,name),md);
   let html=marked.parse(md);
   html=html.replace(/<p><img src="[^"]*\/([a-z-]+)\.png" alt="([^"]*)"\s*\/?><\/p>/g,(_,n)=>i===1&&n==='training-map'?'':`<figure>${diagrams[n]}</figure>`);
   return `<article id="article-${i+1}" aria-label="${i===0?'模型怎样学会任务':'同样的训练怎样跑得更高效'}">${html}</article>`;
 });
 const page=`<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>把训练知识连成体系｜两篇重写</title><style>${css}</style></head><body><main><header><p class="eyebrow">训练知识 · 连起来读</p><h1>先看模型怎样学，再看训练怎样跑</h1><p class="lede">两篇文章沿着同一场训练展开。知识的位置先放进全景图，再用完整的解释、少量例子和图示串起来。</p><nav aria-label="两篇文章"><a href="#article-1">第一篇：模型怎样学会任务</a><a href="#article-2">第二篇：同样的训练怎样跑得更高效</a></nav></header>${articles.join('\n')}<footer>根据你提供的两份文本重写。实验数字沿用原文；用于解释的示例在正文中说明。</footer></main></body></html>`;
 fs.writeFileSync(path.join(dir,'阅读版.html'),page);
 console.log(JSON.stringify({articles:names,diagrams:Object.keys(diagrams),html:'阅读版.html'}));
}
main().catch(e=>{console.error(e);process.exitCode=1});
