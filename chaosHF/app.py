import os
import sys
import threading
import queue
import subprocess
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

app = FastAPI(title="ML Pipeline Debugger")
is_running = False

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ML Pipeline Debugger</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#f5f5f4;color:#1c1917;min-height:100vh;padding:24px}
h1{font-size:20px;font-weight:600;margin-bottom:4px}
.subtitle{font-size:13px;color:#78716c;margin-bottom:20px}
.topbar{display:flex;align-items:center;gap:12px;margin-bottom:16px;flex-wrap:wrap}
button{background:#18181b;color:#fff;border:none;padding:9px 20px;border-radius:8px;font-size:13px;font-weight:500;cursor:pointer}
button:disabled{opacity:.4;cursor:not-allowed}
#clearBtn{background:#fff;color:#18181b;border:1px solid #e5e5e5}
#status{font-size:13px;color:#78716c;display:flex;align-items:center;gap:6px}
.dot{width:8px;height:8px;border-radius:50%;background:#a3a3a3;display:inline-block}
.dot.running{background:#f59e0b;animation:pulse 1s infinite}
.dot.done{background:#22c55e}
.dot.error{background:#ef4444}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.summary-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:20px}
.metric-card{background:#fff;border:1px solid #e5e5e5;border-radius:10px;padding:14px 16px}
.metric-label{font-size:11px;color:#78716c;text-transform:uppercase;letter-spacing:.05em;margin-bottom:4px}
.metric-value{font-size:22px;font-weight:600}
.green{color:#16a34a}.amber{color:#d97706}.red{color:#dc2626}
.episodes{display:flex;flex-direction:column;gap:14px}
.episode-card{background:#fff;border:1px solid #e5e5e5;border-radius:12px;overflow:hidden}
.ep-header{display:flex;align-items:center;gap:10px;padding:14px 18px;border-bottom:1px solid #f0f0ef;cursor:pointer;user-select:none}
.ep-badge{font-size:11px;font-weight:600;padding:3px 10px;border-radius:20px;text-transform:uppercase;letter-spacing:.04em}
.badge-easy{background:#dcfce7;color:#15803d}
.badge-medium{background:#fef9c3;color:#a16207}
.badge-hard{background:#fee2e2;color:#b91c1c}
.ep-title{font-size:14px;font-weight:500;flex:1}
.ep-score{font-size:13px;font-weight:600}
.chevron{font-size:11px;color:#a3a3a3;transition:transform .2s;margin-left:4px}
.chevron.open{transform:rotate(180deg)}
.ep-body{padding:16px 18px;display:none}
.ep-body.open{display:block}
.ep-meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:10px;margin-bottom:14px}
.meta-pill{background:#f5f5f4;border-radius:8px;padding:10px 12px}
.meta-pill .label{font-size:11px;color:#78716c;margin-bottom:2px}
.meta-pill .val{font-size:14px;font-weight:500}
.task-desc{font-size:12px;color:#57534e;background:#f5f5f4;border-radius:8px;padding:10px 12px;margin-bottom:14px;line-height:1.5}
.steps-title{font-size:11px;font-weight:600;color:#78716c;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px}
.step-row{display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-bottom:1px solid #f5f5f4}
.step-row:last-child{border-bottom:none}
.step-num{width:22px;height:22px;border-radius:50%;background:#f0f0ef;font-size:11px;font-weight:600;color:#78716c;display:flex;align-items:center;justify-content:center;flex-shrink:0;margin-top:1px}
.step-content{flex:1}
.action-chip{display:inline-block;font-size:11px;font-weight:500;padding:2px 8px;border-radius:4px;margin-bottom:3px}
.chip-fix_dependency{background:#ede9fe;color:#6d28d9}
.chip-train_model{background:#dbeafe;color:#1d4ed8}
.chip-evaluate{background:#d1fae5;color:#065f46}
.chip-preprocess_data{background:#fef3c7;color:#92400e}
.chip-split_data{background:#fce7f3;color:#9d174d}
.chip-load_model{background:#e0f2fe;color:#075985}
.chip-inspect_logs{background:#f5f5f4;color:#57534e}
.chip-done{background:#f0fdf4;color:#166534}
.chip-unknown{background:#f5f5f4;color:#57534e}
.step-reward{font-size:12px;color:#78716c;margin-top:1px}
.step-reward b{color:#1c1917;font-weight:600}
.step-error{font-size:11px;color:#dc2626;margin-top:2px}
.outcome-row{margin-top:12px;padding:10px 14px;border-radius:8px;font-size:13px;font-weight:500}
.outcome-success{background:#f0fdf4;color:#15803d;border:1px solid #bbf7d0}
.outcome-fail{background:#fff1f2;color:#be123c;border:1px solid #fecdd3}
.hint{text-align:center;padding:40px;color:#a3a3a3;font-size:14px}
.log-section{margin-top:24px}
.log-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:8px}
.log-header-title{font-size:11px;font-weight:600;color:#78716c;text-transform:uppercase;letter-spacing:.05em}
.log-box{background:#18181b;border-radius:10px;padding:14px 16px;height:220px;overflow-y:auto;font-family:monospace;font-size:12px;line-height:1.6;color:#d4d4d4}
.log-box .l-start{color:#60a5fa;font-weight:600}
.log-box .l-step{color:#d4d4d4}
.log-box .l-end{color:#4ade80;font-weight:600}
.log-box .l-warn{color:#fbbf24}
.log-box .l-error{color:#f87171}
.log-box .l-baseline{color:#4ade80;font-weight:600}
.log-box .l-process{color:#71717a;font-style:italic}
.log-box .l-other{color:#a3a3a3}
#logClearBtn{font-size:11px;background:#fff;color:#57534e;border:1px solid #e5e5e5;padding:3px 10px;border-radius:6px;cursor:pointer}
.logs-section{margin-top:24px;background:#fff;border:1px solid #e5e5e5;border-radius:12px;overflow:hidden}
.logs-header{display:flex;align-items:center;gap:10px;padding:12px 18px;cursor:pointer;user-select:none;border-bottom:1px solid #f0f0ef}
.logs-header span{font-size:13px;font-weight:500;flex:1}
.logs-header .log-count{font-size:11px;color:#a3a3a3;background:#f5f5f4;padding:2px 8px;border-radius:20px}
.logs-body{display:none;padding:0}
.logs-body.open{display:block}
.log-console{background:#18181b;color:#d4d4d8;font-family:monospace;font-size:12px;line-height:1.7;padding:14px 16px;max-height:340px;overflow-y:auto;white-space:pre-wrap;word-break:break-all}
.lc-start{color:#60a5fa;font-weight:600}
.lc-step{color:#d4d4d8}
.lc-end{color:#4ade80;font-weight:600}
.lc-error{color:#f87171}
.lc-warn{color:#fbbf24}
.lc-meta{color:#71717a;font-style:italic}
.lc-baseline{color:#4ade80;font-weight:600}
</style>
</head>
<body>
<h1>ML Pipeline Debugger</h1>
<div class="subtitle">Runs three pipeline tasks (easy, medium, hard) and shows what happened in each episode.</div>
<div class="topbar">
  <button id="runBtn" onclick="startRun()">Run inference</button>
  <button id="clearBtn" onclick="clearAll()">Clear</button>
  <div id="status"><span class="dot" id="dot"></span><span id="statusText">Ready</span></div>
</div>
<div class="summary-row" id="summaryRow" style="display:none">
  <div class="metric-card"><div class="metric-label">Tasks run</div><div class="metric-value" id="mTotal">0</div></div>
  <div class="metric-card"><div class="metric-label">Successful</div><div class="metric-value green" id="mSuccess">0</div></div>
  <div class="metric-card"><div class="metric-label">Avg score</div><div class="metric-value" id="mAvg">-</div></div>
  <div class="metric-card"><div class="metric-label">Total steps</div><div class="metric-value" id="mSteps">0</div></div>
</div>
<div class="episodes" id="episodes">
  <div class="hint" id="hint">Click "Run inference" to start all three pipeline tasks.</div>
</div>
<div class="log-section">
  <div class="log-header">
    <span class="log-header-title">Raw logs</span>
    <button id="logClearBtn" onclick="clearLogs()">Clear logs</button>
  </div>
  <div class="log-box" id="logBox"><span style="color:#71717a;font-style:italic">Logs will appear here when inference runs...</span></div>
</div>
<script>
const ACTION_LABELS={fix_dependency:'Fix dependency',train_model:'Train model',evaluate:'Evaluate',preprocess_data:'Preprocess data',split_data:'Split data',load_model:'Load model',inspect_logs:'Inspect logs',done:'Done'};
const TASK_DESC={easy:'Fix a transformers version mismatch, train the model, then evaluate.',medium:'Detect and fix a silent tokenisation mismatch, then evaluate.',hard:'Fix temporal data leakage in the split, retrain the model, then evaluate.'};
let episodes={},currentTask=null,totalSteps=0,es=null;

function logClass(line){
  if(line.startsWith('[START]')) return 'l-start';
  if(line.startsWith('[STEP]')) return 'l-step';
  if(line.startsWith('[END]')) return 'l-end';
  if(line.startsWith('[WARN]')) return 'l-warn';
  if(line.startsWith('[ERROR]')) return 'l-error';
  if(line.includes('# Baseline')) return 'l-baseline';
  if(line.startsWith('[PROCESS]')) return 'l-process';
  return 'l-other';
}
function appendLog(line){
  const box=document.getElementById('logBox');
  const first=box.querySelector('span[style]');if(first)first.remove();
  const d=document.createElement('div');d.className=logClass(line);
  d.textContent=line;box.appendChild(d);
  box.scrollTop=box.scrollHeight;
}
function clearLogs(){document.getElementById('logBox').innerHTML='';}
function parseLine(raw){appendLog(raw);
  let m=raw.match(/\[START\]\s+task=(\w+)/);
  if(m){currentTask=m[1];if(!episodes[currentTask])episodes[currentTask]={steps:[],score:null,success:false,step_count:0};renderEpisode(currentTask);return;}
  m=raw.match(/\[STEP\] step=(\d+) action=(\{.+\}) reward=([\d.]+) done=(\w+) error=(.*)/);
  if(m&&currentTask){
    let at='unknown';try{at=JSON.parse(m[2]).action_type||'unknown';}catch(e){}
    const reward=parseFloat(m[3]),error=m[5].replace(/^"|"$/g,'').trim();
    episodes[currentTask].steps.push({step:parseInt(m[1]),action_type:at,reward,error:error==='null'?'':error});
    episodes[currentTask].step_count=parseInt(m[1]);totalSteps++;
    renderEpisode(currentTask);updateSummary();return;
  }
  m=raw.match(/\[END\]\s+success=(\w+)\s+steps=(\d+)\s+score=([\d.]+)/);
  if(m&&currentTask){episodes[currentTask].success=m[1]==='true';episodes[currentTask].score=parseFloat(m[3]);renderEpisode(currentTask);updateSummary();return;}
  m=raw.match(/# Baseline: avg=([\d.]+)/);
  if(m){const v=(parseFloat(m[1])*100).toFixed(0)+'%';document.getElementById('mAvg').textContent=v;}
}

function scoreClass(s){return s>=0.75?'green':s>=0.4?'amber':'red';}

function renderEpisode(task){
  const ep=episodes[task];
  let card=document.getElementById('ep-'+task);
  const wasOpen=card?card.querySelector('.ep-body.open')!==null:true;
  if(!card){card=document.createElement('div');card.className='episode-card';card.id='ep-'+task;const h=document.getElementById('hint');if(h)h.remove();document.getElementById('episodes').appendChild(card);}
  const score=ep.score;
  const scoreStr=score!==null?(score*100).toFixed(0)+'%':'...';
  let stepsHtml='';
  ep.steps.forEach(s=>{
    const chip=`<span class="action-chip chip-${s.action_type}">${ACTION_LABELS[s.action_type]||s.action_type}</span>`;
    const errHtml=s.error?`<div class="step-error">&#9888; ${s.error}</div>`:'';
    stepsHtml+=`<div class="step-row"><div class="step-num">${s.step}</div><div class="step-content">${chip}<div class="step-reward">Reward: <b>${(s.reward*100).toFixed(0)}%</b></div>${errHtml}</div></div>`;
  });
  const outcome=score!==null?(ep.success?`<div class="outcome-row outcome-success">Episode completed successfully &mdash; final score ${scoreStr}</div>`:`<div class="outcome-row outcome-fail">Episode ended without success &mdash; score ${scoreStr}</div>`):'';
  const sc=score!==null?scoreClass(score):'';
  card.innerHTML=`
<div class="ep-header" onclick="toggleEp('${task}')">
  <span class="ep-badge badge-${task}">${task}</span>
  <span class="ep-title">Pipeline task</span>
  <span class="ep-score ${sc}">${scoreStr}</span>
  <span class="chevron ${wasOpen?'open':''}">&#9660;</span>
</div>
<div class="ep-body ${wasOpen?'open':''}">
  <div class="ep-meta">
    <div class="meta-pill"><div class="label">Steps taken</div><div class="val">${ep.step_count||'...'}</div></div>
    <div class="meta-pill"><div class="label">Final score</div><div class="val ${sc}">${scoreStr}</div></div>
    <div class="meta-pill"><div class="label">Outcome</div><div class="val">${ep.success?'Success':score!==null?'Failed':'Running...'}</div></div>
  </div>
  <div class="task-desc">${TASK_DESC[task]||''}</div>
  <div class="steps-title">Steps</div>
  ${stepsHtml||'<div style="font-size:13px;color:#a3a3a3;padding:6px 0">Waiting for first step...</div>'}
  ${outcome}
</div>`;
}

function toggleLogs(){const b=document.getElementById('logsBody');const c=document.getElementById('logsChevron');b.classList.toggle('open');c.classList.toggle('open');}

let logLineCount=0;

function toggleEp(task){const card=document.getElementById('ep-'+task);card.querySelector('.ep-body').classList.toggle('open');card.querySelector('.chevron').classList.toggle('open');}

function updateSummary(){
  const tasks=Object.keys(episodes);
  document.getElementById('summaryRow').style.display='';
  document.getElementById('mTotal').textContent=tasks.length;
  document.getElementById('mSuccess').textContent=tasks.filter(t=>episodes[t].success).length;
  document.getElementById('mSteps').textContent=totalSteps;
  const scores=tasks.filter(t=>episodes[t].score!==null).map(t=>episodes[t].score);
  if(scores.length)document.getElementById('mAvg').textContent=(scores.reduce((a,b)=>a+b,0)/scores.length*100).toFixed(0)+'%';
}

function setStatus(txt,state){document.getElementById('statusText').textContent=txt;document.getElementById('dot').className='dot '+(state||'');}

function clearAll(){document.getElementById('logBox').innerHTML='';clearAll_inner();}function clearAll_inner(){
  logLineCount=0;
  document.getElementById('logConsole').innerHTML='<span style="color:#71717a;font-style:italic">Logs will appear here when inference runs...</span>';
  document.getElementById('logCount').textContent='0 lines';
  episodes={};currentTask=null;totalSteps=0;
  document.getElementById('episodes').innerHTML='<div class="hint" id="hint">Click "Run inference" to start all three pipeline tasks.</div>';
  document.getElementById('summaryRow').style.display='none';
  document.getElementById('mAvg').textContent='-';
  setStatus('Ready','');
}

function startRun(){
  if(es){es.close();es=null;}
  clearAll();
  document.getElementById('runBtn').disabled=true;
  setStatus('Running...','running');
  es=new EventSource('/stream');
  es.onmessage=function(e){
    if(e.data==='__DONE__'){es.close();document.getElementById('runBtn').disabled=false;setStatus('Finished','done');return;}
    parseLine(e.data);
  };
  es.onerror=function(){es.close();document.getElementById('runBtn').disabled=false;setStatus('Connection lost','error');};
}
</script>
<div class="logs-section">
  <div class="logs-header" onclick="toggleLogs()">
    <span>Raw logs</span>
    <span class="log-count" id="logCount">0 lines</span>
    <span class="chevron" id="logsChevron">&#9660;</span>
  </div>
  <div class="logs-body" id="logsBody">
    <div class="log-console" id="logConsole"><span style="color:#71717a;font-style:italic">Logs will appear here when inference runs...</span></div>
  </div>
</div>
</body>
</html>"""


def run_inference(q: queue.Queue):
    global is_running
    is_running = True
    process = subprocess.Popen(
        [sys.executable, "-u", "inference.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env={**os.environ, "PYTHONUNBUFFERED": "1"}
    )
    def drain(pipe):
        for line in pipe:
            line = line.rstrip()
            if line:
                q.put(line)
    t_err = threading.Thread(target=drain, args=(process.stderr,), daemon=True)
    t_err.start()
    drain(process.stdout)
    t_err.join()
    process.wait()
    q.put(f"[PROCESS] Exited with code {process.returncode}")
    q.put("__DONE__")
    is_running = False


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/stream")
def stream():
    global is_running
    if is_running:
        def busy():
            yield "data: [WARN] Already running\n\ndata: __DONE__\n\n"
        return StreamingResponse(busy(), media_type="text/event-stream")
    q = queue.Queue()
    threading.Thread(target=run_inference, args=(q,), daemon=True).start()
    def event_stream():
        while True:
            try:
                line = q.get(timeout=300)
                yield f"data: {line}\n\n"
                if line == "__DONE__":
                    break
            except queue.Empty:
                yield "data: [WARN] Timeout\n\ndata: __DONE__\n\n"
                break
    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)