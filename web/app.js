const $ = id => document.getElementById(id);
const modeNames = {baseline:'Инерционный baseline · без внешней погоды', archive:'Архивный прогноз · проверка времени доступности', live:'Open-Meteo · текущий прогноз'};
const steps = {prepare:'Подготовка данных', weather:'Получение погоды', train:'Обучение модели', analyze:'Проверка результата', ai:'AI-анализ', save:'Сохранение'};
const number = (x, digits=2) => Number(x).toLocaleString('ru-RU', {maximumFractionDigits:digits, minimumFractionDigits:digits});
const date = value => new Date(value).toLocaleString('ru-RU', {timeZone:'UTC',month:'short',day:'2-digit',hour:'2-digit',minute:'2-digit'});

async function api(path, options) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : 'Проверьте параметры запроса.');
  return body;
}
function showError(message) {$('error').textContent=message; $('error').hidden=false;}
function chart(rows) {
  const w=1000,h=270,left=44,right=15,top=20,bottom=35;
  const x=i=>left+i*(w-left-right)/(rows.length-1);
  const y=v=>top+(1-v)*(h-top-bottom);
  const path=key=>rows.map((row,i)=>`${i?'L':'M'}${x(i).toFixed(2)},${y(row[key]).toFixed(2)}`).join(' ');
  const band=path('upper')+' '+rows.map((row,i)=>[row, i]).reverse().map(([row,i])=>`L${x(i).toFixed(2)},${y(row.lower).toFixed(2)}`).join(' ')+' Z';
  let content='';
  for(let i=0;i<=4;i++){const v=i/4;content+=`<line x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}" stroke="#e9eef1" stroke-dasharray="3 5"/><text x="0" y="${y(v)+4}" fill="#9aa6b3" font-size="11">${v.toFixed(2)}</text>`;}
  for(let i=0;i<rows.length;i+=Math.max(1,Math.floor(rows.length/6))){content+=`<text x="${x(i)}" y="${h-7}" text-anchor="middle" fill="#9aa6b3" font-size="10">${date(rows[i].timestamp)}</text>`;}
  content+=`<path d="${band}" fill="#e6f1eb"/><path d="${path('power')}" fill="none" stroke="#27846a" stroke-width="2.5" stroke-linejoin="round"/>`;
  rows.forEach((r,i)=>{content+=`<circle cx="${x(i)}" cy="${y(r.power)}" r="4" fill="#27846a" opacity=".1"><title>${date(r.timestamp)} UTC: ${number(r.power,3)}</title></circle>`;});
  $('chart').innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Почасовой прогноз нормализованной мощности">${content}</svg>`;
}
function render(result) {
  $('turbine').value=String(result.turbine);$('horizon').value=String(result.horizon);
  $('mode').value=result.mode;originPicker.setValue(result.origin,result.mode);
  $('mean').textContent=number(result.summary.mean_power*100,1)+' %';
  $('peak').textContent=number(result.summary.peak_power*100,1)+' %';
  $('energy').textContent=number(result.summary.equivalent_full_load_hours,1)+' ч';
  $('wind').textContent=number(result.summary.mean_wind_speed,1)+' м/с';
  $('mode-label').textContent=modeNames[result.mode];
  $('period').textContent=`${date(result.forecast[0].timestamp)} — ${date(result.forecast.at(-1).timestamp)} UTC`;
  $('chart-subtitle').textContent=`Турбина 0${result.turbine} · ${result.horizon} часов · ${result.weather.source}`;
  $('usable').textContent=number(result.quality.usable_hours,0);
  $('missing').textContent=number(result.quality.missing_hours,0);
  $('mae').textContent=number(result.validation.mae,4);
  $('rmse').textContent=number(result.validation.rmse,4);
  $('row-count').textContent=`${result.forecast.length} часов`;
  $('ai-text').textContent=result.ai.text;
  $('trace').replaceChildren(...result.trace.map(item=>{const li=document.createElement('li');const title=document.createElement('strong');title.textContent=steps[item.step]||item.step;li.append(title,document.createTextNode(item.message));return li;}));
  $('warnings').replaceChildren(...result.warnings.map(text=>{const li=document.createElement('li');li.textContent=text;return li;}));
  $('forecast-table').replaceChildren(...result.forecast.map(row=>{const tr=document.createElement('tr');[date(row.timestamp),number(row.wind_speed),number(row.temperature),number(row.power,3),`${number(row.lower,3)} – ${number(row.upper,3)}`].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});return tr;}));
  $('download').href=`/api/runs/${result.id}/csv`; $('download').hidden=false;
  chart(result.forecast);
}
async function refreshHistory() {
  const runs=await api('/api/runs');
  $('runs').replaceChildren(...runs.map(run=>{const button=document.createElement('button');button.className='run-item';const label=document.createElement('span');label.textContent=`Турбина 0${run.turbine} · ${run.horizon} ч · ${modeNames[run.mode]}`;const time=document.createElement('span');time.textContent=`${date(run.created_at)} UTC ↗`;button.append(label,time);button.onclick=async()=>{try{render(await api(`/api/runs/${run.id}`));$('error').hidden=true;window.scrollTo({top:0,behavior:'smooth'});}catch(e){showError(e.message);}};return button;}));
  if(!runs.length) $('runs').textContent='Сохранённых запусков пока нет.';
}
async function run(event) {
  event?.preventDefault(); $('error').hidden=true; $('run').disabled=true; $('run').textContent='Расчёт…';
  try {
    const result=await api('/api/forecast',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({turbine:Number($('turbine').value),horizon:Number($('horizon').value),mode:$('mode').value,origin:$('mode').value==='live'?null:originPicker.getValue(),use_ai:$('use-ai').checked})});
    render(result);await refreshHistory();
  } catch(error) {showError(error.message);} finally {$('run').disabled=false;$('run').textContent='Рассчитать ↗';}
}
$('forecast-form').addEventListener('submit',run);
$('mode').addEventListener('change',()=>originPicker.setMode($('mode').value));
$('refresh-history').addEventListener('click',()=>refreshHistory().catch(e=>showError(e.message)));
async function initialize(){
  try{
    const health=await api('/api/health');$('connection').textContent='● Система готова';
    $('ai-status').textContent=health.openai_configured?'ключ настроен · платные запросы':'добавьте ключ в .env';
    $('use-ai').disabled=!health.openai_configured;
    const runs=await api('/api/runs');
    if(runs.length){render(await api(`/api/runs/${runs[0].id}`));await refreshHistory();}
    else await run();
  }catch(error){$('connection').textContent='Нет связи с сервером';showError(error.message);}
}
initialize();
