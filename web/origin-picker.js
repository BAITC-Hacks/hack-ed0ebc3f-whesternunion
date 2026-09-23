// Fixed UTC+5 matches the agreed forecast schedule, regardless of the browser timezone.
const originPicker = (() => {
  const element = id => document.getElementById(id);
  const dialog = element('origin-dialog');
  const pad = value => String(value).padStart(2, '0');
  const initial = {day:'2026-01-31', hour:23};
  const selections = {baseline:{...initial}, archive:{...initial}, live:{...initial}};
  let mode = 'baseline';
  let draft = {...initial};
  let month = new Date('2026-01-01T00:00:00Z');

  function label(selection) {
    const day = new Date(`${selection.day}T00:00:00Z`).toLocaleDateString('ru-RU', {
      timeZone:'UTC', day:'numeric', month:'short', year:'numeric',
    });
    return `${day}, ${pad(selection.hour)}:00`;
  }
  function syncTrigger() {
    element('origin-text').textContent=label(selections[mode]);
    element('origin').disabled=mode==='live';
    element('origin-hint').textContent={
      live:'Live: время расчёта определяется автоматически',
      archive:'Архив: 31.01–28.02.2026 · выпуски в 23:00',
      baseline:'Начальный расчёт: 31.01.2026 в 23:00',
    }[mode];
    element('picker-context').textContent=element('origin-hint').textContent;
  }
  function updateSelection() {
    element('picker-selection').textContent=`Выбрано: ${label(draft)} · UTC+5`;
  }
  function renderDays() {
    element('picker-month').textContent=month.toLocaleDateString('ru-RU', {
      timeZone:'UTC', month:'long', year:'numeric',
    });
    const year=month.getUTCFullYear(), index=month.getUTCMonth();
    const offset=(month.getUTCDay()+6)%7;
    const count=new Date(Date.UTC(year,index+1,0)).getUTCDate();
    const cells=[];
    for(let i=0;i<offset;i++) {
      const blank=document.createElement('span');
      blank.setAttribute('aria-hidden','true'); cells.push(blank);
    }
    for(let day=1;day<=count;day++) {
      const value=`${year}-${pad(index+1)}-${pad(day)}`;
      const button=document.createElement('button');
      button.type='button'; button.textContent=String(day); button.dataset.day=value;
      button.setAttribute('aria-label',new Date(`${value}T00:00:00Z`).toLocaleDateString('ru-RU', {
        timeZone:'UTC', day:'numeric', month:'long', year:'numeric',
      }));
      button.setAttribute('aria-pressed',String(draft.day===value));
      button.addEventListener('click',()=>{
        draft.day=value;
        // Keep keyboard focus on the clicked day instead of replacing its element.
        element('picker-days').querySelectorAll('button').forEach(cell=>{
          cell.setAttribute('aria-pressed',String(cell.dataset.day===value));
        });
        updateSelection();
      });
      cells.push(button);
    }
    element('picker-days').replaceChildren(...cells);
  }
  function renderHours() {
    element('picker-hours').replaceChildren(...Array.from({length:24},(_,hour)=>{
      const button=document.createElement('button');
      button.type='button'; button.textContent=`${pad(hour)}:00`; button.dataset.hour=String(hour);
      button.setAttribute('aria-pressed',String(draft.hour===hour));
      button.addEventListener('click',()=>{
        draft.hour=hour;
        element('picker-hours').querySelectorAll('button').forEach(slot=>{
          slot.setAttribute('aria-pressed',String(Number(slot.dataset.hour)===hour));
        });
        updateSelection();
      });
      return button;
    }));
  }
  element('origin').addEventListener('click',()=>{
    draft={...selections[mode]};
    month=new Date(`${draft.day.slice(0,7)}-01T00:00:00Z`);
    renderDays(); renderHours(); updateSelection();
    dialog.showModal();
    element('picker-days').querySelector('[aria-pressed="true"]')?.focus();
  });
  function changeMonth(delta) {
    month.setUTCMonth(month.getUTCMonth()+delta); renderDays();
  }
  element('picker-prev').addEventListener('click',()=>changeMonth(-1));
  element('picker-next').addEventListener('click',()=>changeMonth(1));
  for(const id of ['picker-close','picker-cancel']) {
    element(id).addEventListener('click',()=>dialog.close());
  }
  element('picker-apply').addEventListener('click',()=>{
    selections[mode]={...draft}; syncTrigger(); dialog.close();
  });
  dialog.addEventListener('click',event=>{
    if(event.target!==dialog) return;
    const rect=dialog.getBoundingClientRect();
    if(event.clientX<rect.left || event.clientX>rect.right || event.clientY<rect.top || event.clientY>rect.bottom) dialog.close();
  });
  // Native <dialog> handles Escape, focus trapping and returning focus to the trigger.
  syncTrigger();
  return {
    getValue() {
      const selection=selections[mode];
      return `${selection.day}T${pad(selection.hour)}:00:00+05:00`;
    },
    setValue(value, nextMode=mode) {
      const timestamp=new Date(value);
      if(!Number.isFinite(timestamp.getTime())) throw new Error('Некорректный момент расчёта.');
      const local=new Date(timestamp.getTime()+5*60*60*1000);
      selections[nextMode]={day:local.toISOString().slice(0,10), hour:local.getUTCHours()};
      mode=nextMode; syncTrigger();
    },
    setMode(nextMode) {
      mode=nextMode; syncTrigger();
    },
  };
})();
