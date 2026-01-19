// Single-source frontend logic for EcoClassify
// - Handles file preview, upload, and rendering of result card
// - Loads history with optional filters (jenis, start/end date)
// - Renders statistics charts using Chart.js
// Default apiBase: when frontend is served statically (python -m http.server)
// requests must go to the backend Flask server on port 5000.
const apiBase = ((window.apiBase || '').replace(/\/+$/,'') || 'http://127.0.0.1:5000')

function qs(sel){ return document.querySelector(sel) }

async function predictImage(file){
  const form = new FormData();
  form.append('image', file, file.name);
  const url = (apiBase ? apiBase : '') + '/api/predict';
  const res = await fetch(url, { method: 'POST', body: form });
  return res.json();
}

function previewFile(file){
  const url = URL.createObjectURL(file);
  // set the single result thumbnail immediately so user sees the selected image
  const thumb = qs('#resThumb'); if(thumb) thumb.src = url;
  // show the result card as a preview placeholder (ready to classify)
  qs('#resultCard').style.display = 'block';
  qs('#resKategori').innerText = 'Menunggu klasifikasi';
  qs('#resJenisBadge').innerText = '';
  qs('#resAkurasiText').innerText = '';
  qs('#resProgressBar').style.width = '0%';
  qs('#resSaran').innerText = '';
  // disable save until a prediction is made
  const saveBtn = qs('#saveResult'); if(saveBtn) saveBtn.disabled = true;
  // enable classify button when a file is present
  const uploadBtn = qs('#uploadBtn'); if(uploadBtn) uploadBtn.disabled = false;
  // add a class to hide dropzone and show single preview/result card
  document.body.classList.add('has-preview');
  // show filename
  const fn = qs('#dzFilename'); if(fn) fn.textContent = file.name;
}

function showPage(name){
  qs('#page-home').style.display = name==='home' ? 'block' : 'none';
  qs('#page-history').style.display = name==='history' ? 'block' : 'none';
  qs('#page-stats').style.display = name==='stats' ? 'block' : 'none';
  document.querySelectorAll('.nav-link').forEach(a=> a.classList.toggle('active', a.dataset.page===name));
}

async function uploadAndPredict(){
  const f = qs('#fileInput').files[0];
  if(!f){ alert('Pilih gambar terlebih dahulu'); return }
  qs('#uploadBtn').disabled = true; qs('#uploadBtn').textContent = 'Memproses...';
  try{
    const j = await predictImage(f);
    // show thumbnail (use the uploaded file preview so user sees what was sent)
    try{
      const thumbUrl = URL.createObjectURL(f);
      const img = qs('#resThumb'); if(img) img.src = thumbUrl;
    }catch(e){}
    renderResult(j);
  }catch(e){ alert('Gagal menghubungi server: '+(e.message||e)); console.error(e) }
  finally{ qs('#uploadBtn').disabled = false; qs('#uploadBtn').textContent = 'Klasifikasikan' }
}

function renderResult(j){
  qs('#resultCard').style.display = 'block';
  // set kategori, jenis, accuracy
  const kategori = j.kategori || j.kategori_prediksi || j.kategori_sampah || '-';
  const jenisRaw = (j.jenis_sampah || j.jenis || j.jenisSampah || '').toString().toLowerCase();
  const jenis = jenisRaw || '-';
  // Normalize akurasi: backend might return 0-100 or 0-1
  let akVal = (j.akurasi || j.accuracy || j.akurasiRataRata || j.accuracyRataRata || 0);
  if(akVal > 0 && akVal <= 1) akVal = akVal * 100;
  const akurasi = Math.round(akVal * 10) / 10;
  qs('#resKategori').innerText = kategori;
  qs('#resJenisBadge').innerText = (jenis === 'organik') ? 'Organik' : (jenis === 'tidak_yakin' ? 'Tidak Yakin' : 'Anorganik');
  qs('#resAkurasiText').innerText = (akurasi) + '%';
  qs('#resProgressBar').style.width = Math.max(1, Math.min(100, akurasi)) + '%';
  qs('#resSaran').innerText = j.saran_edukasi || j.edukasi || '';

  // enable save button (prediction does NOT auto-save). The save button will POST a normalized payload
  const saveBtn = qs('#saveResult');
  if(saveBtn){
    saveBtn.disabled = false;
    saveBtn.onclick = async ()=>{
      // build normalized payload for saving
      const payload = {
        namaGambar: j.nama_gambar || j.file_name || (kategori || 'uploaded_image'),
        kategori: kategori,
        jenisSampah: jenis,
        akurasi: akurasi,
        edukasi: j.saran_edukasi || j.edukasi || ''
      };
      await saveResult(payload);
    };
  }

  // wire classify again button to reset UI
  const again = qs('#classifyAgain');
  if(again) again.onclick = ()=>{ qs('#fileInput').value=''; qs('#resultCard').style.display='none'; qs('#resThumb').src=''; qs('#preview').innerHTML=''; qs('#uploadBtn').disabled=true };
  if(again) again.onclick = ()=>{ qs('#fileInput').value=''; qs('#resultCard').style.display='none'; qs('#resThumb').src=''; qs('#preview').innerHTML=''; qs('#uploadBtn').disabled=true; document.body.classList.remove('has-preview'); const fn = qs('#dzFilename'); if(fn) fn.textContent=''; };
}

async function saveResult(j){
  // Accept either a normalized payload (from renderResult) or a raw prediction object
  const payload = {
    nama_gambar: j.namaGambar || j.nama_gambar || j.file_name || (j.kategori||'image'),
    kategori_sampah: j.kategori || j.kategori_sampah || j.kategori_prediksi,
    jenis_sampah: j.jenisSampah || j.jenis_sampah || j.jenis,
    akurasi: j.akurasi || j.akurasi || j.accuracy || 0,
    saran_edukasi: j.edukasi || j.saran_edukasi || j.edukasi || ''
  };
  try{
    const url = (apiBase ? apiBase : '') + '/api/history';
    const res = await fetch(url, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload) });
    const out = await res.json();
    alert('Disimpan ke riwayat.');
    await loadHistory();
    await loadStats();
  }catch(e){ alert('Gagal menyimpan: '+(e.message||e)); console.error(e) }
}

async function loadHistory(){
  try{
    // include filters: searchKategori (text) and filterJenis (select)
    const search = (qs('#searchKategori')?.value || '').toLowerCase();
    const jenis = qs('#filterJenis')?.value || '';
    const params = new URLSearchParams(); if(jenis) params.append('jenis', jenis);
    const url = (apiBase ? apiBase : '') + '/api/history' + (params.toString() ? ('?' + params.toString()) : '');
    const res = await fetch(url);
    const data = await res.json();
    const arr = Array.isArray(data) ? data : (data.history || data.items || []);
    const container = qs('#historyList');
    if(!arr.length){ container.innerHTML = '<div class="empty">Belum ada riwayat klasifikasi.</div>'; return }
    // build table-like list with delete action
    container.innerHTML = '';
    arr.forEach(h=>{
      const kategori = (h.kategori || h.kategori_sampah || '').toString();
      const jenisS = (h.jenisSampah || h.jenis_sampah || '').toString();
      if(search && !(kategori.toLowerCase().includes(search) || jenisS.toLowerCase().includes(search))) return;
      const row = document.createElement('div'); row.className = 'history-row';
      const nama = h.namaGambar || h.nama_gambar || (kategori||'-');
      const akur = h.akurasi || h.akurasi || 0;
      const waktu = h.waktu || h.waktu || '';
      const eduk = (h.edukasi || h.saran_edukasi || '');
      const id = h.id || '';
      row.innerHTML = `<div class="col col-kategori"><strong>${kategori}</strong><div class="muted">${jenisS}</div></div><div class="col col-akurasi">${akur}%</div><div class="col col-waktu">${waktu}</div><div class="col col-saran">${eduk}</div><div class="col col-actions"><button class="btn-delete" data-id="${id}">Hapus</button></div>`;
      container.appendChild(row);
    })
    // attach delete handlers
    container.querySelectorAll('.btn-delete').forEach(b=> b.addEventListener('click', async (ev)=>{
      const id = b.dataset.id; if(!id) return; if(!confirm('Hapus riwayat ini?')) return;
      try{
        const url = (apiBase ? apiBase : '') + '/api/history/' + id;
        const res = await fetch(url, { method:'DELETE' });
        const out = await res.json(); if(out.deleted){ alert('Dihapus'); await loadHistory(); await loadStats(); }
      }catch(e){ console.warn('delete history', e); alert('Gagal menghapus'); }
    }))
  }catch(e){ console.warn('loadHistory', e); qs('#historyList').innerHTML = '<div class="empty">Gagal memuat riwayat</div>' }
}

async function loadStats(){
  try{
    const url = (apiBase ? apiBase : '') + '/api/statistics';
    const res = await fetch(url); const j = await res.json();
    qs('#totalCount').textContent = j.total || 0;
    qs('#organikCount').textContent = (j.counts && j.counts.organik) ? j.counts.organik : 0;
    qs('#anorganikCount').textContent = (j.counts && j.counts.anorganik) ? j.counts.anorganik : 0;
    qs('#avgAcc').textContent = (j.avg_accuracy || j.avg_accuracy === 0) ? (j.avg_accuracy + '%') : '0%';

    // render charts
    const catCounts = j.counts_per_category || {};
    const labels = Object.keys(catCounts);
    const values = labels.map(l=>catCounts[l]);
    // categories bar chart
    const ctx1 = document.getElementById('chartCategories').getContext('2d');
    if(window._chartCategories) window._chartCategories.destroy();
    window._chartCategories = new Chart(ctx1, { type:'bar', data:{ labels, datasets:[{ label:'Jumlah per Kategori', data: values, backgroundColor:'#3b82f6' }] }, options:{ responsive:true } });
    // organik vs anorganik pie
    const ctx2 = document.getElementById('chartOrgAn').getContext('2d');
    if(window._chartOrgAn) window._chartOrgAn.destroy();
    window._chartOrgAn = new Chart(ctx2, { type:'pie', data:{ labels:['Organik','Anorganik'], datasets:[{ data:[ j.counts.organik || 0, j.counts.anorganik || 0 ], backgroundColor:['#10b981','#f97316'] }] }, options:{ responsive:true } });
  }catch(e){ console.warn('loadStats', e) }
}

document.addEventListener('DOMContentLoaded', ()=>{
  // nav
  document.querySelectorAll('.nav-link').forEach(a=> a.addEventListener('click', (ev)=>{ ev.preventDefault(); showPage(a.dataset.page); }));
  qs('#toUpload')?.addEventListener('click', ()=>{ showPage('home'); qs('#fileInput').focus(); });

  // file + upload
  qs('#fileInput').addEventListener('change', (e)=>{ const f = e.target.files[0]; if(f) { previewFile(f); } });
  qs('#uploadBtn').addEventListener('click', uploadAndPredict);
  qs('#clearBtn').addEventListener('click', ()=>{ qs('#fileInput').value=''; qs('#preview').innerHTML=''; qs('#resultCard').style.display='none'; document.body.classList.remove('has-preview'); const fn = qs('#dzFilename'); if(fn) fn.textContent=''; qs('#resThumb').src=''; qs('#uploadBtn').disabled=true });

  // nav buttons
  qs('#toUpload')?.addEventListener('click', ()=>{ showPage('home'); qs('#fileInput').focus(); });
  qs('#toStats')?.addEventListener('click', ()=>{ showPage('stats'); loadStats(); window.scrollTo({ top: 0, behavior: 'smooth' }); });
  qs('#navHome')?.addEventListener('click', ()=>{ showPage('home'); });
  qs('#navHistory')?.addEventListener('click', async ()=>{ showPage('history'); await loadHistory(); });
  // clear all history button
  qs('#clearAllHistory')?.addEventListener('click', async ()=>{
    if(!confirm('Hapus semua riwayat? Tindakan ini tidak dapat dibatalkan.')) return;
    try{
      const url = (apiBase ? apiBase : '') + '/api/history';
      const res = await fetch(url, { method:'DELETE' });
      const out = await res.json(); if(out.cleared){ alert('Semua riwayat dihapus'); await loadHistory(); await loadStats(); }
    }catch(e){ console.warn('clear all', e); alert('Gagal menghapus semua riwayat'); }
  });
  qs('#navStats')?.addEventListener('click', async ()=>{ showPage('stats'); await loadStats(); });

  // history filters
  qs('#applyFilter')?.addEventListener('click', async ()=>{ await loadHistory(); });
  // trigger search on enter in search box
  const searchBox = qs('#searchKategori');
  if(searchBox) searchBox.addEventListener('keydown', (ev)=>{ if(ev.key === 'Enter'){ ev.preventDefault(); loadHistory(); } });
  // debounce helper
  function debounce(fn, wait){ let t; return function(...a){ clearTimeout(t); t = setTimeout(()=>fn.apply(this,a), wait); } }
  // call loadHistory when filter selection changes
  qs('#filterJenis')?.addEventListener('change', ()=>{ loadHistory(); });
  // call loadHistory as user types (debounced)
  if(searchBox) searchBox.addEventListener('input', debounce(()=>{ loadHistory(); }, 350));

  // history / stats pages
  showPage('home');
  loadHistory(); loadStats();

  // Theme toggle: persist choice in localStorage and apply class to body
  const themeToggle = qs('#themeToggle');
  function applyTheme(t){
    if(t === 'dark'){ document.body.classList.remove('theme-light'); document.body.classList.add('theme-dark'); themeToggle.textContent = '☼'; }
    else { document.body.classList.remove('theme-dark'); document.body.classList.add('theme-light'); themeToggle.textContent = '☾'; }
    try{ localStorage.setItem('ecoclassify:theme', t); }catch(e){}
  }
  // initialize theme
  const stored = (function(){ try{ return localStorage.getItem('ecoclassify:theme') }catch(e){ return null } })() || 'light';
  applyTheme(stored === 'dark' ? 'dark' : 'light');
  if(themeToggle) themeToggle.addEventListener('click', ()=>{ const cur = document.body.classList.contains('theme-dark') ? 'dark' : 'light'; applyTheme(cur === 'dark' ? 'light' : 'dark'); });
});
