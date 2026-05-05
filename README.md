# construction-site-safety-monitor
# 🦺 SafeSite Vision
> Automated PPE compliance monitoring for construction sites using computer vision.

SafeSite Vision is an end-to-end pipeline that processes construction-site images, 
recorded video, or live camera feeds and outputs a **0–100 compliance score** along 
with per-worker violation reports.

It detects missing helmets and reflective vests using a **YOLOv8s model fine-tuned 
on 2,400+ construction images**, tracks individual workers across frames with SORT, 
and delivers results through both a CLI and a Flask drag-and-drop web UI.

## Key Features
- 🎯 Per-worker violation detection (not just frame-level counts)
- 🔁 Persistent worker tracking across frames via SORT + Kalman filtering  
- 📊 Temporal EMA smoothing to eliminate single-frame false alerts
- 🌐 Flask web UI with live compliance timeline (Chart.js)
- ✅ 34-test suite — runs without GPU or model weights
