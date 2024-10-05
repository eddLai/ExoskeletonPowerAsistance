- 其中 `markers.xml` 是先用 OpenSim GUI 打開 OpenCap 錄下的軌跡,  然後從 `Navigator` -> `Markers` 右鍵下載
- 然後再手動刪掉 `model_generic.osim` 沒用到的 body 部分

---

- 然後下載的 `markers.xml` 要做以下處理
- 要把 `<socket_parent_frame>/bodyset/pelvis</socket_parent_frame>` 字串取代成 `<body>pelvis</body>`
- 其他像 `femur_r`, `tibia_r` ... etc 都是一樣處理

---

- 最後從 OpenSim GUI 下載 `model_scaled.osim`
