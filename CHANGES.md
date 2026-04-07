# Cellpose Notebook — Planned Changes

## Status Key
- [ ] Not started
- [x] Done

## Changes
- [x] 1. Fix 0-cell detection — add `CELLPOSE_MODEL` config, confirm thresholds
- [x] 2. Diagnostic cell — print image shape/dtype/min/max, show raw image
- [x] 3. Save segmentation overlay PNGs to `SAVE_DIR/overlays/`
- [x] 4. Progress bar (tqdm) in batch processing loop
- [x] 5. Save violin plot PNG to `SAVE_DIR/`
- [x] 6. Save scatter plot PNG to `SAVE_DIR/`
- [x] 7. Per-condition summary CSV (`SAVE_DIR/summary.csv`)
- [x] 8. Cell size filtering (`MIN_CELL_AREA_PX`, `MAX_CELL_AREA_PX` in config)
- [x] 9. Timestamped output filename for results CSV
- [x] 10. Error handling for corrupt/unreadable images
- [x] 11. Analysis log file (`SAVE_DIR/run_log.txt`)
