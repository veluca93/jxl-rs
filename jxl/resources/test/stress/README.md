# Multi-group stress corpus

Images large enough to decode with multiple groups (most other test images
are single-group and never exercise the parallel decode paths), covering a
matrix of coding features. Mainly useful for the shuttle concurrency tests.

Inputs are procedurally generated (gradients/sine patterns, low-color-count
blocks, repeated tiles, alpha circles, sharp edges; 512x384 up to 2100x300 so
that the wide/tall ones cross the 2048px LF group boundary), then encoded
with cjxl v0.7.0:

| file | cjxl flags |
|---|---|
| stress_photo_lossless | -d 0 -e 7 |
| stress_photo_lossless_g0 | -d 0 -e 7 -g 0 |
| stress_gray_lossless_g1 | -d 0 -e 8 -g 1 |
| stress_photo_resp0 | -d 0 -e 7 -R 0 |
| stress_photo_wp | -d 0 -e 7 -P 6 -g 0 |
| stress_palette_lossless | -d 0 -e 7 |
| stress_palette_delta | -d 1 -m 1 --modular_lossy_palette -e 7 -g 0 |
| stress_photo_vardct | -d 1.0 -e 6 |
| stress_edges_epf3 | -d 2.0 -e 6 --epf=3 |
| stress_photo_progressive | -d 1.0 -e 6 -p |
| stress_photo_qprog | -d 1.0 -e 6 --qprogressive_ac --progressive_dc=1 |
| stress_photo_centerfirst | -d 1.0 -e 6 --group_order=1 |
| stress_photo_upsample4 | -d 1.0 -e 6 --resampling=4 |
| stress_gray_upsample4 | -d 2.0 -e 6 --resampling=4 |
| stress_photo_noise | -d 1.0 -e 6 --photon_noise=ISO3200 |
| stress_tiles_patches | -d 1.0 -e 8 --patches=1 |
| stress_wide_multilf | -d 1.0 -e 6 |
| stress_wide_multilf_lossless | -d 0 -e 7 |
| stress_tall_multilf | -d 1.0 -e 6 --progressive_dc=1 |
| stress_alpha | -d 1.0 -e 6 |
| stress_alpha_premul | -d 1.0 -e 6 --premultiply=1 |
| stress_alpha_lossless | -d 0 -e 7 |
