# Digitized result data

These files are **measured off the published figures** of Frey, Doeller & Barry
(2023), *Probing neural representations of scene perception in a hippocampally
dependent task using artificial neural networks*, arXiv:2303.06367. The original
arrays were not available. They exist only so the figures can be redrawn in the
current style; use the original run outputs for any new analysis.

How each quantity was recovered, from the full-resolution images embedded in the
arXiv PDF (`pdfimages`):

| file | method | accuracy |
|---|---|---|
| `fig3_scene_discrimination.json`, `figS4_allocentricity.json`, `figS5_lesion.json` | axis calibrated on the gridlines by linear fit; box quartiles and median located from the box fill and its median line; points found as connected components of their grey (train) or white (test) fill; whiskers recomputed from the points as the most extreme values within 1.5 IQR | quartiles and medians within about 0.003 of the drawn value. Points hidden entirely behind other points in the original cannot be recovered, so the point lists are incomplete (roughly 60-85 of each layer's points). |
| `figS4_readout.json` | markers found by Hough circles, assigned to a series by fill colour | about 0.01. Object World at PHC and CA3 is almost fully hidden in the original and is estimated from its visible arc (about 0.03). |
| `figS5_remapping.json`, `fig4_single_cells.json` (ratemaps) | grid located from the tick marks; each bin decoded from the median colour of its centre against the colour map, whose path was measured from the images themselves (it is not a stock matplotlib map) | median colour error 4/255. Values are positions along the colour map (0 lowest, 1 highest), not firing rates; the original has no colour bar. |
| `fig4_single_cells.json` (polar) | outer ring fitted by robust circle fit; dark pixels inside it binned by angle (5 degrees), median radius per bin | radius relative to the outer ring. |
