# The hosted copy of the hard task

`hosted_index.html` is the page published as a claude.ai Artifact for remote
participants. It is `index.html` with three changes and nothing else:

1. **Images are published JPEGs under short names.** The trial data still names
   the benchmark's PNGs; `src()` maps
   `<mode>__a<scene>_<A|B>_az<deg>.png` to `<modeIndex><scene><A|B><deg/45>.jpg`
   (478 files, q82, 9.4 MB total). Regenerate them from `images/` if the bundle
   is rebuilt.
2. **Results come back by hand.** The artifact sandbox blocks a page-initiated
   download, so the file is offered through the `downloads` capability, with the
   clipboard and a textarea as the paths that always work. The participant
   emails the JSON back; a completion code lets a returned file be matched to
   its session. Declaring a database instead would make the artifact
   organisation-internal, which rules out naive participants.
3. **The reaction-time clock starts on image decode**, not on `src` assignment,
   and the next two trials are prefetched while the current one is answered.
   Locally the images were instant; over the network they are not, and `latency`
   is a dependent measure.

Everything the comparison depends on is untouched: the 100 trial ids, the
`neutral_anyview` instruction text, the benchmark sha (`5e64a7575cb7`), the
per-participant shuffle, the 2x2 option grid, and the output schema
`collate.py` reads.

`bench/restore_human_images.py <task_dir>` rebuilds a bundle's `images/` folder
from `data/scenes_100/`; the folder is gitignored and absent on a fresh clone.
