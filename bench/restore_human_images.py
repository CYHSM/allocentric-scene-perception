import json, os, re, shutil, sys

task_dir = sys.argv[1]
root = os.path.dirname(os.path.dirname(os.path.abspath(task_dir)))
html = open(os.path.join(task_dir, "index.html")).read()
m = re.search(r"^const DATA = (\{.*\});\s*$", html, re.M)
DATA = json.loads(m.group(1))

names = set()
for t in DATA["trials"]:
    names.add(t["study"]); names.update(t["options"])

out = os.path.join(task_dir, "images")
os.makedirs(out, exist_ok=True)
missing, copied = [], 0
for n in sorted(names):
    mode, rest = n.split("__", 1)          # c0_shape_colour__a064_A_az315.png
    scene = rest.split("_")[0]             # a064
    src = os.path.join(root, "data", "scenes_100", mode, scene, rest)
    dst = os.path.join(out, n)
    if not os.path.exists(src):
        missing.append(src); continue
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
    copied += 1
print(f"{copied}/{len(names)} images in {out}")
if missing:
    print(f"MISSING {len(missing)}:")
    for s in missing[:10]: print("  ", s)
