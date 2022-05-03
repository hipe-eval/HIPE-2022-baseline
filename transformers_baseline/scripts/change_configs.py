import os
import re
import json

conf_path = "transformers_baseline/configs"
os.getcwd()
fnames = [f for f in os.listdir(conf_path) if 'newseye' in f]
for fname in fnames:
    with open(os.path.join(conf_path, fname), 'r') as f:
        conf = json.loads(f.read())

    conf['do_train'] = False

    with open(os.path.join(conf_path, fname), "w") as outfile:
        json.dump(conf, outfile, indent=2, ensure_ascii=False)
