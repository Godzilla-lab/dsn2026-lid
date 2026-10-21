#!/bin/bash
# D7: download the Kaggle run, add the chosen offsets, check, package, make public.
# Usage: bash src/package_d7.sh <offsets.json> [kernel slug=dsn2026-d7-full] [name=d7]
set -e
S=/tmp/claude-0/-home-user-dsn2026-lid/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad
P=$S/kvenv/bin/python
R=/home/user/dsn2026-lid
OFF=$1; KSLUG=${2:-dsn2026-d7-full}; NAME=${3:-d7}
OUT=$S/${NAME}_out; PKG=$S/pkg_$NAME
rm -rf $OUT $PKG; mkdir -p $OUT
cd $OUT && KAGGLE_API_TOKEN="$KAGGLE_API_TOKEN_1" $S/kvenv/bin/kaggle kernels output adewumiadenike/$KSLUG -p . 2>&1 | tail -1
$P -c "import json; d=json.load(open('$OUT/metrics.json')); print({k: d[k] for k in ('n_clips','audio_hours','model_train_s','model_checker_ok','model_bench') if k in d})"
cd $R
$P src/tune_bias.py apply $OUT/model $OFF $PKG | cut -c1-80
$P starting_kit/check_model_folder.py $PKG
APPROACH=$($P -c "
import re; d=open('$S/pkg_d5b/README.md').read(); a=re.search(r'## Approach\n\n(.*?)\n\n## Base model', d, re.S).group(1)
print(a + ' Training data additionally includes Common Voice 27.0 Ibibio and the Mozilla Data Collective Nupe speech collection, after a speaker-overlap check against the competition data.')")
$P src/package_submission.py $PKG godzilla1215/dsn2026-lid-$NAME submissions/${NAME^^} 5 \
  --ext-manifest research/ext_manifest.csv research/voiceafrica_manifest.csv research/ext2_manifest.csv research/cv_ibibio_manifest.csv research/mdc_nupe_manifest.csv \
  --approach "$APPROACH" 2>&1 | grep -E "PASS|uploaded|wrote|failed"
$P -I -c "
from huggingface_hub import HfApi
api=HfApi(); r='godzilla1215/dsn2026-lid-$NAME'; api.update_repo_settings(r, private=False); print(r, 'private:', api.model_info(r).private)"
cat submissions/${NAME^^}/INFO.txt
echo "== done $(date -u +%H:%M)"
