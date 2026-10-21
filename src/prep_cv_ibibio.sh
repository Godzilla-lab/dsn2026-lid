#!/bin/bash
# Common Voice Ibibio (MDC, CC0) -> competition parquet -> voice-overlap check -> Kaggle dataset -> folds_final2.csv
# Usage: bash src/prep_cv_ibibio.sh <archive.tar.gz>
set -e
S=/tmp/claude-0/-home-user-dsn2026-lid/a55f94d8-bdde-575d-8d9b-69b0f597d9e3/scratchpad
P=$S/kvenv/bin/python
R=/home/user/dsn2026-lid
ARCHIVE=$1
X=$S/cvibb_untrusted; OUT=$S/cvibb_data; K=$S/kaggle_cvibb
rm -rf $X $OUT $K; mkdir -p $X $OUT $K

echo "== extract"; tar -xzf "$ARCHIVE" -C $X
LOC=$(find $X -maxdepth 4 -type d -name clips | head -1); LOC=$(dirname "$LOC")
echo "locale dir: $LOC"; ls "$LOC" | head; wc -l "$LOC"/validated.tsv "$LOC"/invalidated.tsv "$LOC"/other.tsv 2>/dev/null || true

echo "== convert"
$P -I $R/src/convert_commonvoice.py "$LOC" ibb $OUT --release cv-corpus-27.0-2026-09-17 --per-speaker 120 \
   --url https://mozilladatacollective.com/datasets/cmu5uwjhw008znq07h4vu9kxp

echo "== voice overlap vs competition ibb/anw/efi speakers"
HF_HUB_OFFLINE=1 $P -I $R/src/voice_overlap.py --new $OUT --ref $S/dsn_data/data --labels ibb,anw,efi --out $OUT/overlap.csv

echo "== folds_final2"
$P - <<EOF
import pandas as pd
a=pd.read_csv('$R/splits/folds_final.csv'); b=pd.read_csv('$OUT/folds_ext_cv_ibb.csv')
ov=pd.read_csv('$OUT/overlap.csv'); bad=set(ov[ov.flagged].new_speaker)
b=b[~b.group.isin(bad)]
pd.concat([a,b]).to_csv('$R/splits/folds_final2.csv', index=False)
print('folds_final2:', len(a), '+', len(b), 'cv ibb rows (', len(bad), 'flagged speakers dropped ) ->', len(a)+len(b))
EOF

echo "== kaggle dataset"
cp $OUT/*.parquet $OUT/manifest.csv $R/splits/folds_final2.csv $K/
cat > $K/dataset-metadata.json <<EOF
{"title": "dsn2026-lid-cvibb", "id": "adewumiadenike/dsn2026-lid-cvibb", "licenses": [{"name": "CC0-1.0"}],
 "subtitle": "Private working copy of Common Voice 27.0 Ibibio (CC0) for team use", "description": "Converted from the Mozilla Data Collective download. Not for redistribution."}
EOF
KAGGLE_API_TOKEN="$KAGGLE_API_TOKEN_1" $S/kvenv/bin/kaggle datasets create -p $K --dir-mode zip 2>&1 | tail -2
echo "== done $(date -u +%H:%M)"
