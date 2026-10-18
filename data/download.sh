#!/usr/bin/env bash
# Fetch raw datasets. Kaggle / Zenodo / GroupLens are not reachable from the cloud environment used
# to build this project, so we use public GitHub mirrors (RecoHut-Datasets). On your own machine you
# can instead download the originals: RetailRocket (kaggle.com/datasets/retailrocket/ecommerce-dataset),
# MovieLens-1M (grouplens.org/datasets/movielens/1m), KuaiRec (kuairec.com).
#   RetailRocket events.csv md5 fddaf7c1c6a9df23a726500a08ffdd7e (2,756,101 events, identical to Kaggle)
set -euo pipefail
cd "$(dirname "$0")"
tmp=$(mktemp -d)

git clone -q --filter=blob:none --no-checkout https://github.com/RecoHut-Datasets/retail_rocket "$tmp/rr"
git -C "$tmp/rr" fetch -q --filter=blob:none origin v2
git -C "$tmp/rr" show FETCH_HEAD:retailrocket.zip > "$tmp/retailrocket.zip"
mkdir -p raw/retailrocket
unzip -o -q -j "$tmp/retailrocket.zip" retailrocket/events.csv -d raw/retailrocket

git clone -q --filter=blob:none --no-checkout https://github.com/RecoHut-Datasets/movielens_1m "$tmp/ml"
mkdir -p raw/ml-1m
for f in ml1m_items.dat ml1m_ratings.dat ml1m_users.dat; do git -C "$tmp/ml" show HEAD:$f > raw/ml-1m/$f; done

rm -rf "$tmp"
echo "KuaiRec: not mirrored; place KuaiRec.zip from kuairec.com in raw/kuairec/ if you want it (needed from step S7)."
