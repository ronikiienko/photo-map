#!/bin/sh
# Download 18 evenly spaced shards (every 4th) of huggan/wikiart into $1. Resumable: run again to carry on.
out="$1"
for i in $(seq 0 4 68); do
  name=$(printf "train-%05d-of-00072.parquet" "$i")
  if curl -sfL -C - --retry 5 -o "$out/$name" "https://huggingface.co/datasets/huggan/wikiart/resolve/main/data/$name"; then
    echo "got $name ($(du -sh "$out" | cut -f1) so far)"
  else
    echo "FAILED $name"
  fi
done
echo finished
