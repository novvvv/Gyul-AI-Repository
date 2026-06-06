#!/bin/bash

set -e

cd /content/fish-speech-s2

mkdir -p checkpoints/s2-pro

hf download fishaudio/s2-pro --local-dir checkpoints/s2-pro

echo "S2-Pro checkpoint downloaded."
