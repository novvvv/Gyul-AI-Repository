#!/bin/bash

set -e

cd /content

rm -rf fish-speech-s2
git clone https://github.com/fishaudio/fish-speech.git fish-speech-s2

cd /content/fish-speech-s2

pip uninstall -y torchvision torchmetrics pytorch-lightning lightning || true

pip install torchmetrics==1.4.0.post0 pytorch-lightning==2.4.0 lightning==2.4.0
pip install -U "huggingface_hub>=0.34.0,<1.0" transformers==4.57.3 tiktoken
pip install -e .
pip uninstall -y torchvision || true

echo "Fish Speech setup complete."
