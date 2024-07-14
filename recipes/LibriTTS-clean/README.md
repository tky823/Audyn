# LibriTTS(-clean) recipes

This directory is typically used to train text-to-speech models.
Originally, speakers in LibriTTS are split into disjointed subsets.
However, we concatenate `train-clean-100`, `dev-clean`, and `test-clean`, and re-split utterances to contain all speakers are included in all subsets.
