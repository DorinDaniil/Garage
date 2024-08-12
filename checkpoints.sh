#!/bin/bash

git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1/ Garage/models/checkpoints/ppt-v2-1
git lfs clone https://huggingface.co/llava-hf/llava-1.5-7b-hf Garage/models/checkpoints/llava-1.5-7b-hf
git lfs clone https://huggingface.co/danulkin/llama Garage/models/checkpoints/llama-3-8b-Instruct
mkdir -p Garage/models/checkpoints/GroundedSegmentAnything
wget -q -P Garage/models/checkpoints/GroundedSegmentAnything https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -q -P Garage/models/checkpoints/GroundedSegmentAnything https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth