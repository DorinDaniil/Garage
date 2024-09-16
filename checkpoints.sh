#!/bin/bash

git lfs clone https://huggingface.co/JunhaoZhuang/PowerPaint-v2-1/ Garage/models/checkpoints/ppt-v2-1
git lfs clone https://huggingface.co/xtuner/llava-llama-3-8b-v1_1-transformers Garage/models/checkpoints/llava-llama-3-8b
mkdir -p Garage/models/checkpoints/GroundedSegmentAnything
wget -q -P Garage/models/checkpoints/GroundedSegmentAnything https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -q -P Garage/models/checkpoints/GroundedSegmentAnything https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth