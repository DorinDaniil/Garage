#!/bin/bash

git clone https://github.com/SunzeY/AlphaCLIP
pip install -e AlphaCLIP/
pip install loralib
wget https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_b16_grit1m_fultune_8xe.pth
