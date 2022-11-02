
# ECLIPSE: Efficient Long-range Video Retrieval using Sight and Sound 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <img src="https://raw.githubusercontent.com/facebookresearch/unbiased-teacher/main/teaser/pytorch-logo-dark.png" width="10%"> 
<!-- [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) -->

This is the PyTorch implementation of our paper: <br>
**ECLIPSE: Efficient Long-range Video Retrieval using Sight and Sound**<br>
[Yan-Bo Lin](https://genjib.github.io/), [Jie Lei](https://jayleicn.github.io/), [Mohit Bansal](https://www.cs.unc.edu/~mbansal/), and [Gedas Bertasius](https://www.gedasbertasius.com/)<br>
In European Conference on Computer Vision, 2022. <br>

[paper](https://arxiv.org/abs/2204.02874) 

### 📝 Preparation 
1. `pip3 install requirements.txt`
2. Dataset:  ActivityNet, QVHighlights, YouCook2, DiDeMo and Charades.
3. extract video frames in 3 fps.
4. extract audio features.


### 💿 Extract images and audio features. 
```shell
ActivityNet/
├── raw_frames/
│       └── VIDEO_NAME/
│           ├── 0001.jpg
│           ├── ...
│           └── 00...jpg
│
└── VGGSound_Audio_features_10s_aligned/
        └── VIDEO_NAME/
            ├── 0000.pt
            ├── ...
            └── 00...pt

```



### 💿 Extracted audio features. 
VGGSound features on ActivityNet Captions: [Google Drive](https://drive.google.com/file/d/1PbZPrgO5HTuG_CORcS_zScQCUeFo1JOL/view?usp=sharing)

### 📚 Train and evaluate
ActivityNet Captions: `bash run_act.sh` \
DiDemo: `bash run_didemo.sh` \
Charades: `bash run_cha.sh` \
QVHighlight:`bash run_qvh.sh` \
YouCook2: `bash run_yc2.sh`




### 🎓 Cite

If you use this code in your research, please cite:

```bibtex
@InProceedings{ECLIPSE_ECCV22,
author = {Yan-Bo Lin and Jie Lei and Mohit Bansal and Gedas Bertasius},
title = {ECLIPSE: Efficient Long-range Video Retrieval using Sight and Sound},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {October},
year = {2022}
}
```

### 👍 Acknowledgments
Our code is based on [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip) and [VGGSound](https://www.robots.ox.ac.uk/~vgg/data/vggsound/)

### ✏ Future works
* Preprocessed video frames and audio features


## License

This project is licensed under [MIT License](LICENSE), as found in the LICENSE file.
