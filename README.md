# High Quality ECG-Codec

-----------------

## Introduction

![Architecture of ECG-Codec](/assets/Fig1-Architecture.png "Overall Architecture")

This is the repo of implementation for **ECG-Codec** from paper: *"Ultra-High Quality ECG Compression for IoMT Application Using Temporal Convolutional Auto-Encoder with Improved RVQ"*

## Features

Our **ECG-Codec** outperformed compared methods in terms of **Quality Score(QS)**, which is a comprehensive metric evaluating performance of an ECG compressor. By maintaining low distortion while boosting our compress ratio, we achieved a **QS** of **42.7**

Besides, we implemented three versions with different compression ratio of **44, 58 and 88**, respectively.

![Results of ECG-Codec](/assets/Fig2-result_compare.png "Results")

To visualize our result intuitively, we showcase our reconstruction result on mit-bih datasets as below:
![reconstruction visualization](/assets/Fig3-mit_reconstruct.png "reconstruction visualization")

## Version

first commit:2025.01.05 version 1

## Citation

If your research have used our code, please cite this repo as below:

```latex
@misc{ECG-Codec,
  title={ECG-Codec: Ultra-High Quality ECG Compression Method},
  author={Yeyi Guan},
  year={2025},
  howpublished={\url{https://github.com/Guan-Y/ECG-Codec}},
}
```

## Acknowledgements

The code is partially referred to:

- [facebookresearch/encodec](https://github.com/facebookresearch/encodec)
- [lucidrains/vector-quantize-pytorch](https://github.com/lucidrains/vector-quantize-pytorch)
