# SDEA-Block
SDEA block addapted from "Suppressing Features That Contain Disparity Edge For Stereo Matching"
`
X. Ai, Z. Yang, W. Yang, Y. Zhao, Z. Yu and F. Li, "Suppressing Features That Contain Disparity Edge For Stereo Matching," 2020 25th International Conference on Pattern Recognition (ICPR), 2021, pp. 7985-7991, doi: 10.1109/ICPR48806.2021.9413024.
`

#
Implemented according to information from the paper, but according to the parameter numbers for PSMNet implementation, it's still missing something.
|Network|Block count |Parameters  |Parameters in paper   |Missing|
|---------|-|----------|-----------|------|
|SDEA-1   | 3|5225027  | 5225158   | 131  |
|SDEA-Net | 6|5225414  | 5225548   | 134  |
|SDEA-2   | 9|5225513  | 5225650   | 137  |

Replacing layer G2 and scaling channel with Conv2D_BatchNorm instead of singular Conv2D fixes it, but it's not in the document.