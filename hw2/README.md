# Computer Vision Homework 2

AI3604 Computer Vision Homework 2. Only programming assignment included.



The final programming results are

- Camera Calibration Matrix:


    $$
    \begin{bmatrix}
    528.97964241 & 3.71885207 & 346.53669518 \\
    0 & 526.38154193 & 237.95483146 \\
    0 & 0 & 1 \\
    \end{bmatrix}
    $$


- Camera Calibration Matrix by OpenCV:


    $$
    \begin{bmatrix}
    532.66260902 & 0 & 341.95798391 \\
    0 & 532.65485994 & 232.05830837 \\
    0 & 0 & 1 \\
    \end{bmatrix}
    $$


- Reprojection Error:

    | Image | Error   | Error by OpenCV |
    |-------|---------|--------------------------|
    | 0     | 0.130   | 0.026                    |
    | 1     | 0.163   | 0.032                    |
    | 2     | 0.257   | 0.026                    |
    | 3     | 0.194   | 0.031                    |
    | 4     | 0.131   | 0.046                    |
    | 5     | 0.199   | 0.032                    |
    | 6     | 0.145   | 0.041                    |
    | 7     | 0.175   | 0.028                    |
    | 8     | 0.210   | 0.024                    |
    | 9     | 0.126   | 0.047                    |
    | 10    | 0.181   | 0.029                    |

### References

Z. Zhang, "A flexible new technique for camera calibration," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 22, no. 11, pp. 1330-1334, Nov. 2000, doi: 10.1109/34.888718.

