#!/usr/bin/env python

import torch

torch.manual_seed(1)
B, H, S, D = 2, 3, 3, 3
a = torch.rand((B, H, S, D))
print(a)

# flat-index i = b * HSD + h * SD + s * D + d
b,h,s,d = 0,1,1,1

k = a.flatten()
print("Flatten: ", a.flatten())
# k = k[b * H*S*D + h * S*D + s * D + d]

print("Accessing element at : ", b,h,s,d,)
print("Element: ", k[b * H*S*D + h * S*D + s * D + d])

# 4D coordinate (b,h,s,d)
i = 13
d1 = i % D
s1 = (i // D) % S
h1 = (i//(D *S)) % H
b1 = (i//(D *S * H)) % B

print("Element (4D): ", a[(b1,h1,s1,d1)])
