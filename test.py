# -*- coding: utf-8 -*-
from chainer import cuda, Variable
from embed_id import EmbedID

xp = cuda.cupy
n_ids = 3
ndim_vec = 2
embed = EmbedID(n_ids, ndim_vec)
embed.to_gpu()

x = Variable(xp.arange(n_ids, dtype=xp.int32))
vec = embed(x)
_x = embed.reverse(Variable(xp.asarray([vec.data[0], vec.data[1], vec.data[2], vec.data[0] + vec.data[1], vec.data[0] - vec.data[1], vec.data[2], vec.data[1]], dtype=xp.float32)))
print _x.data
