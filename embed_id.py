# -*- coding: utf-8 -*-
from chainer import cuda, Variable
from chainer import links as L

class EmbedID(L.EmbedID):

	def cosine_similarity(self, vec):
		vec = vec.data
		W = self.W.data
		xp = cuda.get_array_module(*(vec,))
		w_norm = xp.sqrt(xp.sum(W ** 2, axis=1))
		v_norm = xp.sqrt(xp.sum(vec ** 2, axis=1))
		inner_product = W.dot(vec.T)
		norm = w_norm.reshape(1, -1).T.dot(v_norm.reshape(1, -1)) + 1e-6
		# 最初の軸がIDに対応する
		return inner_product / norm

	def id(self, vec, sample=False):
		xp = cuda.get_array_module(*(vec.data,))
		cos = self.cosine_similarity(vec)
		if sample:
			sum = xp.exp(xp.sum(cos, axis=0))
			sum = sum.reshape(1, -1)
			print xp.exp(cos)
			print sum
			print xp.exp(cos) / sum
		else:
			return Variable(xp.argmax(cos, axis=0))

