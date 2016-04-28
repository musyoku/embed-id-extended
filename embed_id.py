# -*- coding: utf-8 -*-
from chainer import cuda, Variable
from chainer import links as L

class EmbedID(L.EmbedID):

	def reverse(self, vec):
		vec = vec.data
		W = self.W.data
		xp = cuda.get_array_module(*(vec,))
		w_norm = xp.sqrt(xp.sum(W ** 2, axis=1))
		v_norm = xp.sqrt(xp.sum(vec ** 2, axis=1))
		product = W.dot(vec.T)
		norm = w_norm.reshape(1, -1).T.dot(v_norm.reshape(1, -1)) + 1e-6
		return Variable(xp.argmax(product / norm, axis=0))
