import random as rd

class sampler():
	def __init__(self):
		self.sampleBuffer = []

	def setBuffer(self, sampleRange):

		self.sampleBuffer = [i for i in range(sampleRange)]
		rd.shuffle(self.sampleBuffer)

	def sample(self, list1, list2, sampleNum):
		result_list1 = []
		result_list2 = []

		for i in range(sampleNum):
			idx = self.sampleBuffer.pop()
			result_list1.append(list1[idx])
			result_list2.append(list2[idx])

		return result_list1, result_list2

def sample(list1, list2, sampleNum, sampleRange):

	result_list1 = []
	result_list2 = []

	for i in range(sampleNum):
		idx = rd.randrange(0, sampleRange)
		result_list1.append(list1[idx])
		result_list2.append(list2[idx])
	
	return result_list1, result_list2

class FrozenBuffer():

	def __init__(self, d, bufferSize, rate):
		self.buffer = []
		self.maxBuffer = []

		self.dim = d
		self.bufferSize = bufferSize
		self.rate = rate

		self.alpha = 0.75 ** (1/ d)

		self.frozenFlagContainer = []

		for i in range(d):
			self.buffer.append([])
			self.maxBuffer.append(0)
			self.frozenFlagContainer.append(False)

	def _frozenCheck(self, step):

		changeFlag = False
		tmpContainer = []

		for i in range(self.dim):
			tmp = sum(self.buffer[i]) / len(self.buffer[i])
			if tmp < self.rate * (self.alpha ** (self.dim - i)) * self.maxBuffer[i] and not self.frozenFlagContainer[i]:
				tmpContainer.append(i + 1)
				self.frozenFlagContainer[i] = True
				changeFlag = True

		if changeFlag:
			print("step {}: dimension {} will be frozen!".format(step, tmpContainer))

		return changeFlag

	def insert(self, gradContainer, step):

		if len(gradContainer) != self.dim:
			print("need to pass data list which size is equal to dimension of subspace!")
			return
		
		if len(self.buffer[0]) >= self.bufferSize:
			for i in range(self.dim):
				self.buffer[i].pop()
				self.buffer[i].append(gradContainer[i])
				if gradContainer[i] > self.maxBuffer[i]: self.maxBuffer[i] = gradContainer[i]
		else:
			for i in range(self.dim):
				self.buffer[i].append(gradContainer[i])
				if gradContainer[i] > self.maxBuffer[i]: self.maxBuffer[i] = gradContainer[i]

		return self._frozenCheck(step)
	
	def getFlagContainer(self):

		return self.frozenFlagContainer