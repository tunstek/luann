local luann = require("luann")
math.randomseed(89890)

local useVariableLearningRate = true
local learningRate = 0.1 -- set between 0, 1
local err = 0.01 -- train until RMSE < 0.01

--create a network with 2 inputs, 3 hidden cells, and 1 output
myNetwork = luann:new({2, 3, 1}, learningRate, 'sigmoid')
--myNetwork = luann:new({2, 3, 3, 3, 1}, learningRate, 'sigmoid') -- uncomment for an exciting convergance

local inputs = {
	{0,0}, {1,0}, {0,1}, {1,1}
}
local expectedOutputs = {
	{0}, {1}, {1}, {0}
}

-- train the network
myNetwork:train(inputs, expectedOutputs, err, useVariableLearningRate)

--print the signal of the single output cell when :activated with different inputs
print("Results:")
print("0 0 | " .. myNetwork:forwardPropagate({0,0})[1])
print("0 1 | " .. myNetwork:forwardPropagate({0,1})[1])
print("1 0 | " .. myNetwork:forwardPropagate({1,0})[1])
print("1 1 | " .. myNetwork:forwardPropagate({1,1})[1])

--Save the network to a file
luann:saveNetwork(myNetwork, "demoNetwork.dump")

--Load the network from a file
newNetwork = luann:loadNetwork("demoNetwork.dump")

--run the loaded network
print("Results:")
print("Output of 0,0: " .. myNetwork:forwardPropagate({0,0})[1])
print("Output of 0,1: " .. myNetwork:forwardPropagate({0,1})[1])
print("Output of 1,0: " .. myNetwork:forwardPropagate({1,0})[1])
print("Output of 1,1: " .. myNetwork:forwardPropagate({1,1})[1])
