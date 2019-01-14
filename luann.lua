--[[
The MIT License (MIT)

Copyright (c) <2013> <Josh Rowe>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

]]--

--Borrowed table persistence from http://lua-users.org/wiki/TablePersistence, MIT license.
--comments removed, condensed code to oneliners where possible.
local write, writeIndent, writers, refCount;
persistence =
{
	store = function (path, ...)
		local file, e = io.open(path, "w")
		if not file then return error(e)	end
		local n = select("#", ...)
		local objRefCount = {} -- Stores reference that will be exported
		for i = 1, n do refCount(objRefCount, (select(i,...))) end
		local objRefNames = {}
		local objRefIdx = 0;
		file:write("-- Persistent Data\n");
		file:write("local multiRefObjects = {\n");
		for obj, count in pairs(objRefCount) do
			if count > 1 then
				objRefIdx = objRefIdx + 1;
				objRefNames[obj] = objRefIdx;
				file:write("{};"); -- table objRefIdx
			end;
		end;
		file:write("\n} -- multiRefObjects\n");
		for obj, idx in pairs(objRefNames) do
			for k, v in pairs(obj) do
				file:write("multiRefObjects["..idx.."][");
				write(file, k, 0, objRefNames);
				file:write("] = ");
				write(file, v, 0, objRefNames);
				file:write(";\n");
			end;
		end;
		for i = 1, n do
			file:write("local ".."obj"..i.." = ");
			write(file, (select(i,...)), 0, objRefNames);
			file:write("\n");
		end
		if n > 0 then
			file:write("return obj1");
			for i = 2, n do
				file:write(" ,obj"..i);
			end;
			file:write("\n");
		else
			file:write("return\n");
		end;
		file:close();
	end;
	load = function (path)
		local f, e = loadfile(path);
		if f then
			return f();
		else
			return nil, e;
		end;
	end;
}
write = function (file, item, level, objRefNames)
	writers[type(item)](file, item, level, objRefNames);
end;
writeIndent = function (file, level)
	for i = 1, level do
		file:write("\t");
	end;
end;
refCount = function (objRefCount, item)
	if type(item) == "table" then
		if objRefCount[item] then
			objRefCount[item] = objRefCount[item] + 1;
		else
			objRefCount[item] = 1;
			for k, v in pairs(item) do
				refCount(objRefCount, k);
				refCount(objRefCount, v);
			end;
		end;
	end;
end;
writers = {
	["nil"] = function (file, item) file:write("nil") end;
	["number"] = function (file, item)
			file:write(tostring(item));
		end;
	["string"] = function (file, item)
			file:write(string.format("%q", item));
		end;
	["boolean"] = function (file, item)
			if item then
				file:write("true");
			else
				file:write("false");
			end
		end;
	["table"] = function (file, item, level, objRefNames)
			local refIdx = objRefNames[item];
			if refIdx then
				file:write("multiRefObjects["..refIdx.."]");
			else
				file:write("{\n");
				for k, v in pairs(item) do
					writeIndent(file, level+1);
					file:write("[");
					write(file, k, level+1, objRefNames);
					file:write("] = ");
					write(file, v, level+1, objRefNames);
					file:write(";\n");
				end
				writeIndent(file, level);
				file:write("}");
			end;
		end;
	["function"] = function (file, item)
			local dInfo = debug.getinfo(item, "uS");
			if dInfo.nups > 0 then
				file:write("nil --[[functions with upvalue not supported]]");
			elseif dInfo.what ~= "Lua" then
				file:write("nil --[[non-lua function not supported]]");
			else
				local r, s = pcall(string.dump,item);
				if r then
					file:write(string.format("loadstring(%q)", s));
				else
					file:write("nil --[[function could not be dumped]]");
				end
			end
		end;
	["thread"] = function (file, item)
			file:write("nil --[[thread]]\n");
		end;
	["userdata"] = function (file, item)
			file:write("nil --[[userdata]]\n");
		end;
}



local luann = {}
local Layer = {}
local Cell = {}
local Activation = {}


--Define the activation functions
Activation["sigmoid"] = function(signalSum)
	return 1 / (1 + math.exp(-1*signalSum))
end
Activation["relu"] = function(signalSum)
	return math.max(0, signalSum)
end
Activation["leakyrelu"] = function(signalSum)
	return math.max(0.01*signalSum, signalSum)
end


--// Cell:new(numInputs)
-- numInputs -> The number of inputs to the new cell
--		Note: The cell has a structure containing weights that modify the input from the previous layer.
--					Each cell also has a signal, or output.
--					weightsPrev is used to store the current weights for variable learning rate
function Cell:new(numInputs)
	local cell = {delta = 0, weights = {}, weightsPrev = {}, signal = 0}
	for i = 1, numInputs do
		cell.weights[i] = math.random() - 0.5 -- keep values between -0.5 and 0.5
	end
	setmetatable(cell, self)
	self.__index = self
	return cell
end

--// Cell:activate(inputs, bias, actFuncName)
-- inputs -> The collection of inputs to the cell
-- bias -> The bias input to the cell
-- actFuncName -> The name of the activation function (see luann:new)
function Cell:activate(inputs, bias, actFuncName)
		local signalSum = bias
		local weights = self.weights
		for i = 1, #weights do
			signalSum = signalSum + (weights[i] * inputs[i])
		end
		res = Activation[actFuncName](signalSum)
		self.signal = res
end


--// Layer:new([numCells [, numInputs]])
-- numCells -> Optional (default: 1). Number of cells in this layer
-- numInputs -> Optional (default: 1). Number of inputs to this layer
-- 		Note: The layer is a table of cells.
--					biasPrev is used to store the current bias for variable learning rate
function Layer:new(numCells, numInputs)
	numCells = numCells or 1
	numInputs = numInputs or 1
	local cells = {}
	for i = 1, numCells do
			cells[i] = Cell:new(numInputs)
	end
	local layer = {cells = cells, bias = math.random() - 0.5, biasPrev = -1}
	setmetatable(layer, self)
	self.__index = self
	return layer
end


--// luann:new(layers, learningRate [, actiFuncName])
-- layers -> Table of layer sizes from input to output
-- actiFuncName -> Optional (default: 'sigmoid'). Defines the NAME of the function (in the Activation table) to use in for activation.
--   Note: For actiFuncName, the function itself cannot be passed due to issues with table persistence.
--				 This is currently considered a workaround..
function luann:new(layers, learningRate, actiFuncName)
	local network = {learningRate = learningRate, actiFuncName = actiFuncName or 'sigmoid'}
	--initialize the input layer
	network[1] = Layer:new(layers[1], layers[1])
	for i = 2, #layers do
		--initialize the hidden layers and output layer
		network[i] = Layer:new(layers[i], layers[i-1])
	end
	setmetatable(network, self)
	self.__index = self
	return network
end

--// luann:train(inputs, expectedOutput, rmseThreshold)
-- inputs = collection of a collection of inputs to train on
-- expectedOutput = collection of a collection of expected outputs for the given inputs
-- rmseThreshold = train until RMSE falls below this value
function luann:train(inputs, expectedOutput, rmseThreshold)
	local out = {}
	local count = 0
	local rmse = -1
	local rmsePrev = -1
	repeat
		self:backupWeights()
		for i=1, #inputs do
			-- for each set of inputs
			self:bp(inputs[i], expectedOutput[i])
			out[i] = self:getOutputs()
		end
		rmse = self:getRMSE(out, expectedOutput)
		-- variable learning rate
		if rmsePrev ~= -1 then
			-- this is not the first iteration
			if rmse > rmsePrev*1.04 then
				self:restoreWeights()
				self.learningRate = self.learningRate*0.7
				if self.learningRate < 0.00000000001 then self.learningRate = 0.00000000001
				else print("Decreasing learning rate: " .. self.learningRate) print(rmse) end
			elseif rmse < rmsePrev then
				self.learningRate = self.learningRate*1.05
				if self.learningRate > 1 then self.learningRate = 1
				else print("Increasing learning rate: " .. self.learningRate) print(rmse) end
			end
		end

		rmsePrev = rmse
		-- some feedback during training
		if count>10000 then
			print("Current RMSE: " .. rmse)
			count = 0
		end
		count = count + 1
	until(rmse < rmseThreshold)
end

--// luann:forwardPropagate(inputs)
-- propagates the inputs and returns the outputs, a convience function
function luann:forwardPropagate(inputs)
	self:activate(inputs)
	return self:getOutputs()
end

--// luann:getOutputs()
-- gets the set of previous outputs
function luann:getOutputs()
	local out = {}
	for o=1, #self[#self].cells do
		out[o] = self[#self].cells[o].signal
	end
	return out
end

--// luann:getRMSE(predictions, expected)
-- predictions = a collection of a collection of predictions
-- expected = a collection of a collection of expected outputs
function luann:getRMSE(predictions, expected)
	assert(#predictions == #expected, "ERR: #predictions ~= #expected")
	local numElements = 0
	local sum = 0
	for i=1, #predictions do
		-- for each set of predictions
		for j=1, #predictions[i] do
			-- for each output node
			numElements = numElements+1
			sum = sum + math.pow(expected[i][j]-predictions[i][j],2)
		end
	end
	local mse = sum/numElements
	return math.sqrt(mse)
end

--// luann:activate(inputs)
-- inputs -> The collection of inputs to be forward propagated through the network
function luann:activate(inputs)
	for i = 1, #inputs do
		self[1].cells[i].signal = inputs[i]
	end
	for i = 2, #self do
		local passInputs = {}
		local cells = self[i].cells
		local prevCells = self[i-1].cells
		for m = 1, #prevCells do
			passInputs[m] = prevCells[m].signal
		end
		local passBias = self[i].bias
		for j = 1, #cells do
			--activate each cell
			cells[j]:activate(passInputs, passBias, self.actiFuncName)
		end
	end
end

function luann:backupWeights()
	local numLayers = #self
	for i = 2, numLayers do
		-- save the current bias
		self[i].biasPrev = self[i].bias
		for j = 1, #self[i].cells do
			for k = 1, #self[i].cells[j].weights do
				-- save the current weights
				local weights = self[i].cells[j].weights
				local weightsPrev = self[i].cells[j].weightsPrev
				weightsPrev[k] = weights[k]
			end
		end
	end
end

function luann:restoreWeights()
	local numLayers = #self
	for i = 2, numLayers do
		-- restore the prev bias
		self[i].bias = self[i].biasPrev
		for j = 1, #self[i].cells do
			for k = 1, #self[i].cells[j].weights do
				-- restore the prev weights
				local weights = self[i].cells[j].weights
				local weightsPrev = self[i].cells[j].weightsPrev
				weights[k] = weightsPrev[k]
			end
		end
	end
end

--// luann:bp(inputs, expectedOutputs)
-- inputs -> The collection of training inputs for backpropagation
-- expectedOutputs -> The collection of expected outputs for the given training inputs
--		Note: Contains some debug info for when cell weights become inf and cell deltas become NaN
function luann:bp(inputs, expectedOutputs)
	self:activate(inputs) --update the internal inputs and outputs
	local numLayers = #self
	local learningRate = self.learningRate
	for i = numLayers, 2, -1 do --iterate backwards (nothing to calculate for input layer)
		local numCells = #self[i].cells
		local cells = self[i].cells
		for j = 1, numCells do
			-- for each cell in the current layer
			local cellOutput = cells[j].signal
			if i ~= numLayers then
				local weightDelta = 0
				local nextLayerCells = self[i+1].cells
				for k = 1, #nextLayerCells do
					-- for each cell in the next layer
					weightDelta = weightDelta + nextLayerCells[k].weights[j] * nextLayerCells[k].delta
					-- ensure weightDelta does not become inf
					assert(weightDelta ~= math.huge, "weightDelta: INF!\nnextLayerCells["..k.."].weights["..j.."]="..nextLayerCells[k].weights[j]..", \nnextLayerCells["..k.."].delta="..nextLayerCells[k].delta)
				end
				cells[j].delta = cellOutput * (1 - cellOutput) * weightDelta
				-- ensure cell delta does not become NaN
				assert(cells[j].delta == cells[j].delta, "cells["..j.."].delta: NaN!\n" .. "cellOutput="..cellOutput..",\nweightDelta="..weightDelta)
			else --special calculations for output layer
				cells[j].delta = (expectedOutputs[j] - cellOutput) * cellOutput * (1 - cellOutput)
				-- ensure cell delta does not become NaN
				assert(cells[j].delta == cells[j].delta, "cells["..j.."].delta: NaN! OUTPUT\nexpectedOutputs["..j.."]="..expectedOutputs[j]..",\ncellOutput="..cellOutput)
			end
		end
	end
	for i = 2, numLayers do
		-- update the bias
		self[i].bias = self[i].bias + learningRate * self[i].cells[#self[i].cells].delta
		for j = 1, #self[i].cells do
			for k = 1, #self[i].cells[j].weights do
				-- update the weights
				local weights = self[i].cells[j].weights
				weights[k] = weights[k] + learningRate * self[i].cells[j].delta  * self[i-1].cells[k].signal
			end
		end
	end
end

--// luann:saveNetwork(network, savefile)
-- network -> The luann network to be saved
-- path -> The path of the output file
function luann:saveNetwork(network, path)
	print("Saving network to: " .. path)
	persistence.store(path, network)
end

--// luann:loadNetwork(path)
-- path -> The path of the file to load
function luann:loadNetwork(path)
	local ann = persistence.load(path)
	ann.bp = luann.bp
	ann.activate = luann.activate

	for i = 1, #ann do
		for j = 1, #ann[i].cells do
			ann[i].cells[j].activate = Cell.activate
		end
	end
	return(ann)
end

--// luann:loadTrainingDataFromFile(path)
-- path -> The path of the file containing the training data
function luann:loadTrainingDataFromFile(path)
	local trainingData = {}
	local fileLines = {}
	local f = io.open(path, "rb")
	for line in f:lines() do
		table.insert (fileLines, line);
	end
	f:close()
	for i = 1, #fileLines do
		if i%2 == 0 then
			local tempInputs = {}
			for input in fileLines[i]:gmatch("%S+") do
				table.insert(tempInputs, tonumber(input))
			end
			local tempOutputs = {}
			for output in fileLines[i+1]:gmatch("%S+") do
				table.insert(tempOutputs, tonumber(input))
			end
			table.insert(trainingData, {tempInputs, tempOutputs})
		end
	end
	return(trainingData)
end


return(luann)
