require 'torch'
require 'nn'

require 'LanguageModel'


local cmd = torch.CmdLine()
cmd:option('-checkpoint', 'cv/checkpoint_4000.t7')
cmd:option('-text', '')
cmd:option('-gpu', 0)
cmd:option('-gpu_backend', 'cuda')
cmd:option('-verbose', 0)
cmd:option('-ignore_newlines', 1)
local opt = cmd:parse(arg)


local checkpoint = torch.load(opt.checkpoint)
local model = checkpoint.model

local dtype = 'torch.FloatTensor'
local msg
if opt.gpu >= 0 and opt.gpu_backend == 'cuda' then
  require 'cutorch'
  require 'cunn'
  cutorch.setDevice(opt.gpu + 1)
  model:cuda()
  dtype = 'torch.CudaTensor'
  msg = string.format('Running with CUDA on GPU %d', opt.gpu)
elseif opt.gpu >= 0 and opt.gpu_backend == 'opencl' then
  require 'cltorch'
  require 'clnn'
  model:cl()
  dtype = torch.Tensor():cl():type()
  msg = string.format('Running with OpenCL on GPU %d', opt.gpu)
else
  msg = 'Running in CPU mode'
end
if opt.verbose == 1 then print(msg) end

local crit = nn.CrossEntropyCriterion():type(dtype)

local utils = require 'util.utils'


-- Terminal colours
function col(redness, msg)
  local val = 36*math.floor(redness * 5) + 16
  return '\x1b[48;5;' .. val .. 'm' .. msg .. '\x1b[0m'
end

function round(n)
  if n >= 0.5 then
    return math.ceil(n)
  else
    return math.floor(n)
  end
end


function linnt()
  model:resetStates()

  local w = model.net:get(1).weight
  local scores = w.new(1, 1, model.vocab_size):fill(1)

  for line in io.stdin:lines() do
    local text = line .. '\n'
    local textvec = model:encode_string(text):view(1, -1)
    local T = textvec:size(2)
    for t = 1, T do
       local c = textvec[{1,t}]
       local probs = scores:double():exp():squeeze()
       probs:div(torch.sum(probs))

       local loss = crit:forward(scores, c)
       local chr = text:sub(t, t)
       if opt.ignore_newlines == 1 and chr == '\n' then
        io.stdout:write(chr)
       else
        io.stdout:write(col(math.min(loss, 10)/10, chr))
        -- io.stdout:write(col(1 - probs[c], chr))
       end
      scores = model:forward(textvec:select(2, t):view(1,1))
    end
  end

  model:resetStates()
end

model:evaluate()

linnt()
