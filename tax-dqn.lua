--[[

    DQN Tax 

]]--

cmd = torch.CmdLine()
cmd:text()
cmd:text('ADP Tax')
cmd:text()
cmd:text('Options')

-- general options:
cmd:option('-seed', 1, 'initial random seed')
cmd:option('-threads', 1, 'number of threads')

-- gpu
cmd:option('-cuda', 0, 'cuda')

-- model
cmd:option('-a1_size', 101, 'action space size')
cmd:option('-a2_size', 2, 'action space size')
cmd:option('-state_size', 22, 'state space size')

cmd:option('-gamma', -1, 'discount factor')
cmd:option('-eps_start', 0.5, 'start epsilon-greedy policy')
cmd:option('-eps_end', 0.1, 'final epsilon-greedy policy')
cmd:option('-eps_endt', 5000, 'final epsilon-greedy policy episode') -- 5000
cmd:option('-learn_start', 1, 'start learning episode')

cmd:option('-double', 1, 'double q learning')
cmd:option('-replay_memory', 1e+6, 'experience replay size')
-- cmd:option('-replay_priority', 1, 'prioritised experience replay')
cmd:option('-indy', 1, 'Independent Q-Learning')
cmd:option('-action_gap', 0, 'increase the action gap')
cmd:option('-action_gap_alpha', 0.5, 'action gap alpha parameter')
cmd:option('-target_step', 250*10, 'target network updates')
cmd:option('-reward_clip', -1, 'reward clip')

cmd:option('-S', 100.0, 'agents annual income') --@ DCV changed it to 1 from 100 to see if scaling helps
cmd:option('-r', 0.24, 'The tax rate')
cmd:option('-b', 0.24, 'The tax penalty coefficient')
cmd:option('-k', 2, 'Risk aversion coefficient')
cmd:option('-paudit', 0.05, 'the audit probability')
cmd:option('-rint', 0.03, 'interest rate')
cmd:option('-closureprob', 0.2, 'probability that closure will be available')
cmd:option('-closureperi', 0, 'probability that closure will be available')

-- training
cmd:option('-bs', 100, 'batch size')
cmd:option('-nepisodes', 50000, 'number of episodes')
cmd:option('-nsteps', 250, 'number of steps')

cmd:option('-learningRate', 1e-4, 'learning rate')
cmd:option('-step', 100, 'print every')

cmd:option('-load', 0, 'load pretrained model')
cmd:option('-save_strategy', 1, 'save strategy')


cmd:option('-v', 0, 'be verbose')
cmd:text()

opt = cmd:parse(arg)

-- Additional params
opt.t = 0.023 -- * opt.S -- The closure per year cost
opt.pau = (opt.paudit / 5) / 4
opt.gamma = 1 / (1 + opt.rint) -- The RL discount factor
opt.rew_norm = 1 -- math.abs((1 + opt.S) ^ (1 - opt.k)  / (1 - opt.k))

opt.closureperi_count = 1

require 'nngraph'
require 'nn'
require 'optim'
require 'csvigo'
require 'module.rmsprop'
require 'include.base'
local log = require 'include.log'

-- Cuda initialisation
if opt.cuda == 1 then
    require 'cutorch'
    require 'cunn'
    require 'cudnn'
    cutorch.setDevice(1)
    opt.dtype = 'torch.CudaTensor'
    print(cutorch.getDeviceProperties(1))
else
    opt.dtype = 'torch.FloatTensor'
end
print(opt)

-- Set float as default type
torch.manualSeed(opt.seed)
torch.setnumthreads(opt.threads)
torch.setdefaulttensortype('torch.FloatTensor')

-- Functions
function table.val_to_str ( v )
  if "string" == type( v ) then
    v = string.gsub( v, "\n", "\\n" )
    if string.match( string.gsub(v,"[^'\"]",""), '^"+$' ) then
      return "'" .. v .. "'"
    end
    return '"' .. string.gsub(v,'"', '\\"' ) .. '"'
  else
    return "table" == type( v ) and table.tostring( v ) or
      tostring( v )
  end
end

function table.key_to_str ( k )
  if "string" == type( k ) and string.match( k, "^[_%a][_%a%d]*$" ) then
    return k
  else
    return "[" .. table.val_to_str( k ) .. "]"
  end
end

function table.tostring( tbl )
  local result, done = {}, {}
  for k, v in ipairs( tbl ) do
    table.insert( result, table.val_to_str( v ) )
    done[ k ] = true
  end
  for k, v in pairs( tbl ) do
    if not done[ k ] then
      table.insert( result,
        table.key_to_str( k ) .. "=" .. table.val_to_str( v ) )
    end
  end
  return "{" .. table.concat( result, "," ) .. "}"
end

function isnan(x) return x ~= x end

Mno = torch.Tensor({
    { opt.pau, opt.pau, opt.pau, opt.pau, opt.pau, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, opt.pau, opt.pau, opt.pau, opt.pau, opt.pau, opt.pau, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, opt.pau, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, opt.pau, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4 * opt.paudit / 5, 4 * opt.paudit / 5 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1 - opt.pau, 1 - opt.pau, 1 - opt.pau, 1 - opt.pau, 1 - opt.pau, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 1 - opt.pau, 1 - opt.pau, 1 - opt.pau, 1 - opt.pau, 1 - opt.pau, 1 - opt.pau, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 - opt.pau, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 - opt.pau, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 - 4 * opt.paudit / 5, 1 - 4 * opt.paudit / 5 }
})

-- Transition matrix with option available & firm accepts
Ma = torch.Tensor({
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }
})

-- Transition matrix with option available & firm declines
Md = torch.Tensor({
    { 3 * opt.pau, 3 * opt.pau, 3 * opt.pau, 3 * opt.pau, 3 * opt.pau, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 3 * opt.pau, 3 * opt.pau, 3 * opt.pau, 3 * opt.pau, 3 * opt.pau, 3 * opt.pau, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 * opt.pau, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 * opt.pau, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3 * (4 * opt.paudit / 5), 3 * (4 * opt.paudit / 5) },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 1 - 3 * opt.pau, 1 - 3 * opt.pau, 3 * opt.pau, 1 - 3 * opt.pau, 1 - 3 * opt.pau, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 1 - 3 * opt.pau, 1 - 3 * opt.pau, 1 - 3 * opt.pau, 1 - 3 * opt.pau, 1 - 3 * opt.pau, 1 - 3 * opt.pau, 0, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 - 3 * opt.pau, 0, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 - 3 * opt.pau, 0, 0 },
    { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 - 3 * (4 * opt.paudit / 5), 1 - 3 * (4 * opt.paudit / 5) }
})

-- The matrix with history dynamics
H = torch.Tensor({
    { 0, 1, 0, 0, 0 },
    { 0, 0, 1, 0, 0 },
    { 0, 0, 0, 1, 0 },
    { 0, 0, 0, 0, 1 },
    { 0, 0, 0, 0, 0 }
})


function nextstate(x0, u, closureprob)

    -- This function determines the next state of 
    -- the markov model x(k+1)=Ax(k)+Bu(k)
    -- given: the current state x0
    --           the stochastic availability of closure option c
    --           the tax evasion as percentage of profits u1
    --           the decision to use or not the closure option u2

    local u0 = torch.Tensor({ (u[1] - 1) / 100, u[2] - 1 })
    x0 = x0:squeeze()

    if closureprob == nil then
        if opt.closureperi > 0 then
            opt.closureperi_count = (opt.closureperi_count + 1) % opt.closureperi

            if opt.closureperi_count == 0 then
                closureprob = 0
            else
                closureprob = 1
            end
        else 
            closureprob = torch.uniform()
        end
    end

    local A = torch.zeros(opt.state_size, opt.state_size)

    -- if closure is not available
    local c1 = 0
    if closureprob <= opt.closureprob then
        c1 = 1 -- the agent can employ it
    end

    -- If closure available
    if x0[21] == 1 then
        if u0[2] == 0 then --do not use the closure option
            A[{ { 1, 15 }, { 1, 15 } }] = Md
        else
            A[{ { 1, 15 }, { 1, 15 } }] = Ma -- use closure option
        end
    else
        A[{ { 1, 15 }, { 1, 15 } }] = Mno -- closure not available
    end
    A[{ { 16, 20 }, { 16, 20 } }] = H
    local B = torch.zeros(opt.state_size, 2) -- The matrix of the dynamics of control, closure availability

    B[{ { 20 }, { 1 } }] = 1
    B[{ { 21 }, { 2 } }] = 1

    local x
    -- if closure is available
    if x0[21] == 1 then
        -- and the agent choses to use it
        if u0[2] == 1 then
            u0[2] = c1
            x = (A * x0:view(opt.state_size, 1) + B * u0:view(2, 1)):squeeze() -- this is the next state
        else
            -- if the agent does not use the option he can either be audited or not
            u0[2] = c1
            x = (A * x0:view(opt.state_size, 1) + B * u0:view(2, 1)):squeeze()
            local rows = tools.find(x[{ { 1, 15 } }]:ne(0))
            local row1 = x[rows[1]]
            local randstate = 0 + (1 - 0) * torch.uniform() -- the system randomly choses the next state according to the decision of not employing the closure
            if randstate <= row1 then
                x[rows[1]] = 1
                x[rows[2]] = 0
            else
                x[rows[1]] = 0
                x[rows[2]] = 1
            end
        end

    else
        -- if closure is not available
        u0[2] = c1
        x = (A * x0:view(opt.state_size, 1) + B * u0:view(2, 1)):squeeze() -- estimate next state
        local rows = tools.find(x[{ { 1, 15 } }]:ne(0)) -- locate the non zero transitions
        local row1 = x[rows[1]]
        local randstate = 0 + (1 - 0) * torch.uniform() -- randomly chose one
        if randstate <= row1 then
            x[rows[1]] = 1
            x[rows[2]] = 0
        else
            x[rows[1]] = 0
            x[rows[2]] = 1
        end
    end

    -- x[22] = x0[22] + 1 / opt.nsteps

    return x:view(1, opt.state_size)
end

function reward(x, u)
    -- This function estimates the reward of being in a state x, applying action u
    local u0 = { (u[1] - 1) / 100, u[2] - 1 }

    local Smin = opt.S
    local auditdiscount = 3.0 / 5.0

    local S = opt.S
    local r = opt.r
    local b = opt.b
    local k = opt.k
    local t = opt.t

    -- calculate cost/penalty for prior tax evasion, or cost of closure
    local frew = torch.zeros(15)
    local pi = x[{ { 1 }, { 1, 15 } }]:squeeze() -- the vector that shows at which state the agent is
    local ui = x[{ { 1 }, { 16, 20 } }]:squeeze() -- the vector with the decision history
    --print(pi)
    --print(ui)

    frew[1] = -auditdiscount * b * S * r * ui[5] - S * r * ui[5] -- cost when in audit state with 1 past decision
    frew[2] = -auditdiscount * b * S * r * (ui[5] + 2 * ui[4]) - S * r * (ui[5] + ui[4])
    frew[3] = -auditdiscount * b * S * r * (ui[5] + 2 * ui[4] + 3 * ui[3]) - S * r * (ui[5] + ui[4] + ui[3])
    frew[4] = -auditdiscount * b * S * r * (ui[5] + 2 * ui[4] + 3 * ui[3] + 4 * ui[2]) - S * r * (ui[5] + ui[4] + ui[3] + ui[2])
    frew[5] = -auditdiscount * b * S * r * (ui[5] + 2 * ui[4] + 3 * ui[3] + 4 * ui[2] + 5 * ui[1]) - S * r * (ui[5] + ui[4] + ui[3] + ui[2] + ui[1])
    frew[6] = -Smin * t -- cost when chosing closure when available
    frew[7] = -2 * Smin * t
    frew[8] = -3 * Smin * t
    frew[9] = -4 * Smin * t
    frew[10] = -5 * Smin * t
    frew[11] = 0 -- cost when no audit or closure occur
    frew[12] = 0
    frew[13] = 0
    frew[14] = 0
    frew[15] = 0
    -- srew
    local I = (pi:view(15, 1):t() * frew:view(15, 1)):squeeze()


    local rew1 = (S * (1 - r + u0[1] * r) + I)

    local rew = rew1 ^ (1 - k) / (1 - k)

    if k > 0 then
        if isnan(rew) or rew1 < 0 or rew == -math.huge then
            rew = -1
        end
    end

    if rew == math.huge then
        rew = 10000
    end

    return rew
end

function save_model()
    model:clearState()
    model_target:clearState()
    -- save/log current net
    local filename = paths.concat(paths.cwd(), 'model/tax.t7')
    os.execute('mkdir -p ' .. paths.dirname(filename))
    if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    torch.save(filename, { model, model_target, replay, opt, optim_config, optim_state, train_rew, test_rew })
end

function load_model()
    model, model_target, replay, opt, optim_config, optim_state, train_rew, test_rew = unpack(torch.load('model/tax.t7'))
end

function create_network()
    local dim = { 256, 256, 256 }

    local model = nn.Sequential()
    model:add(nn.View(-1, opt.state_size))
    model:add(nn.AddConstant(-0.5))
    model:add(nn.Linear(opt.state_size, dim[1]))
    model:add(nn.ReLU(true))
    model:add(nn.Linear(dim[1], dim[2]))
    model:add(nn.ReLU(true))
    model:add(nn.Linear(dim[2], dim[3]))
    model:add(nn.ReLU(true))
    model:add(nn.Linear(dim[3], opt.a1_size + opt.a2_size))


    if opt.dtype == 'torch.FloatTensor' then
        return model
    else
        model = model:type(opt.dtype)
        return cudnn.convert(model, cudnn)
    end
end

if opt.load == 1 then
    log.info("Loading Neural Net.")
    load_model()
else
    log.info("Creating Neural Net.")
    -- Create network
    model = create_network()

    -- Create target network
    model_target = model:clone()


    rew_bn = nn.Sequential()
        :add(nn.View(-1,1))
        :add(nn.BatchNormalization(1))
        :add(nn.View(-1))

    -- SGD settings
    optim_config = { 
        learningRate = opt.learningRate,
        -- momentum = 0.99,
        -- epsilon = 1e-1
    }
    optim_state = {}
end

-- Get paramters
params, gradParams = model:getParameters()
params_target, _ = model_target:getParameters()

function test(model)
    -- For each episode, start with the same initial state
    local x_0 = torch.zeros(1, opt.state_size) -- The inital state with initial control
    x_0[{ { 1 }, { 1 } }] = 1 -- start from "audit" state
    -- if torch.uniform() <= opt.closureprob then
    --     x_0[{ { 1 }, { 21 } }] = 1
    -- end

    -- Run for N steps
    local x_t, x_t1, a, r
    local r_total = 0

    -- Actions
    local strategy = torch.zeros(opt.nsteps, opt.state_size + 2)

    -- Initial state
    x_t = x_0:clone()
    model:evaluate()
    for i = 1, opt.nsteps do

        -- get argmax_u Q from DQN
        local q = model:forward(x_t:type(opt.dtype)):clone()

        -- Pick an action (epsilon-greedy)
        local max_q, max_a1 = torch.max(q[{ {}, { 1, opt.a1_size } }], 2)
        local a1 = max_a1:squeeze()

        local a2
        if x_t[1][21] == 1 then
            local max_q, max_a2 = torch.max(q[{ {}, { opt.a1_size + 1, opt.a1_size + opt.a2_size } }], 2)
            a2 = max_a2:squeeze()
        else
            a2 = 1
        end

        -- Store actions
        strategy[{{i}, {1, opt.state_size}}] = x_t
        strategy[{{i}, {opt.state_size+1}}] = a1
        strategy[{{i}, {opt.state_size+2}}] = a2

        --compute reward for current state-action pair
        local r = reward(x_t, { a1, a2 }) * opt.rew_norm
        r_total = r_total + r * opt.gamma ^ (i - 1)

        -- Transition to the next Markov state using the last generated action
        x_t1 = nextstate(x_t, { a1, a2 })

        -- Transition to next s
        x_t:copy(x_t1)
    end

    return r_total, strategy
end


function test_random_mean_std()
    -- For each episode, start with the same initial state
    local x_0 = torch.zeros(1, opt.state_size) -- The inital state with initial control
    x_0[{ { 1 }, { 1 } }] = 1 -- start from "audit" state

    -- Run for N steps
    local x_t, x_t1, a, r
    local r_total = 0
    local r_mean_std = {}

    -- Initial state
    x_t = x_0:clone()

    for j = 1, 10000 do
        for i = 1, opt.nsteps do

            -- Pick an action (epsilon-greedy)
            local a1 = torch.random(opt.a1_size)
            local a2
            if x_t[1][21] == 1 then
                a2 = torch.random(opt.a2_size)
            else
                a2 = 1
            end

            --compute reward for current state-action pair
            local r = reward(x_t, { a1, a2 }) * opt.rew_norm            
            r_total = r_total + r * opt.gamma ^ (i - 1)
            r = math.min(r, 10000)
            r = math.max(r, -1)
            table.insert(r_mean_std, r)

            -- Transition to the next Markov state using the last generated action
            x_t1 = nextstate(x_t, { a1, a2 })

            -- Transition to next s
            x_t:copy(x_t1)
        end
    end

    r_mean_std = torch.Tensor(r_mean_std)

    return r_mean_std:mean(), r_mean_std:std()
end

function test_random()
    -- For each episode, start with the same initial state
    local x_0 = torch.zeros(1, opt.state_size) -- The inital state with initial control
    x_0[{ { 1 }, { 1 } }] = 1 -- start from "audit" state

    -- Run for N steps
    local x_t, x_t1, a, r
    local r_total = 0

    -- Initial state
    x_t = x_0:clone()

    for i = 1, opt.nsteps do

        -- Pick an action (epsilon-greedy)
        local a1 = torch.random(opt.a1_size)
        local a2
        if x_t[1][21] == 1 then
            a2 = torch.random(opt.a2_size)
        else
            a2 = 1
        end

        --compute reward for current state-action pair
        local r = reward(x_t, { a1, a2 }) * opt.rew_norm
        r_total = r_total + r * opt.gamma ^ (i - 1)

        -- Transition to the next Markov state using the last generated action
        x_t1 = nextstate(x_t, { a1, a2 })

        -- Transition to next s
        x_t:copy(x_t1)
    end

    return r_total
end


function test_optimal(a1_opt)

    local r_mean_std = {}
    local r_totals = {}

    for e = 1, 100 do

        -- For each episode, start with the same initial state
        local x_0 = torch.zeros(1, opt.state_size) -- The inital state with initial control
        x_0[{ { 1 }, { 1 } }] = 1 -- start from "audit" state

        -- Run for N steps
        local x_t, x_t1, a, r
        local r_total = 0

        if a1_opt == nil then
            a1_opt = opt.a1_size
        end

        -- Initial state
        x_t = x_0:clone()

        for i = 1, opt.nsteps do

            -- Pick an action (epsilon-greedy)
            local a1 = a1_opt
            local a2
            if x_t[1][21] == 1 then
                a2 = 2
            else
                a2 = 1
            end

            --compute reward for current state-action pair
            local r = reward(x_t, { a1, a2 }) * opt.rew_norm
            r_total = r_total + r * opt.gamma ^ (i - 1)
            -- r = math.min(r, 100)
            -- r = math.max(r, -100)
            table.insert(r_mean_std, r)

            -- Transition to the next Markov state using the last generated action
            x_t1 = nextstate(x_t, { a1, a2 })

            -- Transition to next s
            x_t:copy(x_t1)
        end

        table.insert(r_totals, r_total)

    end

    r_mean_std = torch.Tensor(r_mean_std)

    return torch.Tensor(r_totals):mean(), r_mean_std:mean(), r_mean_std:std()

end


local r_mean, r_std = test_random_mean_std()
-- local r_mean = 0
-- local r_std = 1
log.infof('R mean=%.12f, std=%.12f', r_mean, r_std)

-- Compute baseline
local test_rand = 0
log.info('Computing baseline')

for i = 1, 100 do
    local r = test_random() / 100
    test_rand = test_rand + r
end
log.infof('%d, %.1f, Random baseline=%.12f', opt.k, opt.closureprob, test_rand)
-- print(opt.i_min, opt.i_max)
-- exit()

local test_opt
local max_test_opt = -1e+6
local max_test_opt_u1 = 0
local r, r_mean_loc, r_std_loc
for a1 = 1, 101, 1 do
    test_opt, r_mean_loc, r_std_loc = test_optimal(a1)
    if test_opt > max_test_opt then
        -- r_mean = r_mean_loc
        -- r_std = r_std_loc
        max_test_opt = test_opt
        max_test_opt_u1 = a1
    end
end
log.infof('%d, %.1f, Optimal %d baseline=%.12f', opt.k, opt.closureprob, max_test_opt_u1, max_test_opt)
log.infof('%d, %.1f, R mean=%.12f, std=%.12f', opt.k, opt.closureprob, r_mean, r_std)

-- os.exit()

-- Initialise aux vectors
local td_err = torch.Tensor(opt.bs, opt.a1_size + opt.a2_size):type(opt.dtype)

local train_r_episode = torch.zeros(opt.nsteps)
local train_q_episode = torch.zeros(opt.nsteps)

local train_q = 0
local train_q_avg = 0

local train_r = 0
local train_r_avg = test_rand

local test_r = 0
local test_r_avg = test_rand

local step_count = 0
local replay = {}
local log_str = ''

local model_best = model:clone()
local model_best_score = -math.huge

-- Opens a file in append mode
local file_log_path
if opt.closureperi > 0 then
    file_log_path = paths.concat('out', 'strategy_closure_' .. opt.closureperi .. '_k_' .. opt.k .. '_d_' .. opt.double .. '.txt')
else
    file_log_path = paths.concat('out', 'strategy_closure_' .. opt.closureprob .. '_k_' .. opt.k .. '_d_' .. opt.double .. '.txt')
end
local file_log = io.open(file_log_path, "w")


local train = {
    s_t = torch.Tensor(opt.bs, opt.state_size):type(opt.dtype),
    s_t1 = torch.Tensor(opt.bs, opt.state_size):type(opt.dtype),
    r_t = torch.Tensor(opt.bs):type(opt.dtype),
    a1_t = torch.Tensor(opt.bs):type(opt.dtype),
    a2_t = torch.Tensor(opt.bs):type(opt.dtype),
    terminal = torch.Tensor(opt.bs):type(opt.dtype)
}

-- start time
local beginning_time = torch.tic()

local r_mean_std = {}

for e = 1, opt.nepisodes do

    -- Episode and training values storage
    local episode = {}
    episode.s_t = torch.zeros(1, opt.state_size) -- The inital state with initial control
    episode.s_t[{ { 1 }, { 1 } }] = 1 -- start from "audit" state

    episode.terminal = false

    -- Initialise clock
    local time = sys.clock()

    -- Epsilon annealing
    opt.eps = (opt.eps_end +
                math.max(0, (opt.eps_start - opt.eps_end) * (opt.eps_endt -
                math.max(0, e - opt.learn_start)) / opt.eps_endt))

    -- Run for N steps
    model:training()
    rew_bn:training()
    local step = 1
    while step <= opt.nsteps and not episode.terminal do

        -- Compute Q values
        local q = model:forward(episode.s_t:type(opt.dtype)):clone()

        -- Pick an action 1 (epsilon-greedy)
        if torch.uniform() < opt.eps then
            episode.a1_t = torch.random(opt.a1_size)
        else
            local max_q, max_a1 = torch.max(q[{ {}, { 1, opt.a1_size } }], 2)
            episode.a1_t = max_a1:squeeze()
        end

        -- Pick an action 2 (epsilon-greedy)
        if torch.uniform() < opt.eps then
            if episode.s_t[1][21] == 1 then
                episode.a2_t = torch.random(opt.a2_size)
            else
                episode.a2_t = 1
            end
        else
            if episode.s_t[1][21] == 1 then
                local max_q, max_a2 = torch.max(q[{ {}, { opt.a1_size + 1, opt.a1_size + opt.a2_size } }], 2)
                episode.a2_t = max_a2:squeeze()
            else
                episode.a2_t = 1
            end
        end

        --compute reward for current state-action pair
        episode.r_t = reward(episode.s_t, { episode.a1_t, episode.a2_t })
        episode.terminal = step == opt.nsteps

        -- Transition to the next Markov state using the last generated action
        episode.s_t1 = nextstate(episode.s_t, { episode.a1_t, episode.a2_t })

        -- Store rewards
        train_r_episode[step] = episode.r_t * opt.gamma ^ (step - 1) * opt.rew_norm

        -- Reward clipping
        episode.r_t = math.min(episode.r_t, 10000)
        episode.r_t = math.max(episode.r_t, -1)
        table.insert(r_mean_std, episode.r_t)

        -- Store current step
        local r_id = (step_count % opt.replay_memory) + 1
        replay[r_id] = {
            r_t = episode.r_t,
            a1_t = episode.a1_t,
            a2_t = episode.a2_t,
            s_t = episode.s_t:clone(),
            s_t1 = episode.s_t1:clone(),
            terminal = episode.terminal and 1 or 0
        }


        -- Fetch from experiences
        local q_next, q_next_max1, q_next_max_a1, q_next_max2, q_next_max_a2, q_next_max
        if #replay >= opt.bs then

            for b = 1, opt.bs do
                local exp_id = torch.random(#replay)
                train.r_t[b] = replay[exp_id].r_t
                train.a1_t[b] = replay[exp_id].a1_t
                train.a2_t[b] = replay[exp_id].a2_t
                train.s_t[b] = replay[exp_id].s_t
                train.s_t1[b] = replay[exp_id].s_t1
                train.terminal[b] = replay[exp_id].terminal
            end

            train.r_t = (train.r_t - r_mean):div(r_std)

            -- Use target network to predict q_max
            if opt.double == 1 then
                q_next = model:forward(train.s_t1):clone()

                q_next_max1, q_next_max_a1 = torch.max(q_next[{ {}, { 1, opt.a1_size } }], 2)
                q_next_max1 = q_next_max1:squeeze()
                q_next_max_a1 = q_next_max_a1:squeeze()

                q_next_max2, q_next_max_a2 = torch.max(q_next[{ {}, { opt.a1_size + 1, opt.a1_size + opt.a2_size } }], 2)
                q_next_max2 = q_next_max2:squeeze()
                q_next_max_a2 = q_next_max_a2:squeeze()
                
                q_next = model_target:forward(train.s_t1):clone()
                for b = 1, opt.bs do
                    q_next_max1[b] = q_next[{ { b }, { q_next_max_a1[b] } }]
                    if train.s_t1[b][21] == 1 then
                        q_next_max2[b] = q_next[{ { b }, { opt.a1_size + q_next_max_a2[b] } }]
                    else
                        q_next_max2[b] = q_next[{ { b }, { opt.a1_size + 1 } }]
                    end
                end
                q_next_max = (q_next_max1 + q_next_max2) / 2
            else
                q_next = model_target:forward(train.s_t1):clone()
                q_next_max1 = q_next[{ {}, { 1, opt.a1_size } }]:max(2):squeeze(2)
                q_next_max2 = q_next_max1.new():resize(opt.bs)
                for b = 1, opt.bs do
                    q_next_max2[b] = q_next[{ { b }, { opt.a1_size + 1, opt.a1_size + 1 + train.s_t1[b][21] } }]:max()
                end
                q_next_max = (q_next_max1 + q_next_max2) / 2
            end

            -- Compute Q
            q = model:forward(train.s_t):clone()

            -- Q learnt value
            td_err:zero()
            for b = 1, opt.bs do
                if opt.indy == 1 then
                    -- Independent Q-Learning
                    td_err[{ { b }, { train.a1_t[b] } }] = train.r_t[b] + q_next_max1[b] * opt.gamma - q[b][train.a1_t[b]]
                    if train.s_t[b][21] == 1 then
                        td_err[{ { b }, { opt.a1_size + train.a2_t[b] } }] = train.r_t[b] + q_next_max2[b] * opt.gamma - q[b][opt.a1_size + train.a2_t[b]]
                    end
                else
                    -- Average Q-Learning (http://arxiv.org/abs/1506.08941)
                    td_err[{ { b }, { train.a1_t[b] } }] = train.r_t[b] + q_next_max[b] * opt.gamma - q[b][train.a1_t[b]]
                    if train.s_t[b][21] == 1 then
                        td_err[{ { b }, { opt.a1_size + train.a2_t[b] } }] = train.r_t[b] + q_next_max[b] * opt.gamma - q[b][opt.a1_size + train.a2_t[b]]
                    end
                end
            end

            -- Increase the action gap
            if opt.action_gap == 1 then
                local q_target = model_target:forward(train.s_t):clone()
                local V_s_1, V_s_2, V_s_1_1, V_s_1_2
                V_s_1 = q_target[{ {}, { 1, opt.a1_size } }]:max(2):squeeze(2)
                V_s_2 = q_target[{ {}, { opt.a1_size + 1, opt.a1_size + opt.a2_size } }]:max(2):squeeze(2)
                V_s_1_1 = q_next[{ {}, { 1, opt.a1_size } }]:max(2):squeeze(2)
                V_s_1_2 = q_next[{ {}, { opt.a1_size + 1, opt.a1_size + opt.a2_size } }]:max(2):squeeze(2)
                for b = 1, opt.bs do
                    -- Advantage Learning (AL) operator
                    local Q_s_a = q_target[b][train.a1_t[b]]
                    local AL = -opt.action_gap_alpha * (V_s_1[b] - Q_s_a)

                    -- Persistent Advantage Learning (PAL) operator
                    local Q_s_1_a = q_next[b][train.a1_t[b]]
                    local PAL = -opt.action_gap_alpha * (V_s_1_1[b] - Q_s_1_a)

                    td_err[{ { b }, { train.a1_t[b] } }]:add(math.max(AL, PAL))


                    -- Advantage Learning (AL) operator
                    local Q_s_a = q_target[b][opt.a1_size + train.a2_t[b]]
                    local AL = -opt.action_gap_alpha * (V_s_2[b] - Q_s_a)

                    -- Persistent Advantage Learning (PAL) operator
                    local Q_s_1_a = q_next[b][opt.a1_size + train.a2_t[b]]
                    local PAL = -opt.action_gap_alpha * (V_s_1_2[b] - Q_s_1_a)

                    td_err[{ { b }, { opt.a1_size + train.a2_t[b] } }]:add(math.max(AL, PAL))
                end
            end

            -- Backward pass
            local feval = function(x)

                -- Reset parameters
                gradParams:zero()

                -- Stats
                train_q_episode[step] = 0.5 * td_err:clone():pow(2):sum()
                
                -- Backprop
                model:backward(train.s_t, -td_err)

                -- Normalise
                gradParams:div(opt.bs)

                return 0, gradParams
            end

            -- optim.rmspropm(feval, params, optim_config, optim_state)
            optim.adam(feval, params, optim_config, optim_state)

            -- Update target network
            if step_count % opt.target_step == 0 then
                params_target:copy(params)
            end
        end

        -- next state
        episode.s_t = episode.s_t1:clone()
        step = step + 1

        -- Total steps
        step_count = step_count + 1
    end

    -- Compute statistics
    local train_q = train_q_episode:narrow(1, 1, step - 1):mean()
    local train_r = train_r_episode:narrow(1, 1, step - 1):sum()
    local test_r, test_strategy
    if e == 1 or e % 100 == 0 then
        local r
        test_r = 0
        test_strategy = {}
        for i = 1, 100 do
            r, test_strategy[i] = test(model)
            test_r = test_r + r  / 100
        end
        test_r_avg = 0.9 * test_r_avg + 0.1 * test_r

        if test_r > model_best_score then --  * 0.99
            model_best_score = test_r
            model_best = model:clone()

            -- Save strategy
            if opt.save_strategy == 1 then
                for i = 1, #test_strategy do
                    local filename
                    if opt.closureperi > 0 then
                        filename = paths.concat('out', 'strategy_closure_' .. opt.closureperi .. '_k_' .. opt.k .. '_d_' .. opt.double .. '_' .. i .. '.csv')
                    else
                        filename = paths.concat('out', 'strategy_closure_' .. opt.closureprob .. '_k_' .. opt.k .. '_d_' .. opt.double .. '_' .. i .. '.csv')
                    end
                    csvigo.save({
                        path = filename,
                        data = test_strategy[i]:totable(),
                        verbose = false
                    })
                end
            end
        end
    end

    -- Compute moving averages
    if e == 1 then
        train_q_avg = train_q
        train_r_avg = 0.99 * train_r_avg + 0.01 * train_r
    else
        train_q_avg = 0.99 * train_q_avg + 0.01 * train_q
        train_r_avg = 0.99 * train_r_avg + 0.01 * train_r
    end

    -- Print statistics
    if e == 1 or e % opt.step == 0 then

        log_str = string.format('e=%d, tr_q=%.12f, tr_q_avg=%.12f, tr_r=%.12f, te_r=%.12f, opt=%.12f, cur=%.12f, params=%.12f, t/e=%.2f sec, t=%d min.\n',
            e, train_q, train_q_avg, train_r, test_r, max_test_opt, model_best_score, params:norm(),
            sys.clock() - time, torch.toc(beginning_time) / 60)
        
        log.info(log_str)
        file_log:write(log_str)
        file_log:flush()

    end
end

log.info(string.format('R mean=%.12f, std=%.12f', r_mean, r_std))
file_log:write(string.format('R mean=%.12f, std=%.12f', r_mean, r_std))
log.info(string.format('Random baseline=%.12f', test_rand))
file_log:write(string.format('Random baseline=%.12f', test_rand))
log.info(string.format('Optimal %d baseline=%.12f', max_test_opt_u1, max_test_opt))
file_log:write(string.format('Optimal %d baseline=%.12f', max_test_opt_u1, max_test_opt))


log.info(string.format('Current baseline=%.12f', model_best_score))
file_log:write(string.format('Current baseline=%.12f', model_best_score))
file_log:close()


local filename
if opt.closureperi > 0 then
    filename = paths.concat('model', 'model_closure_' .. opt.closureperi .. '_k_' .. opt.k .. '_d_' .. opt.double .. '.t7')
else
    filename = paths.concat('model', 'model_closure_' .. opt.closureprob .. '_k_' .. opt.k .. '_d_' .. opt.double .. '.t7')
end
torch.save(filename, {model_best, model_best_score})
