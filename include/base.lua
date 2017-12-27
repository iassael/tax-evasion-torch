tools = {}

function tools.find(mask)
    local idx = torch.linspace(1,mask:size(1),mask:size(1))
    -- local idx_zero = x:eq(0)
    -- local idx_nonzero = x:ne(0)
    return idx[mask]
end

function tools.f2(f)
    return string.format("%.2f", f)
end

function tools.f3(f)
    return string.format("%.3f", f)
end

function tools.f4(f)
    return string.format("%.4f", f)
end

function tools.f5(f)
    return string.format("%.5f", f)
end

function tools.f6(f)
    return string.format("%.6f", f)
end

function tools.d(f)
    return string.format("%d", torch.round(f))
end