execute(ex) = execute(ex, DefaultAlgebra())
function register(algebra)
    Base.eval(Finch, quote
        @generated function execute(ex, a::$algebra)
            execute_code(:ex, ex, a())
        end
    end)
end

function execute_code(ex, T, algebra = DefaultAlgebra())
    prgm = nothing
    code = contain(LowerJulia(algebra = algebra)) do ctx
        prgm = virtualize(ex, T, ctx)
        prgm = TransformSSA(Freshen())(prgm)
        prgm = with(retuple(map(acc -> access(acc.tns, reader()), getresults(prgm))...), prgm)
        prgm = multi(Dimensionalize(pass()), prgm)
        ctx(prgm)
    end
    code = quote
        @inbounds begin
            $code
        end
    end
    code |>
        lower_caches |>
        lower_cleanup
end

macro finch(args_ex...)
    @assert length(args_ex) >= 1
    (args, ex) = (args_ex[1:end-1], args_ex[end])
    results = Set()
    prgm = IndexNotation.finch_parse_instance(ex, results)
    thunk = quote
        res = $execute($prgm, $(map(esc, args)...))
    end
    for tns in results
        push!(thunk.args, quote
            $(esc(tns)) = res.$tns
        end)
    end
    push!(thunk.args, quote
        res
    end)
    thunk
end

macro finch_code(args_ex...)
    @assert length(args_ex) >= 1
    (args, ex) = (args_ex[1:end-1], args_ex[end])
    prgm = IndexNotation.finch_parse_instance(ex)
    return quote
        $execute_code(:ex, typeof($prgm), $(map(esc, args)...)) |>
        striplines |>
        unblock |>
        unquote_literals
    end
end

"""
    Initialize(ctx)

A transformation to initialize tensors that have just entered into scope.

See also: [`initialize!`](@ref)
"""
@kwdef struct Initialize{Ctx}
    ctx::Ctx
    target=nothing
    escape=[]
end
initialize!(tns, ctx, mode, idxs...) = access(tns, mode, idxs...)
function (ctx::Initialize)(node)
    if istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end

function (ctx::Initialize)(node::IndexNode)
    if node.kind === access && node.tns isa IndexNode && node.tns.kind === virtual
        if (ctx.target === nothing || (getname(node.tns) in ctx.target)) && !(getname(node.tns) in ctx.escape)
            initialize!(node.tns.val, ctx.ctx, node.mode, map(ctx, node.idxs)...)
        else
            return access(node.tns, node.mode, map(ctx, node.idxs)...)
        end
    elseif node.kind === with
        ctx_2 = Initialize(ctx.ctx, ctx.target, union(ctx.escape, map(getname, getresults(node.prod))))
        with(ctx_2(node.cons), ctx_2(node.prod))
    elseif istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end

"""
    Finalize(ctx)

A transformation to finalize output tensors before they leave scope and are
returned to the caller.

See also: [`finalize!`](@ref)
"""
@kwdef struct Finalize{Ctx}
    ctx::Ctx
    target=nothing
    escape=[]
end
finalize!(tns, ctx, mode, idxs...) = tns
function (ctx::Finalize)(node)
    if istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end

function (ctx::Finalize)(node::IndexNode)
    if node.kind === access && node.tns isa IndexNode && node.tns.kind === virtual
        if (ctx.target === nothing || (getname(node.tns) in ctx.target)) && !(getname(node.tns) in ctx.escape)
            access(finalize!(node.tns.val, ctx.ctx, node.mode, node.idxs...), node.mode, node.idxs...)
        else
            access(node.tns, node.mode, map(ctx, node.idxs)...)
        end
    elseif node.kind === with
        ctx_2 = Finalize(ctx.ctx, ctx.target, union(ctx.escape, map(getname, getresults(node.prod))))
        with(ctx_2(node.cons), ctx_2(node.prod))
    elseif istree(node)
        return similarterm(node, operation(node), map(ctx, arguments(node)))
    else
        return node
    end
end