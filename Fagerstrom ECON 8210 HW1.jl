#Matthew J. Fagerstrom, 21 October 2024.
#This file executes all the computations for the first homework for ECON 8210.

#Remember to instantiate the packages!

#Import Packages
using Distributions, Interpolations, Ipopt, JuMP, LaTeXStrings, LinearAlgebra, LineSearches, NLsolve, NonlinearSolve, Optim, Plots, Random, Statistics

###Problem Two###
#This program computes and compares different quadrature methods to a Monte Carlo method.

#Set parameters
T = 100
ρ = 0.04
λ = 0.02

#Define Functions
u(c) = -exp(-c)
f(t) = exp(-ρ * t)*u(1 - exp(-λ * t))

#actual anti-derivative
F(t) = -50*(exp(t/50)-1)exp(exp(-t/50)-t/50-1)
#True value
true_val = F(100)-F(0)


#Plot Function
t = range(0, 100, length = 1001)
y = f.(t)
plot(t,y)

#set number of partitions
partition = [100,1000,5000,10000]

#pre-allocate vectors for solution
integral_midpoint = [0.0,0.0,0.0,0.0]
error_midpoint = [0.0,0.0,0.0,0.0]
time_midpoint = [0.0,0.0,0.0,0.0]

#calculate midpoint quadrature
for pars ∈ 1:4
    t = range(0, T, length = partition[pars])
    time_midpoint[pars] = @elapsed begin
        for segment ∈ eachindex(t) 
            a = (segment - 1) * T / partition[pars]
            b = segment * T / partition[pars]
            integral_midpoint[pars] = integral_midpoint[pars] + (b-a) * (f.((a+b)/2))
        end
    end
end

error_midpoint = integral_midpoint .- true_val

println(integral_midpoint)
println(error_midpoint)
println(time_midpoint)

#Trapezoid Integral

#pre-allocate solution vectors
integral_trapezoid = [0.0,0.0,0.0,0.0]
error_trapezoid = [0.0,0.0,0.0,0.0]
time_trapezoid = [0.0,0.0,0.0,0.0]

#calculate trapezoid quadrature
for pars ∈ 1:4
    t = range(0, T, length = partition[pars])
    time_trapezoid[pars] = @elapsed begin
        for segment ∈ eachindex(t) 
            a = (segment - 1) * T / partition[pars]
            b = segment * T / partition[pars]
            integral_trapezoid[pars] = integral_trapezoid[pars] + ((b-a)/2) * (f.(a)+f.(b))
        end
    end
end

error_trapezoid = integral_trapezoid .- true_val

println(integral_trapezoid)
println(error_trapezoid)
println(time_trapezoid)

#Simpsons Rule

#Pre-allocate solution vectors
integral_simpsons = [0.0,0.0,0.0,0.0]
error_simpsons = [0.0,0.0,0.0,0.0]
time_simpsons = [0.0,0.0,0.0,0.0]

for pars ∈ 1:4
    t = range(0, T, length = partition[pars])
    time_simpsons[pars] = @elapsed begin
        for segment ∈ eachindex(t)
            a = (segment-1) * T / partition[pars]
            b = segment * T / partition[pars]
            integral_simpsons[pars] = integral_simpsons[pars] + ((b-a)/6) * (f.(a)+4*f.((a+b)/2)+f.(b))
        end
    end
end

error_simpsons = integral_simpsons .- true_val

println(integral_simpsons)
println(error_simpsons)
println(time_simpsons)

#Monte Carlo

#set seed
Random.seed!(1453)

sol_mc = [0.0,0.0,0.0,0.0]
error_mc = [0.0,0.0,0.0,0.0]
time_mc = [0.0,0.0,0.0,0.0]

for pars ∈ 1:4
    time_mc[pars] = @elapsed begin
    draw = rand(Uniform(0,100),partition[pars])
    sol_mc[pars] = 100 .* mean(f.(draw)) #adjusting for the fact that our bounds are 0-100 rather than 0-1.
    end
end

error_mc = sol_mc .- true_val

println(sol_mc)
println(error_mc)
println(time_mc)



###Problem Three###
#define the Function
f(x, y) = 100(y-x^2)^2 + (1-x)^2

#define the derivative(s)
fx(x, y) = -400 .* x .* (y-x .^2) .- 2 .* (1-x)
fy(x, y) = 200 .* (y .- x .^2)

#Define the Jacobian
J(x, y) = [fx(x, y) fy(x, y)]

#define the second derivative(s)
fxx(x, y) = -400 .* (y .- 3 .* x.^2) .+ 2
fyy(x, y) = 200
fyx(x, y) = -400 .* x
fxy(x, y) = -400 .* x

#Define the Hessian
H(x, y) = [fxx(x, y) fxy(x, y); fyx(x, y) fyy(x, y)]

#plot the function
x = collect(-1:0.01:1)
y = collect(-1:0.01:1)

plot(x,y,f,st=:surface)

#Newton-Raphson

function newton_raphson(J::Function, H::Function, x, y)
    step = Inf
    tol = 1.0E-8
    while (norm(step) > tol)
        step = J(x, y) / H(x ,y)
        x = x - step[1]
        y = y - step[2]
    end
    z = [x, y]
    return z
end

#initial guess
a = 0.0
b = 0.0

time_nr = @elapsed begin
ans_nr = newton_raphson(J, H, a, b)
end

ans_nr

sol_nr = f(ans_nr[1], ans_nr[2])

error_solnr = norm(0 - sol_nr)

error_ansnr = norm([1,1] - ans_nr)

#BFGS

#recast functions to work with vectors

#define the Function
f(x) = 100(x[2]-x[1]^2)^2 + (1-x[1])^2

#define the derivative(s)
fx(x) = -400 .* x[1] .* (x[2]-x[1] .^2) .- 2 .* (1-x[1])
fy(x) = 200 .* (x[2] .- x[1] .^2)

#Define the Jacobian
J(x) = [fx(x) fy(x)]

#define the second derivative(s)
fxx(x) = -400 .* (x[2] .- 3 .* x[1].^2) .+ 2
fyy(x) = 200
fyx(x) = -400 .* x[1]
fxy(x) = -400 .* x[1]

#Define the Hessian
H(x) = [fxx(x) fxy(x); fyx(x) fyy(x)]

z = [0.0, 0.0]

time_BFGS = @elapsed begin
ans_BFGS = optimize(f, z, BFGS(); autodiff = :forward)
end

ans_BFGS

sol_BFGS = f(ans_BFGS.minimizer[1], ans_BFGS.minimizer[2])

error_solBFGS = norm(0 - sol_BFGS)

error_ansBFGS = norm([1,1] - [ans_BFGS.minimizer[1];ans_BFGS.minimizer[2]])

#Steepest Descent
#recast function to use in optimize function
function j!(J, x)
    J[1] = -400 .* x[1] .* (x[2]-x[1] .^2) .- 2 .* (1-x[1])
    J[2] = 200 .* (x[2] .- x[1] .^2)
end

#set initial value
z = [0.0, 0.0]

time_sd = @elapsed begin
solve_sd = optimize(f, j!, z, GradientDescent(), Optim.Options(iterations=100000))
end

solve_sd

ans_sd = [solve_sd.minimizer[1], solve_sd.minimizer[2]]

sol_sd = f(ans_sd[1], ans_sd[2])

error_solsd = norm(0 - sol_sd)

error_anssd = norm([1,1] - ans_sd)

#Conjugate Descent
time_cd = @elapsed begin
    solve_cd = optimize(f, j!, z, ConjugateGradient(), Optim.Options(iterations=10000))
end

solve_cd

ans_cd = [solve_cd.minimizer[1], solve_cd.minimizer[2]]

sol_cd = f(ans_cd[1], ans_cd[2])

error_solcd = norm(0 - sol_cd)

error_anscd = norm([1,1] - ans_cd)

###Problem Four###

#define parameters
m = 3
n = 3

#pre-allocate
α = zeros(m, n)
ω = zeros(m, n)
λ = zeros(1, n)
endow = zeros(m, n)
x = zeros(m, n)
util = zeros(1, n)

#set weights on goods
α[:, 1] = [1, 1, 1]
α[:, 2] = [1, 1, 1]
α[:, 3] = [1, 1, 1]

ω[:, 1] = [-2, -2, -2]
ω[:, 2] = [-2, -2, -2]
ω[:, 3] = [-2, -2, -2]

#set pareto weights
λ[1, :] = [1/3 1/3 1/3]

#set endowment
endow[:, 1] = [16, 4, 4]
endow[:, 2] = [4, 16, 4]
endow[:, 3] = [4, 4, 16]


#define objective function
function define_obj_fun(m, n, α, ω, λ, x)
    obj_fun = 0
    for j ∈ 1:n
        for i ∈ 1:m
            obj_fun = obj_fun + λ[1, j] .* α[i,j] .* ((x[i,j])^(1+ω[i,j])/(1+ω[i,j]))
        end
    end
    return obj_fun
end



#analytic solution
x_sol = [8 8 8; 8 8 8; 8 8 8]

define_obj_fun(m, n, α, ω, λ, x_sol)

#sum endowments

function define_tot_endow(m, n, endow)
    tot_endow = zeros(m)
    for i ∈ 1:m
        tot_endow[i] = 0
        for j ∈ 1:n
            tot_endow[i] = tot_endow[i] + endow[i, j]    
        end
    end
    return tot_endow
end

tot_endow = define_tot_endow(m, n, endow)



#define constraint


function define_constr(m, n, tot_endow, x)
    constr = zeros(m, 1)
    for i ∈ 1:m
        constr[i, 1] = tot_endow[i, 1]
        for j ∈ 1:n
            constr[i, 1] = constr[i, 1] - x[i, j]
        end
    end
    return constr
end

define_constr(m, n, tot_endow, x)

tot_endow

function solve_social_planner(m, n, α, ω, λ, tot_endow)
    model = Model(Ipopt.Optimizer)
    @variable(model, x[1:m, 1:n] >= 0 + 0.01)
    @constraint(model, sum(eachcol(x)) <= tot_endow)
    @objective(model, Max, sum(λ .* (α .* (x .^ (1 .+ ω)) ./ (1 .+ ω))))
    optimize!(model)
    return value.(x)
end

num_sol = solve_social_planner(m, n, α, ω, λ, tot_endow)

function compute_obj(α, ω, λ, x)
    sol = sum(λ .* (α .* (x .^ (1 .+ ω)) ./ (1 .+ ω)))
    return sol
end

sol_an = compute_obj(α, ω, λ, x_sol)
sol_num = compute_obj(α, ω, λ, num_sol)

#try new parameters
α = zeros(m, n)
ω = zeros(m, n)
λ = zeros(1, n)
endow = zeros(m, n)

#set weights on goods
α[:, 1] = [3, 1, 1]
α[:, 2] = [1, 3, 1]
α[:, 3] = [1, 1, 3]

ω[:, 1] = [-0.5, -2, -2]
ω[:, 2] = [-2, -0.5, -2]
ω[:, 3] = [-2, -2, -0.5]

#set pareto weights
λ[1, :] = [1/3 1/3 1/3]

endow[:, 1] = [16, 4, 4]
endow[:, 2] = [4, 16, 4]
endow[:, 3] = [4, 4, 16]

changed_sol = solve_social_planner(m, n, α, ω, λ, tot_endow)


#m=n=10
m = 10
n = 10

α = zeros(m, n)
ω = zeros(m, n)
λ = zeros(1, n)
endow = zeros(m, n)

α = 1 .+ α

ω = -2 .+ ω

#set pareto weights
λ = 1/3 .+ λ

endow = 7 .+ endow

endow[diagind(endow)] .= 17

tot_endow = define_tot_endow(m, n, endow)

big_sol = solve_social_planner(m, n, α, ω, λ, tot_endow)

###Problem Five###

#return to m=n=3
m = 3
n = 3
α = zeros(m, n)
ω = zeros(m, n)
λ = zeros(1, n)
endow = zeros(m, n)

#set weights on goods
α[:, 1] = [1, 1, 1]
α[:, 2] = [1, 1, 1]
α[:, 3] = [1, 1, 1]

ω[:, 1] = [-2, -2, -2]
ω[:, 2] = [-2, -2, -2]
ω[:, 3] = [-2, -2, -2]

#set vector of initial prices
p = [1.0, 2.0, 3.0]


endow[:, 1] = [16, 4, 4]
endow[:, 2] = [4, 16, 4]
endow[:, 3] = [4, 4, 16]

p_sol = [1.0, 1.0, 1.0]

function excess_demand(p, α, ω, endow, i)
    excess_func = 0
    for j in 1:n
     excess_func = excess_func + (p[i] * α[i,j])^(1 / ω[i, j]) * (sum(p .* endow[:, j]) / sum( ((1 .+ ω[:,j]) ./ (ω[:,j])) .* (1 ./ α[:, j]).^(1 ./ω[:, j]) .* p .^( (1 .+ ω[:, j]) ./ ω[:,j]) .- (p ./ α[:, j]).^(1 ./ ω[:,j]) .* (1 ./ ω[:, j]))) - endow[i, j]
    end
    return excess_func
end

excess_demand(p_sol, α, ω, endow, 1)


function excess_demand(m, p, α, ω, endow)
    excess_func = zeros(m)
    for j in 1:n
     excess_func = excess_func + ((p .* α[:,j]).^(1 ./ ω[:, j]) .* (sum(p .* endow[:, j]) ./ sum(((1 .+ ω[:,j]) ./ (ω[:,j])) .* (1 ./ α[:, j]).^(1 ./ω[:, j]) .* p .^( (1 .+ ω[:, j]) ./ ω[:,j]) .- (p ./ α[:, j]).^(1 ./ ω[:,j]) .* (1 ./ ω[:, j]))) .- endow[:, j])
    end
    return excess_func
end

excess_demand(m, p_sol, α, ω, endow)

function ed_i(p, i)
    excess_func = 0
    for j in 1:n
     excess_func = excess_func + (p[i] * α[i,j])^(1 / ω[i, j]) * (sum(p .* endow[:, j]) / sum( ((1 .+ ω[:,j]) ./ (ω[:,j])) .* (1 ./ α[:, j]).^(1 ./ω[:, j]) .* p .^( (1 .+ ω[:, j]) ./ ω[:,j]) .- (p ./ α[:, j]).^(1 ./ ω[:,j]) .* (1 ./ ω[:, j]))) - endow[i, j]
    end
    return excess_func
end

function f!(F, p)
    F[1] = ed_i(p, 1)
    F[2] = ed_i(p, 2)
    F[3] = ed_i(p, 3)
end

nlsolve(f!, p, autodiff = :forward)

#alternative, more flexible solution
function solve_excess_demand(p)
    excess_func = zeros(m)
    for j in 1:n
     excess_func = excess_func + ((p .* α[:,j]).^(1 ./ ω[:, j]) .* (sum(p .* endow[:, j]) ./ sum(((1 .+ ω[:,j]) ./ (ω[:,j])) .* (1 ./ α[:, j]).^(1 ./ω[:, j]) .* p .^( (1 .+ ω[:, j]) ./ ω[:,j]) .- (p ./ α[:, j]).^(1 ./ ω[:,j]) .* (1 ./ ω[:, j]))) .- endow[:, j])
    end
    return excess_func
end

ans = nlsolve(solve_excess_demand, p, autodiff = :forward)

p_num = ans.zero

#find allocations

x = zeros(m, n)

function demand_curve(m, p, α, ω, endow)
    demand_func = zeros(m,n)
    for j in 1:n
     demand_func[:, j] =  ((p .* α[:, j]).^(1 ./ ω[:, j]) .* (sum(p .* endow[:, j]) ./ sum(((1 .+ ω[:,j]) ./ (ω[:,j])) .* (1 ./ α[:, j]).^(1 ./ω[:, j]) .* p .^( (1 .+ ω[:, j]) ./ ω[:,j]) .- (p ./ α[:, j]).^(1 ./ ω[:,j]) .* (1 ./ ω[:, j]))))
    end
    return demand_func
end

x = demand_curve(m, p_num, α, ω, endow)

###Problem Six###
#Set-up
β = 0.97
α = 0.33
δ = 0.9
τ_grid = [0.2 0.25 0.3]

grid_τ = [0.2, 0.25, 0.3]

τ_trans = [0.9 0.1 0.0; 0.05 0.9 0.05; 0 0.1 0.9]

z_grid = [-0.0673 -0.0336 0 0.0336 0.0673]

grid_z = [-0.0673, -0.0336, 0, 0.0336, 0.0673]

z_trans =[0.9727 0.0273 0 0 0; 0.0041 0.9806 0.0153 0 0; 0 0.0082 0.9836 0.0082 0; 0 0 0.0153 0.9806 0.0041; 0 0 0 0.0273 0.9727]

#Problem 6.2
function solve_social_planner_ss(z, β, α, δ)
    model = Model(Ipopt.Optimizer)
    @variable(model, k >= 0 + 0.01)
    @variable(model, l >= 0 + 0.01)
    @variable(model, c >= 0 + 0.01)
    @variable(model, i >= 0 + 0.01)
    @variable(model, g >= 0 + 0.01)
    @constraint(model, c + i + g == exp(z)*k^(α)*l^(1-α))
    @constraint(model, k == δ*k + (1- 0.05(i/i - 1)^2)*i)
#   @constraint(model, l == 1) #normalized l_ss = 1
    @constraint(model, 1/β - 0.9 == α* exp(z)*k^(α-1)*l^(1-α))
    @constraint(model, (k^(α) * l^(-α) * (1-α))/l  == c)
    @constraint(model, 0.2/g == 1/c)
    @objective(model, Max, β/(1-β) * (log(c) + 0.2 * log(g) - (l^2)/2))
    optimize!(model)
    return value.(k), value.(c), value.(l), value.(g), value.(i)
end

spss = solve_social_planner_ss(0.0, β, α, δ)
println(spss)

function solve_steady_state(z, β, α, δ, τ)
    model = Model(Ipopt.Optimizer)
    @variable(model, k >= 0 + 0.01)
    @variable(model, l >= 0 + 0.01)
    @variable(model, c >= 0 + 0.01)
    @variable(model, i >= 0 + 0.01)
    @variable(model, w >= 0 + 0.01)
    @variable(model, r >= 0 + 0.01)
    @variable(model, g >= 0 + 0.01)
    @constraint(model, w == (1-α)*exp(z)*k^(α)*l^(-α))
    @constraint(model, r == (α)*exp(z)*k^(α-1)*l^(1-α))
    @constraint(model, k == δ*k + (1- 0.05(i/i - 1)^2)*i)
    @constraint(model, c + i == (1-τ)*w * l + r*k)
    @constraint(model, c == (1-τ)*w/l)
    @constraint(model, g == τ*w*l)
    @constraint(model, 1/β - 0.9 == α* exp(z)*k^(α-1)*l^(1-α))
    #@constraint(model, l == 1) #normalized l_ss = 1
    @objective(model, Max, β/(1-β) * (log(c) + 0.2 * log(g) - (l^2)/2))
    optimize!(model)
    return value.(k), value.(c), value.(l), value.(i), value.(w), value.(r), value.(g)
end

ss = solve_steady_state(0.0, β, α, δ, 0.25)

println(ss)

k_ss = ss[1]
i_ss = ss[4]

(log(spss[2]) + 0.2 * log(ss[4]) - (ss[3]^2)/2)

ss_u = (log(ss[2]) + 0.2 * log(ss[7]) - (ss[3]^2)/2)

#Problem 6.3

#Create Grid
#Grid lengths
KL = 25
IL = 5

k_grid = LinRange(0.7*k_ss, 1.3*k_ss, KL)
i_grid = LinRange(0.5*i_ss, 1.5*i_ss, IL)

grid_k = collect(k_grid)

grid_i = collect(i_grid)

wage(z, k, l, α) = (1-α).*exp(z).*k.^(α).*l.^(-α)

rents(z, k, l, α) = (α).*exp(z).*k.^(α-1).*l.^(1-α)

budget(z, k, l, α, τ,) = (1-τ).*wage(z,k,l,α).*l .+ rents(z,k,l,α).*k

prod(z, k, l, α) = exp(z).*k.^(α).*l.^(1-α)

invest(kp, k, i, i_l, δ) = kp .- (δ.*k .- (0.05 .*(i ./i_l .-1).^2).*i)

kp(k, i, i_l, δ) = δ .* k + (1-0.05(i ./ i_l .- 1).^2).*i

gov(τ, z, k, l, α) = τ.*wage(z, k, l, α).*l

cons(kp, k, i, i_l, δ, τ, z, l, α) = prod(z, k, l, α) .- invest(kp, k, i, i_l, δ) .- gov(τ, z, k, l, α) 

p_u(kp, k, i, i_l, δ, τ, z, l, α) = log(cons(kp, k, i, i_l, δ, τ, z, l, α)) .+ 0.2 .* log(gov(τ, z, k, l, α)) .- (l^2)/(2)

calc_u = p_u(ss[1], ss[1], ss[4], ss[4], δ, 0.25, 0, ss[3], α)

ss_u ≈ calc_u

calc_cons = cons(ss[1], ss[1], ss[4], ss[4], δ, 0.25, 0, ss[3], α)

calc_cons ≈ ss[2]

(0.75)*wage(0,ss[1],ss[3],α)*ss[3] .+ rents(0, ss[1], ss[3], α)*ss[1] .- invest(ss[1], ss[1], ss[4], ss[4], δ)

gov(0.25, 0, ss[1], ss[3], α)

ss[7]

ss[2]

wage(0, ss[1], ss[3], α)

kpol = zeros(KL, IL, 5, 3)
lpol = zeros(KL, IL, 5, 3)
ipol = zeros(KL, IL, 5, 3)
v = zeros(KL, IL, 5, 3)

#xs = 1:0.2:5
#ys = 1:0.2:5
#zs = 1:0.2:5

#samp_fun(x, y, z) = sin(y) + 0.1*x^2 + log(z)

#A = [samp_fun.(x, y, z) for x ∈ xs, y ∈ ys, z ∈ zs]

#interp_linear = LinearInterpolation((collect(xs), collect(ys), collect(zs)), A, extrapolation_bc=Line())

#interp_linear(3.1, 2.9, 3.1)

#expec_u = sum(z_trans[3, :] .* sum(τ_trans[2, :] .* p_u(ss[1], ss[1], ss[4], ss[4], δ, 0.25, 0, ss[3], α)))


#Guess r0 and w0, both are  k x i x z x τ arrays.
#Then, given r and w do value function iteration on the HH problem.
    #while sup_norm(V_1 p V_0) > tol: loop over the state space
        #V_1 = max_(c, l, k, i) {U(c,l) + β ∑_(τ',z') V_0(⋅) }
        #Then kpol, ipol, lpol are the argmax of the whole thing, which will be the policy function.
#Then given V_1, kpol, ipol, lpol check that:
    #r1 = α kpol.^(α) lpol.^(1-α), similar for w1
    #Check: max(sup_norm(r1 - r0), sup_norm(w1 - w0))  < tol if yes, then done,
    #if no, r0 <- r1, w0 <- w1
#while max(sup_norm(r1-r0), sup_norm(w1-w0)) > tol
    #while sup_norm(V1-V0) > tol
    #compute r1 & r0

    #for (a, k) ∈ enumerate(k_grid)
        #for (b, i) ∈ enumerate(i_grid)
            #for (c, z) ∈ enumerate(z_grid)
                #for (d, τ) ∈ enumerate(τ_grid)
#simplify so that you are only choosing investment and labor
#we may only need to guess r or w, not both    


#u(x; k, w, r, τ) = log.((1-τ) .*w .* x[2] .+ r .* k .- x[1]) .- ((x[2].^2) ./ 2)


ss_l = ((ss[4]-ss[6]*ss[1])+((ss[6]*ss[1]-ss[4])^2+4*((1-0.25)*ss[5])^2)^(1/2))/(2*(1-0.25)*ss[5])

ss_l ≈ ss[3]

l(i,k,τ,w,r) = ((i-r*k)+((r*k-i)^2+4*((1-τ)*w)^2)^(1/2))/(2*(1-τ)*w)

l(ss[4],ss[1],0.25,ss[5],ss[6])

function expect_v(x, c, d, i0, k, grid_z, grid_τ, grid_k, grid_i, v)
    v_func = LinearInterpolation((grid_k, grid_i, grid_z, grid_τ), v, extrapolation_bc=Line())
    expect = 0
    for e ∈ 1:5
        for f ∈ 1:3
            zp = grid_z[e]
            τp = grid_τ[f]
            kp = (δ * k + (1.0 - 0.05*((x[1] / i0 - 1)^2))*x[1])
            val = v_func(kp, x[1], zp, τp)
            expect += (z_trans[c, e] * τ_trans[d, f] * (val))
        end
    end
    return (expect)
end

function l(x, r, k, τ, w) 
    l = (((x-r*k)+((r*k-x)^2+4*((1-τ)*w)^2)^(1/2))/(2*(1-τ)*w))
    return(l)
end


function bellman(v_iter, grid_k, grid_i, grid_z, grid_τ, rpol, wpol)
    v = v_iter
    Tv = zeros(KL, IL, 5, 3)
    ipol = zeros(KL, IL, 5, 3)
    lpol = zeros(KL, IL, 5, 3)
    for a ∈ 1:KL
        for b ∈ 1:IL
            for c ∈ 1:5
                for d ∈ 1:3
                    r = rpol[a, b, c, d]
                    w = wpol[a, b, c, d]
                    k = grid_k[a]
                    i0 = grid_i[b]
                    z = grid_z[c]
                    τ = grid_τ[d]
                    lower = [0.0, 0.0]
                    upper = [grid_i[IL], Inf]
                    inner_optimizer = GradientDescent()
                    f(x) = -1 * ((log(((1-τ) * w * x[2])  + r * k - x[1])) - (x[2]^2)/2 + 
                    β * expect_v(x, c, d, i0, k, grid_z, grid_τ, grid_k, grid_i, v))
                    x_i = [0.1, 1.0]
                    results = optimize(f, lower, upper, x_i, Fminbox(inner_optimizer))
                    Tv[a,b,c,d] = results.minimum
                    ipol[a,b,c,d] = results.minimizer[1]
                    lpol[a,b,c,d] = results.minimizer[2]
                end
            end
        end
    end
    return (; v = Tv, ipol, lpol)
end

rpol = zeros(KL, IL, 5, 3)
wpol = zeros(KL, IL, 5, 3)

for a ∈ 1:KL
    for b ∈ 1:IL
        for c ∈ 1:5
            for d ∈ 1:3
                rpol[a,b,c,d] = (α)*exp(grid_z[c])*grid_k[a]^(α-1)*ss[3]^(1-α)
            end
        end
    end
end

rpol

for a ∈ 1:KL
    for b ∈ 1:IL
        for c ∈ 1:5
            for d ∈ 1:3
                wpol[a,b,c,d] = (1-α)*exp(grid_z[c])*grid_k[a]^(α)*ss[3]^(-α)
            end
        end
    end
end

wpol

wpol

initial_v = zeros(KL, IL, 5, 3)

time_bellman = @elapsed begin
    results_bellman = bellman(initial_v, grid_k, grid_i, grid_z, grid_τ, rpol, wpol)
end

iv = results_bellman.v

results_bellman.ipol

results_bellman.lpol

abs.(rpol - wpol)

maximum(abs.(iv - initial_v))

function solv_opt(iv, grid_k, grid_i, grid_z, grid_τ, rpol_old, wpol_old, tolerance = 1E-10, maxiter = 1000)
    v_old = iv
    normdiff = Inf
    iter = 1
    rpol = rpol_old
    wpol = wpol_old
    while normdiff > tolerance && iter <= maxiter
        v_new = bellman(v_old, grid_k, grid_i, grid_z, grid_τ, rpol, wpol).v
        normdiff = maximum(abs.(v_old - v_new))
        v_old = v_new
        iter = iter + 1
    end
    out = bellman(v_old, grid_k, grid_i, grid_z, grid_τ, rpol, wpol)
    ipol = out.ipol
    return (; value = v_old, ipol, normdiff, iter)
end

initial_v[24,3,2,2]

time_opt = @elapsed begin
   sol_opt = solv_opt(iv, grid_k, grid_i, grid_z, grid_τ, rpol, wpol)
end

iv = sol_opt.value

ipol = sol_opt.ipol

check = bellman(sol_opt.value, grid_k, grid_i, grid_z, grid_τ, rpol, wpol).v

itercheck = sol_opt.iter

check ≈ sol_opt.value

initial_r = zeros(KL, IL, 5, 3)

for a ∈ 1:KL
    for b ∈ 1:IL
        for c ∈ 1:5
            for d ∈ 1:3
                initial_r[a,b,c,d] = (α)*exp(grid_z[c])*grid_k[a]^(α-1)*max(0.000001,l(ipol[a,b,c,d],grid_k[a],grid_τ[d],wpol[a,b,c,d],rpol[a,b,c,d]))^(1-α)
            end
        end
    end
end

initial_w = zeros(KL, IL, 5, 3)

for a ∈ 1:KL
    for b ∈ 1:IL
        for c ∈ 1:5
            for d ∈ 1:3
                initial_w[a,b,c,d] = (1-α)*exp(grid_z[c])*grid_k[a]^(α)*max(0.000001,l(ipol[a,b,c,d],grid_k[a],grid_τ[d],wpol[a,b,c,d],rpol[a,b,c,d]))^(-α)
            end
        end
    end
end


function solve_w_r(initial_w, initial_r, initial_v; grid_k, grid_i, grid_z, grid_τ, outer_tol = 1E-10, outer_maxiter = 1000)
    rpol_old = initial_r
    wpol_old = initial_w
    v_old = initial_v
    outer_iter = 1
    outer_diff = Inf
    while outer_diff > outer_tol && outer_iter <= outer_maxiter
        sol_opt = solv_opt(v_old, grid_k, grid_i, grid_z, grid_τ, rpol_old, wpol_old)
        ipol = sol_opt.ipol
        wpol_new = zeros(KL, IL, 5, 3)
        rpol_new = zeros(KL, IL, 5, 3)
        for a ∈ 1:KL
            for b ∈ 1:IL
                for c ∈ 1:5
                    for d ∈ 1:3
                        wpol_new[a,b,c,d] = (1-α)*exp(grid_z[c])*grid_k[a]^(α)*max(0.000001,l(ipol[a,b,c,d],grid_k[a],grid_τ[d],wpol[a,b,c,d],rpol[a,b,c,d]))^(-α)
                    end
                end
            end
        end
        for a ∈ 1:KL
            for b ∈ 1:IL
                for c ∈ 1:5
                    for d ∈ 1:3
                        rpol_new[a,b,c,d] = (α)*exp(grid_z[c])*grid_k[a]^(α-1)*max(0.000001,l(ipol[a,b,c,d],grid_k[a],grid_τ[d],wpol[a,b,c,d],rpol[a,b,c,d]))^(1-α)
                    end
                end
            end
        end
        outer_diff = max(maximum(abs.(wpol_old - wpol_new)), maximum(abs.(rpol_old - wpol_new)))
        wpol_adj = 0.8 .* wpol_old
        wpol_old = wpol_adj .+ 0.2 .* wpol_new
        rpol_adj = 0.8 .* rpol_old
        rpol_old = rpol_adj .+ 0.2 .* rpol_new
        v_new = sol_opt.value
        v_old = v_new
        outer_iter = outer_iter + 1
    end
    out = bellman(v_old, grid_k, grid_i, grid_z, grid_τ, rpol_old, wpol_old)
    ipol = out.ipol
    for a ∈ 1:KL
        for b ∈ 1:IL
            for c ∈ 1:5
                for d ∈ 1:3
                    wpol_final[a,b,c,d] = (1-α)*exp(grid_z[c])*grid_k[a]^(α)*max(0.000001,l(ipol[a,b,c,d],grid_k[a],grid_τ[d],wpol[a,b,c,d],rpol[a,b,c,d]))^(-α)
                end
            end
        end
    end
    for a ∈ 1:KL
        for b ∈ 1:IL
            for c ∈ 1:5
                for d ∈ 1:3
                    rpol_final[a,b,c,d] = (α)*exp(grid_z[c])*grid_k[a]^(α-1)*max(0.000001,l(ipol[a,b,c,d],grid_k[a],grid_τ[d],wpol[a,b,c,d],rpol[a,b,c,d]))^(1-α)
                end
            end
        end
    end 
    return (; value = v_old, ipol, wpol_final, rpol_final, outer_diff, outer_iter)
end

time_solwr = @elapsed begin
    sol_wr = solve_w_r(initial_w, initial_r, iv; grid_k, grid_i, grid_z, grid_τ, outer_tol = 1E-7, outer_maxiter = 1000)
end