module KernelPkg

using ITensors

export compute_tile

# Gate definitions, since TKET's are slightly different
function ITensors.op(::OpName"TKET_Rx", t::SiteType"Qubit"; α::Number)
  θ = π*α/2
  return [
    cos(θ) -im*sin(θ)
    -im*sin(θ) cos(θ)
  ]
end

function ITensors.op(::OpName"TKET_Rz", t::SiteType"Qubit"; α::Number)
  θ = π*α/2
  return [
    exp(-im * θ) 0
    0 exp(im * θ)
  ]
end

function ITensors.op(::OpName"TKET_XXPhase", t::SiteType"Qubit"; α::Number)
  θ = π*α/2
  return [
    cos(θ) 0 0 -im*sin(θ)
    0 cos(θ) -im*sin(θ) 0
    0 -im*sin(θ) cos(θ) 0
    -im*sin(θ) 0 0 cos(θ)
  ]
end

function ITensors.op(::OpName"TKET_ZZPhase", t::SiteType"Qubit"; α::Number)
  θ = π*α/2
  return [
    exp(-im * θ) 0 0 0
    0 exp(im * θ) 0 0
    0 0 exp(im * θ) 0
    0 0 0 exp(-im * θ)
  ]
end

# Build and simulate the given circuit (as a list of gates)
function build_and_sim_circ(circuit, site_inds, cutoff::Float64)
  gates::Vector{ITensor} = []

  for (name, qubits, params) in circuit
    if name == "H"
      append!(gates, [op("H", site_inds, 1+qubits[1])])
    elseif name == "Rx"
      append!(gates, [op("TKET_Rx", site_inds, 1+qubits[1]; α=params[1])])
    elseif name == "Rz"
      append!(gates, [op("TKET_Rz", site_inds, 1+qubits[1]; α=params[1])])
    elseif name == "XXPhase"
      append!(gates, [op("TKET_XXPhase", site_inds, 1+qubits[1], 1+qubits[2]; α=params[1])])
    elseif name == "ZZPhase"
      append!(gates, [op("TKET_ZZPhase", site_inds, 1+qubits[1], 1+qubits[2]; α=params[1])])
    elseif name == "SWAP"
      append!(gates, [op("SWAP", site_inds, 1+qubits[1], 1+qubits[2])])
    else
      error("KernelPkg error: Unrecognised gate.")
    end
  end

  # Simulate the circuit
  mps_time = @elapsed begin
    ψ = apply(gates, MPS(site_inds, "0"); cutoff=cutoff)
  end
  max_chi = maxlinkdim(ψ)
  return ψ, max_chi, mps_time
end

# Compute all of the entries of a tile; includes MPS simulation of the circuits
function compute_tile(n_qubits::Int64, x_circs, y_circs, cutoff::Float64)
  tile = zeros(size(y_circs, 1), size(x_circs, 1))
  site_inds = siteinds("Qubit", n_qubits)

  # Generate all MPS
  
  x_mps = []
  x_chi = []
  x_time = []
  for circ in eachrow(x_circs)
    mps, chi, time = build_and_sim_circ(circ, site_inds, cutoff)
    push!(x_time, time)
    push!(x_chi,chi)
    push!(x_mps,mps)
  end

  y_mps = []
  y_chi = []
  y_time = []
  for circ in eachrow(y_circs)
    mps, chi, time = build_and_sim_circ(circ, site_inds, cutoff)
    push!(y_time, time)
    push!(y_chi,chi)
    push!(y_mps,mps)
  end
  
  # For each pair, compute the inner product
  vdot_time = []
  pairs = [(i, j) for i in 1:length(y_mps) for j in 1:length(x_mps)]
  for (i, j) in pairs
    time = @elapsed begin
      tile[i, j] = abs(inner(y_mps[i], x_mps[j]))^2 
    end
    push!(vdot_time, time)
  end

  return tile, x_chi, y_chi, x_time, y_time, vdot_time
end

end  #module
