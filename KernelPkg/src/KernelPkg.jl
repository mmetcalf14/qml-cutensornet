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
function build_and_sim_circ(circuit, site_inds, value_of_zero::Float64)
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
  ψ = apply(gates, MPS(site_inds, "0"); cutoff=value_of_zero)
  max_chi = maxlinkdim(ψ)
  # println(ψ)
  return ψ , max_chi
end

# Compute all of the entries of a tile; includes MPS simulation of the circuits
function compute_tile(n_qubits::Int64, x_circs, y_circs, value_of_zero::Float64)
  tile = zeros(size(y_circs, 1), size(x_circs, 1))
  site_inds = siteinds("Qubit", n_qubits)

  # Generate all MPS
  
  x_mps = []
  x_chi = []
  for circ in eachrow(x_circs)
    mps, chi = build_and_sim_circ(circ, site_inds, value_of_zero)
    append!(x_chi,chi)
    push!(x_mps,mps)
  end

  y_mps = []
  y_chi = []
  for circ in eachrow(y_circs)
    mps, chi = build_and_sim_circ(circ, site_inds, value_of_zero)
    append!(y_chi,chi)
    push!(y_mps,mps)
  end
  #x_mps = [build_and_sim_circ(circ, site_inds, value_of_zero) for circ in #eachrow(x_circs)]
  #y_mps = [build_and_sim_circ(circ, site_inds, value_of_zero) for circ in #eachrow(y_circs)]
  # For each pair, compute the inner product
  pairs = [(i, j) for i in 1:length(y_mps) for j in 1:length(x_mps)]
  for (i, j) in pairs
    tile[i, j] = abs(inner(y_mps[i], x_mps[j]))^2
  end

  return tile, x_chi, y_chi
end

end  #module
