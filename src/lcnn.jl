# using ArrayViews
import Base.*

# Types and function definitions for linear shift convolutional neural nets
# the type of CNN here is a set of 2D filters that are shifted in the
# columnar direction but not the row direction.

# TODO: unify interface for scatter()


# TODO: most sane way is to have:
# type ShiftFilterBank{T} <: AbstractMatrix{T}
# which encapsulates the idea of 'filter as a matrix'.
# various operations on SFB and vectors would then
# have a common 'wrap-unwrap' implementation function

# this represents a linear 1D filter
type Filter1D{T}
	fk::Vector{Complex{T}}
end

*(f::Filter1D, x::Vector) = real(ifft(conj(f.fk) .* fft(x,1)))
ctranspose(f::Filter1D) = Filter1D(conj(f.fk))

# simulate δ*x' where x is the input and δ are the errors.
scatter{T}(f::Type{Filter1D{T}}, δ, x) = Filter1D(conj(fft(δ)).*fft(x))

# this represents a set of filters that can be
# circularly shifted along the columnar direction
# filters are represented as a matrix, with
# <filter i, column j> in row i, column j of the block matrix
type ShiftFilterBank{T}
	filts::Matrix{Filter1D{T}}
end

# function *{T}(filts::Matrix{Filter1D{T}}, x::Vector{Vector{T}})
# 	[
# 		reduce(.+, [filts[i,j]*x[j] for j=1:size(filts,1)])
# 		for i = 1:size(filts,2)
# 	]
# end

# apply a filter bank to a vectorized image
# this is nothing but a matrix-vector multiplication;
# applying each 'block' of <filts> to the corresponding
# 'slice' of x
function *{T}(w::ShiftFilterBank{T}, x::Vector{T})
	filts = w.filts
	N = length(filts[1].fk)
	@show N, length(x), size(filts,2)
	outv = Array(T,0)
	for i = 1 : size(filts,1)
		cs = [filts[i,j]*x[N*(j-1)+1:N*j] for j=1:size(filts,2)]
		c = reduce(.+, cs)
		append!(outv, c)
	end
	outv
end

function *{T}(w::ShiftFilterBank{T}, x::Matrix{T})
	filts = w.filts
	size(x,2) == 1 || throw(ArgumentError("batch training not supported for lcnn"))
	filts*vec(x)
end

function applylayer{T}(w::ShiftFilterBank{T}, b::AbstractVector{T}, x::Matrix)
	filts = w.filts
	(N, r) = divrem(size(x,1), size(filts,2))
	r == 0 || throw(ArgumentError("Dimension mismatch"))
	#@show size(x)
	#@show size(vec(repmat(b,int(length(x)/length(b)),1)))
	outl = Array(T, N*length(b), size(x,2))
	for i = 1 : size(x,2)
		@show i
		outl[:,i] = w*x[:,i] .+ vec(repmat(b,N,1))
	end
	outl
end

# simulate δ*x' where x is the input and δ are the errors.
function scatter{T}(w::ShiftFilterBank{T}, δ::Vector{T}, x::Vector{T})
	filts = w.filts

	# ncol is the number of image columns
	(nrow, ncol) = size(w)

	# N is the length of each column
	(N, r) = divrem(length(x),ncol)
	r == 0 || throw(ArgumentError("Dimension mismatch"))

	thisδ = [δ[N*(j-1)+1:N*j] for j=1:nrow]
	outw = Array(ShiftFilterBank{T},nrow,ncol)
	for i = 1 : ncol
		thisx = x[N*(i-1)+1:N*i]
		for j = 1 : nrow
			outw[i,j] = scatter(Filter1D{T}, thisδ[j], thisx)
		end
	end
	outw, map(sum, thisδ)
end

# randfilt1d() = Filter1D(fft(randn(4)))
# w = [randfilt1d() for i=1:2,j=1:2]
# w * randn(8)


# (u,e,v)=svd(z)
# e[1]*u[:,1]*(v[:,1]')
