# using ArrayViews
import Base:ctranspose,*
using ArrayViews

# Types and function definitions for linear shift convolutional neural nets
# the type of CNN here is a set of 2D filters that are shifted in the
# columnar direction but not the row direction.

# TODO: unify interface for scatter()


# The most sane way is to have:
# type ShiftFilterBank{T}
# which encapsulates the idea of 'filter as a matrix'.
# various operations on SFB and vectors would then
# have a common 'wrap-unwrap' implementation function

# TODO: make filter1d*vector operation faster if fft(vector) can
# be precomputed.


# this represents a linear 1D filter
type Filter1D{T}
	fk::Vector{Complex{T}}
end
#Filter1D(x::Vector{Real}) = Filter1D(conj(fft(x)))

*(f::Filter1D, x::AbstractVector) = real(ifft(conj(f.fk) .* fft(x,1)))
.*{T}(f::Filter1D{T}, x::T) = Filter1D{T}(f.fk .* x)
.+{T}(f::Filter1D{T}, x::T) = Filter1D{T}(f.fk .+ x)
 *{T}(f::Filter1D{T}, x::T) = Filter1D{T}(f.fk .* x)
-{T}(f::Filter1D{T}, g::Filter1D{T}) = Filter1D{T}(f.fk - g.fk)
+{T}(f::Filter1D{T}, g::Filter1D{T}) = Filter1D{T}(f.fk + g.fk)
ctranspose(f::Filter1D) = Filter1D(conj(f.fk))

# simulate δ*x' where x is the input and δ are the errors.
scatter{T}(::Type{Filter1D{T}}, δ, x) = Filter1D(conj(fft(δ)).*fft(x))

# this represents a set of filters that can be
# circularly shifted along the columnar direction
# filters are represented as a matrix, with
# <filter i, column j> in row i, column j of the block matrix
type ShiftFilterBank{T}
	filts::Matrix{Filter1D{T}}
end
ctranspose(w::ShiftFilterBank) = ShiftFilterBank(w.filts')
.*{T}(w::ShiftFilterBank{T}, x::T) = ShiftFilterBank(convert(Array{Filter1D{T}}, w.filts.*x))
 *{T}(w::ShiftFilterBank{T}, x::T) = ShiftFilterBank(convert(Array{Filter1D{T}}, w.filts.*x))
 *{T}(x::T, w::ShiftFilterBank{T}) = ShiftFilterBank(convert(Array{Filter1D{T}}, w.filts.*x))
.+{T}(w::ShiftFilterBank{T}, x::T) = ShiftFilterBank(convert(Array{Filter1D{T}}, w.filts.+x))
 +{T}(w::ShiftFilterBank{T}, v::ShiftFilterBank{T}) = ShiftFilterBank(w.filts  + v.filts)
.-{T}(w::ShiftFilterBank{T}, v::ShiftFilterBank{T}) = ShiftFilterBank(w.filts .- v.filts)

# given a vector, chop it up into a set of views
# with each view of length N
choparray(x, N) = [view(x, (1:N)+i) for i = 0:N:length(x)-1]

# given an array of vectors, concatenate them into a single vector
# TODO: could this be faster?
catarray(x) = cat(1,x...)

# this is nothing but a matrix-vector multiplication;
# applying each 'block' of <filts> to the corresponding
# 'slice' of x
function *{T}(filts::Matrix{Filter1D{T}}, x::Vector{AbstractVector{T}})
	[
		reduce(.+, [filts[i,j]*x[j] for j=1:size(filts,2)])
		for i = 1:size(filts,1)
	]
end

# apply a filter bank to a vectorized image
# this first 'lowers' w and x to a common representation
# as abstract matrices and vectors, then raises the result
function *{T}(w::ShiftFilterBank{T}, x::Vector{T})
	N = length(w.filts[1].fk)
	xv = choparray(x, N)
	xv::Vector{AbstractVector{T}}
	w.filts::Matrix{Filter1D{T}}
	outv = (w.filts * xv)
	catarray(outv)
end

function *{T}(w::ShiftFilterBank{T}, x::Matrix{T})
	size(x,2) == 1 || throw(ArgumentError("batch training not supported for lcnn"))
	w*vec(x)
end

function applylayer{T}(w::ShiftFilterBank{T}, b::AbstractVector{T}, x::Matrix)
	filts = w.filts
	(N, r) = divrem(size(x,1), size(filts,2))
	r == 0 || throw(ArgumentError("Dimension mismatch"))

	outl = Array(T, N*length(b), size(x,2))
	for i = 1 : size(x,2)
		@show i
		outl[:,i] = w*x[:,i] .+ vec(repmat(b,N,1))
	end
	outl
end

# simulate δ*x' where x is the input and δ are the errors.
function scatter{T}(w::ShiftFilterBank{T}, δ::Array{T}, x::Array{T})
	size(δ, 2) == 1 || throw(DomainError("batch δ not supported"))
	size(x, 2) == 1 || throw(DomainError("batch x not supported"))

	N = length(w.filts[1].fk)
	
	chopδ = choparray(δ, N)
	chopx = choparray(x, N)

	outw = [scatter(Filter1D{T}, thisδ, thisx) for thisδ in chopδ, thisx in chopx]
	ShiftFilterBank(outw), map(sum, chopδ)
end

# randfilt1d() = Filter1D(fft(randn(4)))
# w = [randfilt1d() for i=1:2,j=1:2]
# w * randn(8)


# (u,e,v)=svd(z)
# e[1]*u[:,1]*(v[:,1]')
