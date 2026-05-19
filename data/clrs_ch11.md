# CLRS Chapter 11 — Hash Tables (study notes)

> Self-contained summary written for the RAG mini-lab. Replace this file with
> your own CLRS Chapter 11 excerpt if you want to ingest the full textbook
> material; the ingest pipeline reads anything under `data/` matching `*.md`.

## 11.1 Direct-address tables

Direct addressing is the simplest possible dictionary scheme. We assume that
every key comes from a small universe `U = {0, 1, ..., m-1}` and we maintain an
array `T[0..m-1]` where slot `T[k]` holds the element whose key is `k`, or NIL
if no such element exists. The three dictionary operations — SEARCH, INSERT,
and DELETE — each run in `O(1)` worst-case time, because the key is itself the
index. The catch is space: when the universe `U` is large but the number of
actually stored keys `n` is small, most slots sit empty and the table wastes
memory proportional to `|U|`. Direct addressing is therefore appropriate only
when the universe is small enough that allocating a slot per possible key is
affordable.

## 11.2 Hash tables

A hash table generalises direct addressing for the common case `n << |U|`. It
stores `n` keys in a table of size `m`, where typically `m = O(n)` rather than
`m = |U|`. A hash function `h : U -> {0, 1, ..., m-1}` maps each key `k` to a
slot `h(k)`; we say `k` *hashes to* slot `h(k)`. Because `|U| > m`, the
pigeonhole principle guarantees that two distinct keys can collide — that is,
hash to the same slot. Collision resolution is therefore the central design
question for any hash table.

### 11.2.1 Collision resolution by chaining

In chaining, every slot `T[j]` points to a doubly linked list (the *chain*)
containing all elements whose key hashes to `j`. INSERT prepends an element to
the chain in `O(1)`. SEARCH walks the chain looking for the target key, and
DELETE unlinks a known element in `O(1)` once it is found. Under the simple
uniform hashing assumption — every key is equally likely to hash into any of
the `m` slots independently of where other keys hashed — the expected length
of any chain is the load factor `alpha = n / m`. The expected cost of an
unsuccessful SEARCH is `Theta(1 + alpha)`, and the expected cost of a
successful SEARCH is also `Theta(1 + alpha)`. When we keep `alpha` bounded by
a constant by resizing, all dictionary operations run in expected `O(1)` time.

## 11.3 Hash functions

A good hash function approximates simple uniform hashing: each key should be
equally likely to land in any slot, and regularities in the input data should
not produce collisions. Two classic constructions are the division method and
the multiplication method.

### Division method

Set `h(k) = k mod m`. The method is fast — a single division — but the choice
of `m` matters. Powers of two are bad because then `h(k)` only depends on the
low-order bits of `k`. A common rule of thumb is to pick `m` as a prime that
is not too close to an exact power of two, so that all bits of `k` influence
the result.

### Multiplication method

Pick a constant `A` with `0 < A < 1`, compute `k * A`, take the fractional
part, and multiply by `m`. That is, `h(k) = floor( m * frac(k * A) )`. Knuth
suggests `A = (sqrt(5) - 1) / 2 ≈ 0.6180339887` because of its favourable
distribution properties. Unlike the division method, the multiplication method
works well for any value of `m`, which is convenient when `m` is chosen as a
power of two for efficient implementation.

### Universal hashing

If an adversary knows the fixed hash function in advance, it can choose keys
that all map to the same slot, forcing worst-case `Theta(n)` operations. The
defence is to pick the hash function at random from a carefully designed
family `H` of functions. A family `H` is *universal* if for any two distinct
keys `x` and `y`, the probability that a randomly chosen `h ∈ H` collides on
them is at most `1/m`. Universal hashing guarantees expected `O(1 + alpha)`
performance no matter what input the adversary supplies. A standard
construction picks a prime `p` larger than every key, draws `a ∈ {1, ..., p-1}`
and `b ∈ {0, ..., p-1}` uniformly at random, and defines
`h_{a,b}(k) = ((a*k + b) mod p) mod m`. The set `{h_{a,b}}` is a universal
family.

## 11.4 Open addressing

Open addressing avoids storing chains by placing every element directly in
the table; therefore `n <= m` and the load factor satisfies `alpha <= 1`. To
INSERT a key `k`, we *probe* a sequence of slots `h(k, 0), h(k, 1), ...` until
we find an empty one. SEARCH probes the same sequence until it either finds
`k` or hits an empty slot. DELETE is awkward: marking a slot empty would break
the probe sequence for other keys, so deleted slots are usually marked with a
special DELETED sentinel that is treated as occupied for SEARCH but free for
INSERT.

Three common probe sequences are linear probing, quadratic probing, and
double hashing.

### Linear probing

`h(k, i) = (h'(k) + i) mod m`. Linear probing is simple and cache-friendly
because consecutive probes touch consecutive slots. Its weakness is *primary
clustering*: long runs of occupied slots tend to grow longer, because any key
hashing into the run lengthens it.

### Quadratic probing

`h(k, i) = (h'(k) + c1 * i + c2 * i^2) mod m`. Quadratic probing avoids primary
clustering but suffers from *secondary clustering*: any two keys with the same
initial hash follow the same probe sequence. The constants `c1`, `c2`, and the
table size `m` must be chosen carefully to ensure that the probe sequence
visits every slot.

### Double hashing

`h(k, i) = (h1(k) + i * h2(k)) mod m`. Double hashing uses two auxiliary hash
functions: `h1` gives the initial slot, `h2` gives the step size. To make sure
the probe sequence covers the whole table, `h2(k)` must be relatively prime to
`m`; one common choice is `m` prime and `h2(k) = 1 + (k mod (m-1))`. Double
hashing achieves Theta(m^2) distinct probe sequences, far better than the
`m` sequences of linear or quadratic probing, and its performance is close to
that of an idealised uniform hashing scheme.

### Analysis of open addressing

Under uniform hashing, the expected number of probes in an unsuccessful
search is at most `1 / (1 - alpha)`, and the expected number of probes in a
successful search is at most `(1 / alpha) * ln(1 / (1 - alpha))`. Both
quantities blow up as `alpha` approaches 1, which is why practical open
addressing tables are kept well below full.

## 11.5 Perfect hashing

When the set of keys is *static* — known in advance and never changing — we
can construct a hashing scheme with `O(1)` *worst-case* SEARCH time. The
classic two-level construction (Fredman, Komlós, and Szemerédi, 1984) picks a
universal hash function `h` to map the `n` keys into `m = n` primary slots,
then resolves the collisions in slot `j` (which holds `n_j` keys) with a
secondary hash table of size `m_j = n_j^2` and another universal hash function
`h_j`. The quadratic blow-up in secondary table size guarantees, by a
counting argument, that the secondary tables can be made collision-free with
constant probability, so a randomised construction succeeds in expected
`O(n)` total time and total space `O(n)`. Once built, every SEARCH performs
exactly two hash evaluations and two memory accesses, regardless of input.

## Recap

* Direct addressing: O(1) operations, but space is O(|U|).
* Chaining: simple, robust, expected O(1 + alpha) per operation.
* Open addressing: better cache behaviour and no pointers, but harder DELETE
  and rapid degradation as alpha approaches 1.
* Universal hashing: randomise the hash function so no fixed input is bad.
* Perfect hashing: for static sets, achieve worst-case O(1) SEARCH using a
  two-level scheme of size O(n).
