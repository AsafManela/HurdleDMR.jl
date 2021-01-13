# code to compile the Stock Market Execss and the State of the Union Address texts
# into a matching DataFrame and Document-Term matrix

using Lasso, CSV, DataDeps, DataFrames, FamaFrenchData, TextAnalysis, SparseArrays

register(DataDep("sotu",
        """
        Dataset: State of the Union Adresses
        Website: http://stateoftheunion.onetwothree.net
        Author: Brad Borevitz
        """,
        "http://stateoftheunion.onetwothree.net/texts/stateoftheunion1790-2020.txt.zip",
        "f61dfaf9b487872173e664488e6daad1783fa42691ed8009b2c5245866939b97",
        post_fetch_method = unpack
    ))

# fast way to remove sparse terms using the dtm, returns a new dtm with only terms that occur at least minoccurances times in the corpus
function remove_sparse_terms(D::TextAnalysis.DocumentTermMatrix, minoccurances = 3)
  m,n=size(D.dtm)
  rows = rowvals(D.dtm)
  vals = nonzeros(D.dtm)
  newrows = Int[]
  newcols = Int[]
  newvals = Int[]
  newterms = String[]
  new_column_indices = Dict{String, Int}()
  colsums = sum(D.dtm, dims=1)
  col = 1
  for i = 1:n
    if colsums[i]>=minoccurances
      for j in nzrange(D.dtm, i)
        row = rows[j]
        val = vals[j]
        push!(newrows, row)
        push!(newcols, col)
        push!(newvals, val)
      end
      term = D.terms[i]
      push!(newterms, term)
      new_column_indices[term] = col
      col += 1
    end
  end
  newterms
  if length(newrows) > 0
      newdtm = sparse(newrows, newcols, newvals)
  else
      newdtm = spzeros(Int,m, 0)
  end
  DocumentTermMatrix(newdtm, newterms, new_column_indices)
end

# get the SOTU text
function loadsotu()
    sotutxt = datadep"sotu/stateoftheunion1790-2020.txt"
    sotustr = read(sotutxt, String)
    addresses = strip.(split(sotustr, "***", keepempty=false)[2:end-1])

    # typical structure:
    #
    # ***
    #
    # State of the Union Address
    # Harry S. Truman
    # January 6, 1947
    #
    # Mr. President, Mr. Speaker, Members of the Congress of the United States:
    #
    # It looks like a good many of you have moved over to the left since I was
    # here last! ...

    # vectors to save parsed documents and dates
    docs = NGramDocument[]
    years = Int[]

    for a in addresses[1:end]
        # an IOBuffer provides an efficient way of reading lines
        strbuf = IOBuffer(a)

        # read header
        title = readline(strbuf)
        speaker = readline(strbuf)
        datestr = readline(strbuf)
        year = parse(Int, datestr[end-3:end])
        if 1927 <= year <= 2019
        
            # skip "Dear Congress" at start
            readline(strbuf)
            readline(strbuf)
            readline(strbuf)

            speechtext = read(strbuf, String)
            # remove everything after "NOTE:"
            ix = findfirst("NOTE:", speechtext)
            if !isnothing(ix)
                speechtext = speechtext[1:ix[1]-1]
            end

            # initial preprocessing
            sdoc = StringDocument(speechtext)
            prepare!(sdoc, strip_non_letters | strip_case)
            prepare!(sdoc, strip_stopwords)
            prepare!(sdoc, strip_whitespace | strip_corrupt_utf8)

            # build TextAnalysis NGramDocument of bigrams (complexity=2)
            doc = NGramDocument(text(sdoc), 2)
            author!(doc, speaker)
            timestamp!(doc, datestr)

            # save parsed documents into vectors
            if length(years) > 0 && year == years[end]
                # overwrite last one added becasue sometimes (e.g. 1953) the outgoing president addresses too
                docs[end] = doc
            else
                push!(docs, doc)    
                push!(years, year)
            end
        end
    end

    corpus = Corpus(docs)
    @info "built SOTU corpus with $(length(corpus)) speeches"

    update_lexicon!(corpus)
    D = DocumentTermMatrix(corpus)
    D = remove_sparse_terms(D, 20)
    @info "built Document Term Matrix with bigrams occuring at least 20 times in the corpus"
    
    sotu = DataFrame(
            Date=[parse(Int, ts[end-3:end]) for ts in timestamps(corpus)],
            President=author.(corpus)
        )

    sotu, D
end

# load fama french annual market excess Returns
function loadreturns()
    tables, tablenotes, filenotes = readFamaFrench("F-F_Research_Data_Factors")
    FF3_annual = tables[2]
    rename!(FF3_annual, "Mkt-RF"=>"Rem")
    FF3_annual
end

# load merged sotu + market excess returns dataset
function loadsoturet(;cache=true)
    csvfile = joinpath(datadep"sotu", "sotu.ret.csv")
    dtmfile = joinpath(datadep"sotu", "sotu.dtm.csv")
    termsfile = joinpath(datadep"sotu", "sotu.terms.csv")

    if cache && isfile(csvfile) && isfile(dtmfile) && isfile(termsfile)
        df = CSV.read(csvfile, DataFrame)
        
        countsIJV = CSV.read(dtmfile, DataFrame)
        counts = sparse(countsIJV.I, countsIJV.J, countsIJV.V)
        
        terms = CSV.read(termsfile, DataFrame).term
        @info "SOTU + Rem loaded from cached $(datadep"sotu")"
    else
        sotu, D = loadsotu()
        ret = loadreturns()

        # merged dataset
        ixret = findall(in(sotu.Date), ret.Date)
        @assert sotu.Date == ret.Date[ixret] "sotu and return years do not lineup as expected"
        
        df = innerjoin(ret[ixret, ["Date", "Rem"]], sotu, on="Date")
     
        # save to csv file
        df |> CSV.write(csvfile)
        
        # save dtm in IJV sparse format
        counts = dtm(D)
        I, J, V = findnz(counts)
        DataFrame(I=I, J=J, V=V) |> CSV.write(dtmfile)
        
        # save terms corresponding to the columns of the counts matrix
        terms = D.terms
        DataFrame(index=1:length(terms), term=terms) |> CSV.write(termsfile)

        @info "SOTU + Rem loaded and saved to $(datadep"sotu")"
    end

    df, counts, terms
end

"""
converts a python/scipy sprase matrix to julia
Code from https://github.com/JuliaPy/PyCall.jl/issues/735#issuecomment-577336332
"""
function scipyCSC_to_julia(A)
    m, n = A.shape
    colPtr = Int[i+1 for i in PyArray(A."indptr")]
    rowVal = Int[i+1 for i in PyArray(A."indices")]
    nzVal = Vector{Float64}(PyArray(A."data"))
    B = SparseMatrixCSC{Float64,Int}(m, n, colPtr, rowVal, nzVal)
    return PyCall.pyjlwrap_new(B)
end

"""
converts a python/pandas dataframe to julia
Code from https://discourse.julialang.org/t/converting-pandas-dataframe-returned-from-pycall-to-julia-dataframe/43001/3
"""
function pd_to_df(df_pd)
    colnames = map(Symbol, df_pd.columns)
    df = DataFrame(Any[Array(df_pd[c].values) for c in colnames], colnames)
end

covarsdf, counts, terms = loadsoturet(cache=false)