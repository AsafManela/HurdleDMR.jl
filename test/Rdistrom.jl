# using Rdistrom
using RDatasets
fgl = dataset("MASS","fgl")

covars = fgl[:,1:9]
categories = fgl[:Type]
function collapse(categories)
  cat_indicators = DataFrame()
  unique_categories=union(categories)
  for c in unique_categories
       cat_indicators[symbol(c)] = map(x->ifelse(x == c,1,0),categories)
  end
  (unique_categories,cat_indicators)
end

(unique_categories,cat_indicators) = collapse(categories)
unique_categories

counts=sparse(convert(Array{Int64,2},cat_indicators))

fits = dmr(covars, counts, verb=1; nlocal_workers=2, gamma=0)

dmrplots(fits,unique_categories)

z = srproj(fits,counts)
z2 = srproj2(fits,counts)
z==z2

z = srproj(fits,counts,2)
z2 = srproj2(fits,counts,2)
z==z2
