using HurdleDMR
using Documenter

makedocs(
  modules = [HurdleDMR],
  clean = true,
  format = Documenter.HTML(
        # Use clean URLs, unless built as a "local" build
        prettyurls = !("local" in ARGS),
        canonical = "https://asafmanela.github.io/HurdleDMR.jl/stable/",
        analytics = "UA-4385132-6",
    ),
  sitename = "HurdleDMR.jl",
  authors = "Asaf Manela",
  linkcheck = !("skiplinks" in ARGS),
  pages = [
    "Home" => "index.md",
    "Tutorials" => "tutorials/index.md"
    ],
    )

deploydocs(
  repo = "github.com/AsafManela/HurdleDMR.jl.git",
  )
