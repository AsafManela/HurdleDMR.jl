using HurdleDMR
using Documenter

makedocs(
  modules = [HurdleDMR],
  clean = false,
  format = :html,
  sitename = "HurdleDMR.jl",
  authors = "Asaf Manela",
  analytics = "UA-4385132-6",
  linkcheck = !("skiplinks" in ARGS),
  pages = ["Home" => "index.md"],
  # Use clean URLs, unless built as a "local" build
  html_prettyurls = !("local" in ARGS),
  html_canonical = "https://asafmanela.github.io/HurdleDMR.jl/stable/",
)

deploydocs(
  repo = "github.com/AsafManela/HurdleDMR.jl.git",
  target = "build",
  julia = "nightly",
  deps = nothing,
  make = nothing,
)
