using HurdleDMR
using Documenter

makedocs(
  modules = [HurdleDMR],
  clean = true,
  format = :html,
  sitename = "HurdleDMR.jl",
  authors = "Asaf Manela",
  analytics = "UA-4385132-6",
  linkcheck = !("skiplinks" in ARGS),
  pages = ["Home" => "index.md",
    "Tutorials" => "tutorials/index.md"],
  # Use clean URLs, unless built as a "local" build
  html_prettyurls = !("local" in ARGS),
  html_canonical = "https://asafmanela.github.io/HurdleDMR.jl/stable/",
)

deploydocs(
  repo = "github.com/AsafManela/HurdleDMR.jl.git",
  target = "build",
  julia = "1.0",
  osname = "linux",
  deps = nothing,
  make = nothing,
)
