import Lake
open Lake DSL

package «DeflationaryAgentProofs» where
  moreLeanArgs := #["-DautoImplicit=false", "-Dlinter.missingDocs=false"]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.25.0"

