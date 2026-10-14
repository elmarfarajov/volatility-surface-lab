# 31. The wheel is tested on its own, without the development extras

- **Status:** Accepted
- **Date:** 2026-10-15

## Context

The tests run from the source tree with every development dependency installed:
QuantLib, mpmath, Hypothesis and the rest. A package that works only like that is not a
package. Day 9 found that the platform's report imported the gallery's house style, and
with it every gallery module, some of which import QuantLib at the top. An installed wheel
would therefore have needed a dependency it does not declare.

## Decision

- **The gallery imports its chart modules lazily**, inside the function that lists them.
  Its house style, which the report shares, needs only matplotlib.
- **A `package` job in CI** does the following:
  - builds the sdist and the wheel;
  - checks them with `twine`;
  - installs the wheel alone into a clean environment;
  - from there checks that `py.typed` ships, then runs `volsurf price`,
    `volsurf demo --quick` and `volsurf reproduce`;
  - uploads the distributions as an artifact.
- **The package declares itself typed** (`py.typed`, `Typing :: Typed`). Strict mypy covers
  every rebuilt package, the platform and the command line.

## Consequences

- **A user who installs the wheel can run the whole platform** and reproduce its result.
  The development extras are needed only for the tests, the validation suite and the
  gallery.
- **The two install paths cannot drift apart.** CI tests the installed wheel alongside the
  source tree on every push.
