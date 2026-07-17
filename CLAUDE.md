# Project guidance for this repo

## Ask before major design changes

Do not implement a design change to the modeling/strategy logic (new
states, new signals, new thresholds representing a different mechanism,
new modules that change how a strategy behaves) without first describing
the proposed design and getting explicit confirmation. This applies even
mid-task, if a request turns out to imply a bigger change than initially
scoped.

Small, contained parameter tweaks that don't change the mechanism (e.g.
adjusting an already-agreed threshold value) don't require a fresh
confirmation round, but should still be called out clearly in the response.

When in doubt about whether something counts as "major," ask.