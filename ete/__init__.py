"""Entity-Tracking Externalization probe.

Tests whether forcing entity-state tracking into an explicit chain of thought
(externalization) changes a model's accuracy on PUT/REMOVE/MOVE tasks compared
to letting it answer directly (parallel final-token aggregation).

The hypothesis under test comes from the synthesis page
`test-time-sampling-vs-retraining-ood`: if externalization *routes around* the
fragile global-suppression REMOVE mechanism documented in
arXiv:2605.30233, then the chain-of-thought condition should show a
disproportionate accuracy gain on REMOVE-heavy items and fewer "ghost"
(suppression-failure) errors.
"""

__version__ = "0.1.0"
