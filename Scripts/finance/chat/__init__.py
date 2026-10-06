"""Finance chat: a declarative analysis spec run by deterministic code, with a tool-calling chat engine.

Modules: ``data`` (gold transactions, cache, catalog), ``profile`` (data profile: coverage,
usable/limited dimensions, example questions), ``merchants`` (descriptor
normalization), ``analysis_spec`` (AnalysisSpec, periods, repair), ``executor``
(run_analysis, narrative number guard), ``charts`` (plotly figures), ``tools`` (model tool
definitions and dispatch), ``prompts`` (chat instruction text), ``session`` (ChatSession), and
``engine`` (run_turn: the model/tool loop and UI events).
"""
