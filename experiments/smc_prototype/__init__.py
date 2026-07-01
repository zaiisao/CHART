"""FIVO/SMC bar-pointer prototype: fixed prior + strong learned emission + particle-filter deploy.

See train_smc.py for the entry point and the module docstrings in bar_pointer_smc.py / emission.py /
evaluate_smc.py for the design. This is a self-contained prototype (does not modify the VBPM package)
built to compare against the plain amortized-encoder VBPM (model/bar_pointer_vae.py) on the SAME
leak-test protocol (evaluate.py's real/shuffle/zero conditions).
"""
