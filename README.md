# ACCORD O-RAN Environment

Files:

- `config.yaml`: simulation/training parameters from Table II.
- `env.py`: network environment implementing the paper's Eqs. (5)-(15), the
  two control time scales, CIO steering, dwell masking, diurnal traffic,
  mobility, service degradation, handover, and power consumption.
- `test_env.py`: smoke test using All-on + CIO=0.
- `requirements.txt`: minimal dependencies.

## Install

```bash
pip install -r requirements.txt
```

## Test

```bash
python test_env.py
```

## Action format

At an ES epoch (`t % K == 0`):

```python
action = {
    "es": np.array([1, 1, 1, 1, 1, 1, 1]),
    "mlb": np.array([0., 0., 0., 0., 0., 0., 0.]),
}
```

At other slots:

```python
action = {
    "es": None,
    "mlb": np.array([0., 0., 0., 0., 0., 0., 0.]),
}
```

The MLB action is required every slot. The ES action is accepted only once per
`K=60` slots and is dwell-masked.

## Important implementation assumptions

The paper fixes the simulation parameters but does not fully specify:

1. the reference path loss for the log-distance model,
2. the exact outer boundary of the seven-cell cluster / boundary behavior of
   the random walk,
3. the initial dwell-timer state,
4. temporal correlation of shadowing (defaulted to one realization per episode).

Those choices are therefore explicit under the `implementation:` block of
`config.yaml`, so they can be changed without touching the paper-derived
parameters.
