# Integration Examples

Integrations consume synchronized debug results outside CUDA Graph callbacks.
They are optional and may require dependencies that are not installed with the
core package.

`tensorboard_export.py` records one cloned `TensorSnapshot` per replay, then passes the
complete series to `export_snapshots_to_tensorboard`. The application owns the
`SummaryWriter`, synchronization, log directory, and writer shutdown. Install
TensorBoard separately and run the example from the repository root:

```bash
python examples/integrations/tensorboard_export.py --logdir /tmp/tcgd-tensorboard
```
