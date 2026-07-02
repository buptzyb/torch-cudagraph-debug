# Integration Examples

Integrations consume synchronized debug results outside CUDA Graph callbacks.
They are optional and may require dependencies that are not installed with the
core package.

`tensorboard_export.py` retains one independent `TensorProbeSnapshot` per replay,
using the probe's replay index as the TensorBoard step, then passes the complete
series to `export_snapshots_to_tensorboard`. It passes the replay stream to
`snapshot()` so each query waits only for that stream. The application owns the
`SummaryWriter`, stream choice, log directory, and writer shutdown. Install
TensorBoard separately and run the example from the repository root:

```bash
python examples/integrations/tensorboard_export.py --logdir /tmp/tcgd-tensorboard
```
