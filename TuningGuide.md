# AOgmaNeo parameter tuning guide

## Structural

Structural parameters are set in the descriptors upon the creation of a hierarchy. There are 2 descriptors, IODesc and LayerDesc.

**Note:** Radius parameters are used to determine receptive field sizes. Receptive fields are squares, and the diameter is `2 * radius + 1`. The area is `diameter ^ 2` columns.

**Note:** In C++ we use Pascal case for structures and snake case for everything else. In Python, we use camel case for structures and snake case for everything else (PEP8). C++ Example structure: Layer_Desc. Python: LayerDesc.

### IODesc

Describes an input/output

- size: Size (Int3) of the input/output layer (width, height, column_size).

- type: Enum type of the IO layer. Can be none (input only), prediction, and actor (RL).

- up_radius: Receptive field radius from this IO layer to the first layer (encoder radius). Usually 2

- down_radius: Receptive field radius from the first layer to this IO layer (decoder/actor radius). Usually 2

- history_capacity: (RL only) the credit assignment horizon. Usually around 128 or 256

### LayerDesc

Describes a higher (not IO) layer

- hidden_size: Size (Int3) of the encoder's hidden state for that layer (width, height, column_size). Usually width/height are in the range [4, 16] and column_size is in the range [16, 64] depending on the requirements.

- up_radius: Receptive field radius from this previous (lower) layer to this layer (encoder radius). Usually 2

- down_radius: Receptive field radius from this layer to the previous (lower) layer (decoder/actor radius). Usually 2

- ticks_per_update: Exponential Memory stride size. Usually 2 (doubling memory every layer)

- temporal_horizon: Memory window size of the layer, must be >= ticks_per_update. Usually also 2

## Runtime-adjustable

Runtime adjustable parameters are accessed via the hierarchy's params member. params contains ios (IO layer parameters) and layers (higher layer parameters) members.

Example access (Python):

```python
h.params.ios[2].decoder.lr = 0.1
h.params.ios[1].actor.alr = 0.01

h.params.layers[3].encoder.lr = 0.05
h.params.layers[4].decoder.scale = 64.0
```

### IOParams

- decoder: Decoder parameters

- actor: Actor parameters

- importance: Importance scaling of this IO layer's input. Affects encoding, defaults to 1. Can be used to tweak the relative influences of the inputs, which can accelerate learning if properly adjusted.

### LayerParams

- decoder: Decoder parameters

- encoder: Encoder parameters

### DecoderParams

- scale: Range scaling for byte-weights. Unlikely you need to change this, best left as-is.

- lr: Learning rate, usually in [0.0, 1.0] range. Can be > 1 in some instances!

- gcurve: Gate curve, used to prevent forgetting. Also unlikely that you will need to change this.

### EncoderParams

Same as decoder.

- scale: Range scaling for byte-weights. Unlikely you need to change this, best left as-is.

- lr: Learning rate, usually in [0.0, 1.0] range. Can be > 1 in some instances!

- gcurve: Gate curve, used to prevent forgetting. Also unlikely that you will need to change this.

### ActorParams:

- vlr: Value function learning rate. Around 0.01, cannot be > 1!

- alr: Actor learning rate, Around 0.01, can be > 1 (but this is very rare).

- discount: Reinforcement learning discounting factor (lambda). Around 0.99. Must be < 1.

- min_steps: Minimum samples before the sample can be used in training. Must be < history_capacity, otherwise it won't learn. Usually around 8.

- history_iters: Number of iterations over the history to assign credit with. Usually 8 or 16.

## Assorted Tips

- be aware of the receptive field coverage. The radii can be increased to increase coverage, at the cost of more compute. You can also add more layers to bridge spatial gaps in the receptive fields (higher level processing).

- the defaults are likely fine for most tasks.

- sometimes it is desirable to increase the actor alr parameter when in mimic mode, otherwise the smaller learning rate it has by default will make learning take forever (the default is made for RL).

- generally it is more efficient to use more layers than to increase temporal_horizon per-layer to get more memory.

- on tasks with highly reward variance w.r.t. timesteps, you may want to decrease the discount factor. On the other end, if the task is quite continuous and rewards are sparse, you may want to increase the discount factor.
