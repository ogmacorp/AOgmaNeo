# AOgmaNeo variable naming reference

## Encoder/Decoder

**vl** - visible layer

**vld** - visible layer descriptor

**hc** - hidden column index. Range [0, hidden_size.z). Used to index inside of one hidden column

**vc** - visible column index. Range [0, vld.size.z). Used to index inside of one visible/input column

**hidden_cell_index** - index derived from 3D cell position in hidden layer

**count** - counter of number of inputs typically, used for normalization

**radius** - receptive field radius

**diam** - receptive field diameter, diam = 2 * radius + 1

**area** - receptive field area, area = diam * diam

**h_to_v** - hidden to visible projection factors. Scale by this to go from hidden column position to visible column position using the project function

**project** - function that scales positions between visible/hidden layers

**visible_center** - center of the receptive field onto the input/visible layer

**field_lower_bound** - lower bound of the receptive field, disregarding edge bounding (allows it to go over edge of CSDR)

**iter_lower_bound/iter_upper_bound** - lower and upper bounds of the receptive field, but with clamping to the edge of the CSDR

**hidden_stride** - amount of weight indices strided when the hidden cell index changes by 1, used for partial index calculation

**in_ci** - input column index, an element of the input/visible CSDR

**offset** - position into the receptive field, in range [0, diam) for both elements (x and y)

**wi_offset** - partially computed weight index, just missing the influence of the hidden cell

**wi** - weight index, the final index into the weight matrix

**wi_start** - partially computed weight index, just missing the visible cell (vc) influence

**max_index** - index of highest activation

**max_activation** - the highest activation itself

**total** - like count, but usually a float for doing softmax

**delta** - typically a weight increment

**state** - RNG state, or state needed for working memory, depending on where it is

## Actor

TODO

## Hierarchy

**ticks** - clocks for exponential memory

**ticks_per_update** - number of ticks before a layer activates

**i_indices** - for mapping the decoders of the first layer to their corresponding input indices (io index)

**d_indices** - for mapping the io indices to their corresponding decoders, if it exists (-1 otherwise)

**updates** - whether or not a layer "ticked" (updated) this step

**io_types** - IO type of each IO layer index ("port")

**io_sizes** - size (Int3) of each IO layer ("port")

**t** - time index

**i** - input index

**d** - decoder index

## Suffixes and Prefixes

**prev** - indicates a value from the previous timestep

**next** - indicates a value for the next timestep

**base** - cached partial computation, typically for random sub-seeds (RNG)

**acts** - activations

**cis** - column indices

**visible/input** - relating to the visible/input portion of an encoder/decoder

**hidden** - relating to the hidden/output portion of an encoder/decoder

**probs** - probabilities

**size** - usually used to count the size in bytes of an encoder/decoder/hierarchy/image_encoder for serialization

**e** - encoder

**d** - decoder

**a** - actor

## Misc

**params** - paramters, kept separately from encoders/decoders so they can be shared

**importance** - used to scale the relative influence of input/visible fields
