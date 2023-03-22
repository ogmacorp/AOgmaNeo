// ----------------------------------------------------------------------------
//  aogma_neo
//  copyright(c) 2020-2023 ogma intelligent systems corp. all rights reserved.
//
//  this copy of aogma_neo is licensed to you under the terms described
//  in the aogmaneo_license.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "helpers.h"

namespace aon {
// image coder
class image_encoder {
public:
    // visible layer descriptor
    struct visible_layer_desc {
        int3 size; // size of input

        int radius; // radius onto input

        // defaults
        visible_layer_desc()
        :
        size(4, 4, 16),
        radius(2)
        {}
    };

    // visible layer
    struct visible_layer {
        byte_buffer protos;

        byte_buffer reconstruction;
    };

private:
    int3 hidden_size; // size of hidden/output layer

    float_buffer hidden_rates;

    int_buffer hidden_cis; // hidden states

    // visible layers and associated descriptors
    array<visible_layer> visible_layers;
    array<visible_layer_desc> visible_layer_descs;
    
    // --- kernels ---
    
    void forward(
        const int2 &column_pos,
        const array<const byte_buffer*> &inputs,
        bool learn_enabled
    );

    void reconstruct(
        const int2 &column_pos,
        const int_buffer* recon_cis,
        int vli
    );

public:
    float lr;

    // defaults
    image_encoder()
    :
    lr(0.05f)
    {}

    void init_random(
        const int3 &hidden_size, // hidden/output size
        const array<visible_layer_desc> &visible_layer_descs // descriptors for visible layers
    );

    // activate the sparse coder (perform sparse coding)
    void step(
        const array<const byte_buffer*> &inputs, // input states
        bool learn_enabled // whether to learn
    );

    void reconstruct(
        const int_buffer* recon_cis
    );

    const byte_buffer &get_reconstruction(
        int vli
    ) const {
        return visible_layers[vli].reconstruction;
    }

    // serialization
    int size() const; // returns size in bytes

    void write(
        stream_writer &writer
    ) const;

    void read(
        stream_reader &reader
    );

    // get the number of visible layers
    int get_num_visible_layers() const {
        return visible_layers.size();
    }

    // get a visible layer
    const visible_layer &get_visible_layer(
        int vli // index of visible layer
    ) const {
        return visible_layers[vli];
    }

    // get a visible layer descriptor
    const visible_layer_desc &get_visible_layer_desc(
        int vli // index of visible layer
    ) const {
        return visible_layer_descs[vli];
    }

    // get the hidden states
    const int_buffer &get_hidden_cis() const {
        return hidden_cis;
    }

    // get the hidden size
    const int3 &get_hidden_size() const {
        return hidden_size;
    }
};
} // namespace aon
