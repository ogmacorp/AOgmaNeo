// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2023 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "helpers.h"

namespace aon {
// image coder
class Image_Encoder {
public:
    // visible layer descriptor
    struct Visible_Layer_Desc {
        Int3 size; // size of input

        int radius; // radius onto input

        // defaults
        Visible_Layer_Desc()
        :
        size(4, 4, 16),
        radius(2)
        {}
    };

    // visible layer
    struct Visible_Layer {
        Byte_Buffer weights0;
        Byte_Buffer weights1;

        Byte_Buffer reconstruction;
    };

    struct Params {
        float choice;
        float vigilance;
        float lr; // learning rate
        
        Params()
        :
        choice(0.1f),
        vigilance(0.95f),
        lr(0.1f)
        {}
    };

private:
    Int3 hidden_size; // size of hidden/output layer

    Int_Buffer hidden_cis; // hidden states

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;
    
    // --- kernels ---
    
    void forward(
        const Int2 &column_pos,
        const Array<const Byte_Buffer*> &inputs,
        bool learn_enabled
    );

    void reconstruct(
        const Int2 &column_pos,
        const Int_Buffer* recon_cis,
        int vli
    );

public:
    Params params;

    void init_random(
        const Int3 &hidden_size, // hidden/output size
        const Array<Visible_Layer_Desc> &visible_layer_descs // descriptors for visible layers
    );

    // activate the sparse coder (perform sparse coding)
    void step(
        const Array<const Byte_Buffer*> &inputs, // input states
        bool learn_enabled // whether to learn
    );

    void reconstruct(
        const Int_Buffer* recon_cis
    );

    const Byte_Buffer &get_reconstruction(
        int vli
    ) const {
        return visible_layers[vli].reconstruction;
    }

    // serialization
    int size() const; // returns size in bytes

    void write(
        Stream_Writer &writer
    ) const;

    void read(
        Stream_Reader &reader
    );

    // get the number of visible layers
    int get_num_visible_layers() const {
        return visible_layers.size();
    }

    // get a visible layer
    const Visible_Layer &get_visible_layer(
        int vli // index of visible layer
    ) const {
        return visible_layers[vli];
    }

    // get a visible layer descriptor
    const Visible_Layer_Desc &get_visible_layer_desc(
        int vli // index of visible layer
    ) const {
        return visible_layer_descs[vli];
    }

    // get the hidden states
    const Int_Buffer &get_hidden_cis() const {
        return hidden_cis;
    }

    // get the hidden size
    const Int3 &get_hidden_size() const {
        return hidden_size;
    }
};
}
