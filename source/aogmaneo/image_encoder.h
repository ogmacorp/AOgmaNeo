// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
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
        size(32, 32, 1),
        radius(2)
        {}
    };

    // visible layer
    struct Visible_Layer {
        Byte_Buffer weights0; // regular
        Byte_Buffer weights1; // complement
        Byte_Buffer weights_recon; // for reconstruction

        Byte_Buffer reconstruction;
    };

    struct Params {
        float choice;
        float vigilance;
        float lr; // learning rate
        float scale;
        float rr; // reconstruction rate
        float active_ratio; // 2nd stage inhibition activity ratio
        int l_radius; // lateral 2nd stage inhibition radius
        
        Params()
        :
        choice(0.0001f),
        vigilance(0.9f),
        lr(0.5f),
        scale(2.0f),
        rr(0.05f),
        active_ratio(0.5f),
        l_radius(1)
        {}
    };

private:
    Int3 hidden_size; // size of hidden/output layer

    Int_Buffer hidden_cis; // hidden states

    Byte_Buffer hidden_learn_flags;

    Float_Buffer hidden_comparisons;

    // visible layers and associated descriptors
    Array<Visible_Layer> visible_layers;
    Array<Visible_Layer_Desc> visible_layer_descs;
    
    // --- kernels ---
    
    void forward(
        const Int2 &column_pos,
        const Array<Byte_Buffer_View> &inputs
    );

    void learn(
        const Int2 &column_pos,
        const Array<Byte_Buffer_View> &inputs
    );

    void learn_reconstruction(
        const Int2 &column_pos,
        Byte_Buffer_View inputs,
        int vli,
        unsigned long* state
    );

    void reconstruct(
        const Int2 &column_pos,
        Int_Buffer_View recon_cis,
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
        const Array<Byte_Buffer_View> &inputs, // input states
        bool learn_enabled, // whether to learn
        bool learn_recon = true // whether to learn reconstruction weights
    );

    void reconstruct(
        Int_Buffer_View recon_cis
    );

    const Byte_Buffer &get_reconstruction(
        int vli
    ) const {
        return visible_layers[vli].reconstruction;
    }

    // serialization
    long size() const; // returns size in bytes
    long state_size() const; // returns state size in bytes
    long weights_size() const; // returns weights size in bytes

    void write(
        Stream_Writer &writer
    ) const;

    void read(
        Stream_Reader &reader
    );

    void write_state(
        Stream_Writer &writer
    ) const;

    void read_state(
        Stream_Reader &reader
    );

    void write_weights(
        Stream_Writer &writer
    ) const;

    void read_weights(
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

    // merge list of image encoders and write to this one
    void merge(
        const Array<Image_Encoder*> &image_encoders,
        Merge_Mode mode
    );
};
}
