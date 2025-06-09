// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2025 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "hierarchy.h"

using namespace aon;

void Hierarchy::init_random(
    const Array<IO_Desc> &io_descs,
    const Array<Layer_Desc> &layer_descs,
    int delay_capacity
) {
    assert(delay_capacity >= 0);

    // create layers
    encoders.resize(layer_descs.size());
    decoders.resize(layer_descs.size());
    hidden_cis_prev.resize(layer_descs.size());
    feedback_cis_prev.resize(layer_descs.size() - 1);

    // cache input sizes
    io_sizes.resize(io_descs.size());
    io_types.resize(io_descs.size());

    int num_predictions = 0;

    for (int i = 0; i < io_sizes.size(); i++) {
        io_sizes[i] = io_descs[i].size;
        io_types[i] = static_cast<Byte>(io_descs[i].type);

        if (io_descs[i].type == prediction)
            num_predictions++;
    }

    // iterate through layers
    for (int l = 0; l < layer_descs.size(); l++) {
        // create sparse coder visible layer descriptors
        Array<Encoder::Visible_Layer_Desc> e_visible_layer_descs;

        // if first layer
        if (l == 0) {
            e_visible_layer_descs.resize(io_sizes.size() + (layer_descs[l].recurrent_radius > -1));

            for (int i = 0; i < io_sizes.size(); i++) {
                e_visible_layer_descs[i].size = io_sizes[i];
                e_visible_layer_descs[i].radius = io_descs[i].up_radius;
            }

            decoders[l].resize(num_predictions);

            i_indices.resize(io_sizes.size());
            d_indices = Int_Buffer(io_sizes.size(), -1);

            // create decoders
            int d_index = 0;

            for (int i = 0; i < io_sizes.size(); i++) {
                if (io_descs[i].type == prediction) {
                    // decoder visible layer descriptors
                    Array<Decoder::Visible_Layer_Desc> d_visible_layer_descs(1 + (l < encoders.size() - 1));

                    d_visible_layer_descs[0].size = layer_descs[l].hidden_size;
                    d_visible_layer_descs[0].radius = io_descs[i].down_radius;

                    if (l < encoders.size() - 1)
                        d_visible_layer_descs[1] = d_visible_layer_descs[0];

                    decoders[l][d_index].init_random(io_sizes[i], io_descs[i].num_dendrites_per_cell, d_visible_layer_descs);

                    i_indices[d_index] = i;
                    d_indices[i] = d_index;
                    d_index++;
                }
            }
        }
        else {
            e_visible_layer_descs.resize(1 + (layer_descs[l].recurrent_radius > -1));

            e_visible_layer_descs[0].size = layer_descs[l - 1].hidden_size;
            e_visible_layer_descs[0].radius = layer_descs[l].up_radius;

            decoders[l].resize(1);

            // decoder visible layer descriptors
            Array<Decoder::Visible_Layer_Desc> d_visible_layer_descs(1 + (l < encoders.size() - 1));

            d_visible_layer_descs[0].size = layer_descs[l].hidden_size;
            d_visible_layer_descs[0].radius = layer_descs[l].down_radius;

            if (l < encoders.size() - 1)
                d_visible_layer_descs[1] = d_visible_layer_descs[0];

            // create decoders
            decoders[l][0].init_random(layer_descs[l - 1].hidden_size, layer_descs[l].num_dendrites_per_cell, d_visible_layer_descs);
        }

        if (layer_descs[l].recurrent_radius > -1) {
            int last_index = e_visible_layer_descs.size() - 1;

            e_visible_layer_descs[last_index].size = layer_descs[l].hidden_size;
            e_visible_layer_descs[last_index].radius = layer_descs[l].recurrent_radius;
        }
        
        // create the sparse coding layer
        encoders[l].init_random(layer_descs[l].hidden_size, e_visible_layer_descs);

        hidden_cis_prev[l] = encoders[l].get_hidden_cis();

        if (l < encoders.size() - 1)
            feedback_cis_prev[l] = encoders[l].get_hidden_cis();
    }

    // initialize params
    params.layers = Array<Layer_Params>(layer_descs.size());
    params.ios = Array<IO_Params>(io_descs.size());

    // delay buffer
    states_size = 0;
    states.resize(delay_capacity);

    int size_of_state = state_size();

    for (int t = 0; t < states.size(); t++)
        states[t].resize(size_of_state);
}

void Hierarchy::step(
    const Array<Int_Buffer_View> &input_cis,
    bool learn_enabled,
    int delay
) {
    assert(params.layers.size() == encoders.size());
    assert(params.ios.size() == io_sizes.size());

    assert(delay >= 0);

    bool needs_reset = false;

    // if updating a past state
    if (delay > 0 && delay < states_size) {
        // set old state
        Buffer_Reader state_reader;
        state_reader.buffer = states[delay];

        read_state(state_reader);

        needs_reset = true;
    }

    // set importances from params
    for (int i = 0; i < io_sizes.size(); i++)
        encoders[0].get_visible_layer(i).importance = params.ios[i].importance;

    // forward
    for (int l = 0; l < encoders.size(); l++) {
        // copy to prev
        hidden_cis_prev[l] = encoders[l].get_hidden_cis();

        if (l < encoders.size() - 1)
            feedback_cis_prev[l] = decoders[l + 1][0].get_hidden_cis();

        Array<Int_Buffer_View> layer_input_cis(encoders[l].get_num_visible_layers());

        if (l == 0) {
            for (int i = 0; i < io_sizes.size(); i++)
                layer_input_cis[i] = input_cis[i];
        }
        else
            layer_input_cis[0] = encoders[l - 1].get_hidden_cis();

        if (is_layer_recurrent(l)) {
            int last_index = encoders[l].get_num_visible_layers() - 1;

            layer_input_cis[last_index] = hidden_cis_prev[l];

            encoders[l].get_visible_layer(last_index).importance = params.layers[l].recurrent_importance;
        }

        // activate sparse coder
        encoders[l].step(layer_input_cis, learn_enabled, params.layers[l].encoder);
    }

    // backward
    for (int l = decoders.size() - 1; l >= 0; l--) {
        Array<Int_Buffer_View> layer_input_cis(1 + (l < encoders.size() - 1));

        if (learn_enabled) {
            layer_input_cis[0] = hidden_cis_prev[l];
            
            if (l < encoders.size() - 1) {
                // learn on feedback
                layer_input_cis[1] = feedback_cis_prev[l];

                for (int d = 0; d < decoders[l].size(); d++)
                    decoders[l][d].learn(layer_input_cis, (l == 0 ? input_cis[i_indices[d]] : encoders[l - 1].get_hidden_cis()), (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));

                if (params.anticipation) {
                    // learn on actual
                    layer_input_cis[1] = encoders[l].get_hidden_cis();

                    for (int d = 0; d < decoders[l].size(); d++) {
                        // need to re-activate for this
                        decoders[l][d].activate(layer_input_cis, (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));

                        decoders[l][d].learn(layer_input_cis, (l == 0 ? input_cis[i_indices[d]] : encoders[l - 1].get_hidden_cis()), (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));
                    }
                }
            }
            else {
                for (int d = 0; d < decoders[l].size(); d++)
                    decoders[l][d].learn(layer_input_cis, (l == 0 ? input_cis[i_indices[d]] : encoders[l - 1].get_hidden_cis()), (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));
            }
        }

        layer_input_cis[0] = encoders[l].get_hidden_cis();
        
        if (l < encoders.size() - 1)
            layer_input_cis[1] = decoders[l + 1][0].get_hidden_cis();

        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].activate(layer_input_cis, (l == 0 ? params.ios[i_indices[d]].decoder : params.layers[l].decoder));
    }

    if (needs_reset) {
        // set latest state back
        Buffer_Reader state_reader;
        state_reader.buffer = states[0];

        read_state(state_reader);
    }
    else if (states.size() > 0) { // not doing a delayed update, so back up fresh "real" states
        states.push_front();
        
        if (states_size < states.size())
            states_size++;

        // save state
        if (delay == 0) {
            Buffer_Writer state_writer;
            state_writer.buffer = states[0];

            write_state(state_writer);
        }
    }
}

void Hierarchy::clear_state() {
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].clear_state();
        
        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].clear_state();
    }

    states_size = 0;
}

long Hierarchy::size() const {
    long size = 3 * sizeof(int) + io_sizes.size() * sizeof(Int3) + io_types.size() * sizeof(Byte) + i_indices.size() * sizeof(int) + d_indices.size() * sizeof(int);

    for (int l = 0; l < encoders.size(); l++) {
        size += encoders[l].size();

        for (int d = 0; d < decoders[l].size(); d++)
            size += decoders[l][d].size();
    }

    // params
    size += encoders.size() * sizeof(Layer_Params);
    size += io_sizes.size() * sizeof(IO_Params);

    // state buffer
    size += 2 * sizeof(int) + states.size() * state_size();

    return size;
}

long Hierarchy::state_size() const {
    long size = 0;

    for (int l = 0; l < encoders.size(); l++) {
        size += encoders[l].state_size();
        
        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            size += decoders[l][d].state_size();
    }

    return size;
}

long Hierarchy::weights_size() const {
    long size = 0;

    for (int l = 0; l < encoders.size(); l++) {
        size += encoders[l].weights_size();

        for (int d = 0; d < decoders[l].size(); d++)
            size += decoders[l][d].weights_size();
    }

    return size;
}

void Hierarchy::write(
    Stream_Writer &writer
) const {
    int num_layers = encoders.size();

    writer.write(&num_layers, sizeof(int));

    int num_io = io_sizes.size();

    writer.write(&num_io, sizeof(int));

    int num_predictions = decoders[0].size();

    writer.write(&num_predictions, sizeof(int));

    writer.write(&io_sizes[0], num_io * sizeof(Int3));
    writer.write(&io_types[0], num_io * sizeof(Byte));

    writer.write(&i_indices[0], i_indices.size() * sizeof(int));
    writer.write(&d_indices[0], d_indices.size() * sizeof(int));

    for (int l = 0; l < num_layers; l++) {
        encoders[l].write(writer);

        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].write(writer);
    }
    
    // params
    for (int l = 0; l < encoders.size(); l++)
        writer.write(&params.layers[l], sizeof(Layer_Params));

    for (int i = 0; i < io_sizes.size(); i++)
        writer.write(&params.ios[i], sizeof(IO_Params));

    writer.write(&params.anticipation, sizeof(Byte));

    int delay_capacity = states.size();

    writer.write(&delay_capacity, sizeof(int));
    writer.write(&states_size, sizeof(int));

    for (int t = 0; t < states.size(); t++)
        writer.write(&states[t][0], states[t].size() * sizeof(Byte));
}

void Hierarchy::read(
    Stream_Reader &reader
) {
    int num_layers;

    reader.read(&num_layers, sizeof(int));

    int num_io;

    reader.read(&num_io, sizeof(int));

    int num_predictions;

    reader.read(&num_predictions, sizeof(int));

    io_sizes.resize(num_io);
    io_types.resize(num_io);

    reader.read(&io_sizes[0], num_io * sizeof(Int3));
    reader.read(&io_types[0], num_io * sizeof(Byte));

    encoders.resize(num_layers);
    decoders.resize(num_layers);
    hidden_cis_prev.resize(num_layers);
    feedback_cis_prev.resize(num_layers - 1);

    i_indices.resize(num_io);
    d_indices.resize(num_io);

    reader.read(&i_indices[0], i_indices.size() * sizeof(int));
    reader.read(&d_indices[0], d_indices.size() * sizeof(int));
    
    for (int l = 0; l < num_layers; l++) {
        encoders[l].read(reader);
        
        decoders[l].resize(l == 0 ? num_predictions : 1);

        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].read(reader);

        hidden_cis_prev[l] = encoders[l].get_hidden_cis();

        if (l < encoders.size() - 1)
            feedback_cis_prev[l] = encoders[l].get_hidden_cis();
    }

    params.layers.resize(num_layers);
    params.ios.resize(num_io);

    for (int l = 0; l < num_layers; l++)
        reader.read(&params.layers[l], sizeof(Layer_Params));

    for (int i = 0; i < num_io; i++)
        reader.read(&params.ios[i], sizeof(IO_Params));

    reader.read(&params.anticipation, sizeof(Byte));

    int delay_capacity;

    reader.read(&delay_capacity, sizeof(int));
    reader.read(&states_size, sizeof(int));

    states.resize(delay_capacity);

    int size_of_state = state_size();

    for (int t = 0; t < states.size(); t++) {
        states[t].resize(size_of_state);

        reader.read(&states[t][0], states[t].size() * sizeof(Byte));
    }
}

void Hierarchy::write_state(
    Stream_Writer &writer
) const {
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].write_state(writer);

        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].write_state(writer);
    }
}

void Hierarchy::read_state(
    Stream_Reader &reader
) {
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].read_state(reader);
        
        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].read_state(reader);
    }
}

void Hierarchy::write_weights(
    Stream_Writer &writer
) const {
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].write_weights(writer);

        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].write_weights(writer);
    }
}

void Hierarchy::read_weights(
    Stream_Reader &reader
) {
    for (int l = 0; l < encoders.size(); l++) {
        encoders[l].read_weights(reader);
        
        // decoders
        for (int d = 0; d < decoders[l].size(); d++)
            decoders[l][d].read_weights(reader);
    }
}

void Hierarchy::merge(
    const Array<Hierarchy*> &hierarchies,
    Merge_Mode mode
) {
    Array<Encoder*> merge_encoders(hierarchies.size());
    Array<Decoder*> merge_decoders(hierarchies.size());

    for (int l = 0; l < encoders.size(); l++) {
        for (int h = 0; h < hierarchies.size(); h++)
            merge_encoders[h] = &hierarchies[h]->encoders[l];

        encoders[l].merge(merge_encoders, mode);

        // decoders
        for (int d = 0; d < decoders[l].size(); d++) {
            for (int h = 0; h < hierarchies.size(); h++)
                merge_decoders[h] = &hierarchies[h]->decoders[l][d];

            decoders[l][d].merge(merge_decoders, mode);
        }
    }
}
