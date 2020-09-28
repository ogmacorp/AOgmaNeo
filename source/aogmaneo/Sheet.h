// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "Predictor.h"
#include "Actor.h"

namespace aon {
class Sheet {
public:
    struct IODesc {
        Int3 size;

        int radius;
    };

    Actor actor;
    Array<Predictor> predictors;
    IntBuffer actorHiddenCsPrev;
    FloatBuffer actorHiddenErrors;

    // Create a randomly initialized hierarchy
    void initRandom(
        const Array<IODesc> &inputDescs,
        int recurrentRadius,
        const Array<IODesc> &outputDescs,
        Int3 actorSize
    );

    void step(
        const Array<const IntBuffer*> &inputCs,
        const Array<const IntBuffer*> &targetCs,
        int subSteps,
        bool learnEnabled
    );

    // Serialization
    void write(
        StreamWriter &writer
    ) const;

    void read(
        StreamReader &reader
    );

    // Retrieve predictions
    const IntBuffer &getPredictionCs(
        int i // Index of input layer to get predictions for
    ) const {
        return predictors[i].getHiddenCs();
    }
};
} // namespace aon
