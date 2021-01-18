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
    struct InputDesc {
        Int3 size;

        int radius;

        bool recurrent;

        InputDesc()
        :
        size(4, 4, 16),
        radius(2),
        recurrent(false)
        {}
    };

    struct OutputDesc {
        Int3 size;

        int radius;

        OutputDesc()
        :
        size(4, 4, 16),
        radius(2)
        {}
    };

    Actor actor;
    Array<Predictor> predictors;
    IntBuffer actorHiddenCIsPrev;
    FloatBuffer actorHiddenErrors;

    // Create a randomly initialized hierarchy
    void initRandom(
        const Array<InputDesc> &inputDescs,
        int recurrentRadius,
        const Array<OutputDesc> &outputDescs,
        const Int3 &actorSize
    );

    void step(
        const Array<const IntBuffer*> &inputCIs,
        const Array<const IntBuffer*> &targetCIs,
        int subSteps,
        bool learnEnabled,
        bool clearState
    );

    void step(
        const Array<const IntBuffer*> &inputCIs,
        const Array<const IntBuffer*> &targetCIs,
        Array<IntBuffer> &intermediates,
        bool learnEnabled,
        bool clearState
    );

    // Serialization
    void write(
        StreamWriter &writer
    ) const;

    void read(
        StreamReader &reader
    );

    // Retrieve predictions
    const IntBuffer &getPredictionCIs(
        int i // Index of input layer to get predictions for
    ) const {
        return predictors[i].getHiddenCIs();
    }
};
} // namespace aon
