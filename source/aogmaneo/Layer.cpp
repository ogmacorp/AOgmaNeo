// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2022 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#include "Layer.h"

using namespace aon;

void Layer::initRandom(
    const Int3 &hiddenSize,
    const Array<Encoder::VisibleLayerDesc> &visibleLayerDescs,
    int pRadius,
    bool hasFeedBack
) {
    enc.initRandom(hiddenSize, visibleLayerDescs);

    Array<Predictor::VisibleLayerDesc> pVisibleLayerDescs(hasFeedBack ? 2 : 1);

    pVisibleLayerDescs[0].size = hiddenSize;
    pVisibleLayerDescs[0].radius = pRadius;

    if (hasFeedBack)
        pVisibleLayerDescs[1] = pVisibleLayerDescs[0];

    pred.initRandom(hiddenSize, pVisibleLayerDescs);

    predInputCIs.resize(pVisibleLayerDescs.size());
}

void Layer::stepUp(
    const Array<const IntBuffer*> &inputCIs,
    bool learnEnabled
) {
    enc.step(inputCIs, learnEnabled);
}

void Layer::stepDown(
    const IntBuffer* feedBackCIs,
    bool learnEnabled
) {
    if (learnEnabled)
        pred.learn(&enc.getHiddenCIs());

    predInputCIs[0] = &enc.getHiddenCIs();

    if (feedBackCIs != nullptr) {
        assert(pred.getNumVisibleLayers() == 2);

        predInputCIs[1] = feedBackCIs;
    }

    pred.activate(predInputCIs);
}

int Layer::size() const {
    return enc.size() + pred.size();
}

int Layer::stateSize() const {
    return enc.stateSize() + pred.stateSize();
}

void Layer::write(
    StreamWriter &writer
) const {
    enc.write(writer);
    pred.write(writer);
}

void Layer::read(
    StreamReader &reader
) {
    enc.read(reader);
    pred.read(reader);

    predInputCIs.resize(pred.getNumVisibleLayers());
}

void Layer::writeState(
    StreamWriter &writer
) const {
    enc.writeState(writer);
    pred.writeState(writer);
}

void Layer::readState(
    StreamReader &reader
) {
    enc.readState(reader);
    pred.readState(reader);
}

