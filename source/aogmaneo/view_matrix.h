// ----------------------------------------------------------------------------
//  AOgmaNeo
//  Copyright(c) 2020-2024 Ogma Intelligent Systems Corp. All rights reserved.
//
//  This copy of AOgmaNeo is licensed to you under the terms described
//  in the AOGMANEO_LICENSE.md file included in this distribution.
// ----------------------------------------------------------------------------

#pragma once

#include "helpers.h"

namespace aon {
template<typename T>
class View_Matrix {
private:
    T* p;
    int s, rs, cs; // size, rows, columns

public:
    View_Matrix()
    :
    p(nullptr),
    s(0),
    rs(0),
    cs(0)
    {}

    View_Matrix(
        T* p,
        int rs,
        int cs
    )
    :
    p(p),
    rs(rs),
    cs(cs)
    {
        s = rs * cs;
    }

    T &operator[](
        int index
    ) {
        assert(index >= 0 && index < s);

        return p[index];
    }

    const T &operator[](
        int index
    ) const {
        assert(index >= 0 && index < s);
        
        return p[index];
    }

    T &operator()(
        int r,
        int c
    ) {
        assert(r >= 0 && r < rs);
        assert(c >= 0 && c < cs);

        return p[r + c * rs];
    }

    const T &operator()(
        int r,
        int c
    ) const {
        assert(r >= 0 && r < rs);
        assert(c >= 0 && c < cs);
        
        return p[r + c * rs];
    }

    void multiply_a_at(
        T* vs, // source vector
        T* vd // destination vector
    ) {
        // zero out vd
        for (int c = 0; c < cs; c++)
            vd[c] = 0;

        // accumulate from above diagonal (symmetric)
        for (int r = 0; r < rs; r++) {
            for (int r2 = 0; r2 < r; r2++) { // only compute up to diagonal
                int a_at_elem = 0;

                for (int c = 0; c < cs; c++)
                    a_at_elem += p[r + c * rs] * p[r2 + c * rs];

                // exploit symmetry
                for (int c = 0; c < cs; c++) {
                    // upper triangle
                    vd[c] += a_at_elem * vs[c];

                    // lower triangle
                    int c_comp = cs - c;

                    vd[c_comp] += a_at_elem * vs[c_comp];
                }
            }
        }

        // accumlate from diagonal
        for (int r = 0; r < rs; r++) {
            int a_at_elem = 0;

            for (int c = 0; c < cs; c++)
                a_at_elem += p[r + c * rs] * p[r + c * rs];

            for (int c = 0; c < cs; c++)
                vd[c] += a_at_elem * vs[c];
        }
    }

    int size() const {
        return s;
    }
    
    int num_rows() const {
        return rs;
    }

    int num_columns() const {
        return cs;
    }

    void fill(
        T value
    ) {
        for (int i = 0; i < s; i++)
            p[i] = value;
    }
};
}

