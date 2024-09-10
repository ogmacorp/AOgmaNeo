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
        for (int c = 0; c < cs; c++) {
            for (int c2 = 0; c2 < c; c2++) { // only compute up to diagonal
                int a_at_elem = 0;

                for (int r = 0; r < rs; r++)
                    a_at_elem += p[r + c * rs] * p[r + c2 * rs];

                // exploit symmetry
                for (int r = 0; r < rs; r++) {
                    // upper triangle
                    vd[r] += a_at_elem * vs[c];

                    // lower triangle
                    int r_comp = rs - r;

                    vd[r_comp] += a_at_elem * vs[r_comp];
                }
            }
        }

        // accumlate from diagonal
        for (int c = 0; c < cs; c++) {
            int a_at_elem = 0;

            for (int c = 0; c < cs; c++) {
                int index = c + c * rs;

                a_at_elem += p[index] * p[index];
            }

            for (int r = 0; r < rs; r++)
                vd[r] += a_at_elem * vs[c];
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

