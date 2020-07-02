﻿#pragma once


#include "tmesh.h"

#include <time.h>
#include <vector>
#include <map>
#include "tqueue.h"

#pragma unmanaged

namespace marchingcubes
{

//
// Marching cubes法
//
// threshold と 画素値が同じ場合に
// 複数頂点が一点に縮退する場合がある．
// この時は一部の面の法線が計算できなくなるので注意

void t_MarchingCubes(
    const EVec3i &vRes,
    const EVec3f &vPitch,
    const short   *volume,
    const short   thresh,
    const int    *minIdx,
    const int    *maxIdx,
    std::vector<EVec3f> &Vs, //output vertices
    std::vector<TPoly > &Ps  //output polygons 
);


void t_MarchingCubes_PolygonSoup(
    const EVec3i &vRes,
    const EVec3f &vPitch,
    const short   *volume,
    const short   thresh,
    const int    *minIdx,
    const int    *maxIdx,

    int     &polygon_verts_num, 
    EVec3f* &polygon_verts     //v0,v1,v2,  v0,v1,v2, ...
);


void t_MarchingCubes_PolygonSoup(
    const EVec3i &vRes,
    const EVec3f &vPitch,
    const short   *volume,
    const short   thresh,
    const int    *minIdx,
    const int    *maxIdx,
    
    TTriangleSoup &triangle_soup
);


void t_MarchingCubes(
    const EVec3i &vRes,
    const EVec3f &vPitch,
    const short  *volume,
    const short   thresh,
    const int    *minIdx,
    const int    *maxIdx,
    TMesh &mesh);
};



#pragma managed
