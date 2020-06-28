#pragma once
#pragma warning(disable : 4996)

//stl
#include <iostream>
#include <list>
#include <vector>
#include <set>
#include <map>

#include "tmath.h"
#include "tqueue.h"
#include "OglForCLI.h"

#pragma unmanaged


//左回り
//
//  +v1 --e1--  + v2
//  |           |
//  e0          e2
//  |           |
//  +v0 --e3--  + v3
//   

class TQuad
{
public:
  int vi[4], ei[4];

  TQuad(const EVec4i &_vi, const EVec4i &_ei) {
    for ( int i = 0; i < 4; ++i) vi[i] = _vi[i];
    for ( int i = 0; i < 4; ++i) ei[i] = _ei[i];
  }

  TQuad(const TQuad &p) { Set(p); }
  TQuad& operator= (const TQuad  &v) { Set(v); return *this; }

  void Set(const TQuad &p) {
    memcpy(vi, p.vi, sizeof(int) * 4);
    memcpy(ei, p.ei, sizeof(int) * 4);
  }
};



//       v1
//       |
//quad_l |  quad_r
//       |
//       v0
// note v0 < v1
class TQuadEdge
{
public:
  int vi[2];
  int quad_l, quad_r;

  TQuadEdge(int v0 = 0, int v1 = 0, int left = -1, int right = -1) {
    vi[0] = std::min(v0,v1); 
    vi[1] = std::max(v0,v1); 
    quad_l = left;
    quad_r = right;
  }

  TQuadEdge(const TQuadEdge &p) { Set(p); }
  TQuadEdge& operator= (const TQuadEdge  &v) { Set(v); return *this; }

  void Set(const TQuadEdge &p) {
    memcpy(vi, p.vi, sizeof(int) * 2);
    quad_l = p.quad_l;
    quad_r = p.quad_r;
  }
};






// very simple mesh representation 

class TQuadMesh
{

public:
  std::vector<EVec3f> m_verts;
  std::vector<EVec3f> m_norms;
  std::vector<TQuad > m_quads;
  std::vector<TQuadEdge> m_edges;


  TQuadMesh()
  {
    m_verts.clear();
    m_norms.clear();
    m_quads.clear();
    m_edges.clear();
  }

  ~TQuadMesh()
  {
    clear();
  }

  void clear()
  {
    m_verts.clear();
    m_norms.clear();
    m_quads.clear();
    m_edges.clear();
  }

  void Set(const TQuadMesh &src)
  {
    clear();
    m_verts = src.m_verts;
    m_norms = src.m_norms;
    m_quads = src.m_quads;
    m_edges = src.m_edges;
  }

  TQuadMesh(const TQuadMesh& src)
  {
    Set(src);
  }

  TQuadMesh& operator=(const TQuadMesh& src)
  {
    Set(src);
    return *this;
  }


  // Vs : array of vertices
  // Qs : array of quad (vids) 左回りが表面
  void initialize(
    const std::vector<EVec3f> &Vs, 
    const std::vector<EVec4i> &Qs)
  {
    clear();
    m_verts = Vs;
    
    //(id, eid) --> id: 各頂点に対して作成したエッジの接続先頂点id, eid:その辺のid
    std::vector<std::map<int,int>> v_connected_vtx( m_verts.size() );

    for ( const auto &q : Qs )
    {
      EVec4i e;
      
      for ( int i = 0; i < 4; ++i )
      {
        int idx1 = q[i], idx2 = q[(i+1)%4];
        
        auto it = v_connected_vtx[ idx1 ].find( idx2 );
        if ( it == v_connected_vtx[ idx1 ].end() ) 
        {
          m_edges.push_back( TQuadEdge(idx1, idx2) );
          e[i] = (int)m_edges.size()-1;  
          v_connected_vtx[ idx1 ].insert(std::make_pair(idx2, e[i]));
          v_connected_vtx[ idx2 ].insert(std::make_pair(idx1, e[i]));
        }
        else
        {
          e[i] = it->second;
        }
        
        if ( idx1 < idx2 ) m_edges[e[i]].quad_l = (int)m_quads.size();
        else m_edges[e[i]].quad_r = (int)m_quads.size();
      }

      m_quads.push_back(TQuad(q,e));
    }
    
    UpdateNormal();


    //check
    std::cout << " information ------------------------------------- \n";
    std::cout << "verts:" << m_verts.size() << " quads: " << m_quads.size() << " edges: " << m_edges.size() << "\n";
    for ( auto e : m_edges) 
      std::cout << e.vi[0] << ", " << e.vi[1] << " -- " << e.quad_l << ", " << e.quad_r << "\n";

    
  }

  void Translate(const EVec3f t) { 
    for ( auto &v : m_verts) v += t; 
  }

  void Scale(const float  s) { 
    for ( auto &v : m_verts) v *= s; 
  }

  void Rotate(Eigen::AngleAxis<float> &R) { 
    for ( auto &v : m_verts) v = R * v; 
    for ( auto &v : m_norms) v = R * v; 
  }

  void Rotate(const EMat3f &R) { 
    for ( auto &v : m_verts) v = R * v; 
    UpdateNormal();
  }

  void MultMat(const EMat4f M)
  {
    EMat3f R;
    R << M(0, 0), M(0, 1), M(0, 2),
         M(1, 0), M(1, 1), M(1, 2),
         M(2, 0), M(2, 1), M(2, 2);
    EVec3f t(M(0, 3), M(1, 3), M(2, 3));
    for ( auto &v : m_verts) v = R * v + t;
    UpdateNormal();
  }

  EVec3f getGravityCenter() const
  {
    EVec3f p(0,0,0);
    for ( auto &v : m_verts) p += v;
    return p / (float)m_verts.size();
  }

  void getBoundBox(EVec3f &minV, EVec3f &maxV) const
  {
    minV << FLT_MAX, FLT_MAX, FLT_MAX;
    maxV << -FLT_MAX, -FLT_MAX, -FLT_MAX;

   for ( auto &v : m_verts)
    {
      minV[0] = std::min( minV[0], v[0]);
      minV[1] = std::min( minV[1], v[1]);
      minV[2] = std::min( minV[2], v[2]);
      maxV[0] = std::max( maxV[0], v[0]);
      maxV[1] = std::max( maxV[1], v[1]);
      maxV[2] = std::max( maxV[2], v[2]);
    }
  }

  void checkError() const
  {
    /*
    for (int i = 0; i < m_pSize; ++i)
    {
      int *idx = m_pPolys[i].idx;
      if( idx[0] < 0 || m_vSize <= idx[0] ) std::cout << "er1";
      if( idx[1] < 0 || m_vSize <= idx[1] ) std::cout << "er2";
      if( idx[2] < 0 || m_vSize <= idx[2] ) std::cout << "er3";
    }

    std::cout << "ch--";
    GLenum errcode=glGetError();
    if(errcode!=GL_NO_ERROR)
    {
      const GLubyte *errstring = gluErrorString(errcode);
      std::cout << "aaaaaa " << errcode << " " << errstring;
    }
    */
  }

  void UpdateNormal()
  {
    m_norms.resize( m_verts.size() );

#pragma omp parallel for
    for (int i = 0, s = (int)m_norms.size(); i < s ; ++i) 
      m_norms[i].setZero();

    for (int i = 0, s = (int)m_quads.size(); i < s ; ++i)
    { 
      int *idx = m_quads[i].vi;
      const EVec3f &x0 = m_verts[idx[0]], &x1 = m_verts[idx[1]];
      const EVec3f &x2 = m_verts[idx[2]], &x3 = m_verts[idx[3]];
  
      EVec3f n1 = ( x1 - x0).cross( x2 - x0);
      EVec3f n2 = ( x2 - x0).cross( x3 - x0);
      float l1 = n1.norm();  
      float l2 = n2.norm();
      if ( l1 > 0 ) n1 *= 1.f / l1;  
      if ( l2 > 0 ) n2 *= 1.f / l2;
      m_norms[idx[0]] += n1 + n2;  
      m_norms[idx[1]] += n1;  
      m_norms[idx[2]] += n1 + n2;  
      m_norms[idx[3]] += n2;  
    }

#pragma omp parallel for
    for (int i = 0, s = (int)m_norms.size(); i < s ; ++i) 
    {
      float len = m_norms[i].norm();
      if ( len > 0 ) m_norms[i] *= 1.0f / len;
    }

  }
  

  void draw() const
  {
    if (m_verts.size() == 0 || m_quads.size() == 0) return;
/*

    glBegin ( GL_TRIANGLES );
    for ( int i = 0, s = (int)m_quads.size(); i < s; ++i )
    {
      const int *idx = m_quads[i].vi;
      glNormal3fv( m_norms[idx[0]].data()); glVertex3fv( m_verts[idx[0]].data() );
      glNormal3fv( m_norms[idx[1]].data()); glVertex3fv( m_verts[idx[1]].data() );
      glNormal3fv( m_norms[idx[2]].data()); glVertex3fv( m_verts[idx[2]].data() );
      glNormal3fv( m_norms[idx[0]].data()); glVertex3fv( m_verts[idx[0]].data() );
      glNormal3fv( m_norms[idx[2]].data()); glVertex3fv( m_verts[idx[2]].data() );
      glNormal3fv( m_norms[idx[3]].data()); glVertex3fv( m_verts[idx[3]].data() );
    }

    glEnd();
*/

    unsigned int *indices = new unsigned int[ 3 * 2 * m_quads.size() ]; 
    float *verts = new float[ 3 * m_verts.size() ];
    float *norms = new float[ 3 * m_verts.size() ];
    
#pragma omp parallel for 
    for ( int i = 0, s = (int)m_verts.size(); i < s; ++i )
    {
      verts[3*i+0] = m_verts[i][0];
      verts[3*i+1] = m_verts[i][1];
      verts[3*i+2] = m_verts[i][2];
      norms[3*i+0] = m_norms[i][0];
      norms[3*i+1] = m_norms[i][1];
      norms[3*i+2] = m_norms[i][2];
    }

#pragma omp parallel for 
    for ( int i = 0, s = (int)m_quads.size(); i < s; ++i )
    {
      const int *idx = m_quads[i].vi;
      indices[6*i + 0] = idx[0];
      indices[6*i + 1] = idx[1];
      indices[6*i + 2] = idx[2];
      indices[6*i + 3] = idx[0];
      indices[6*i + 4] = idx[2];
      indices[6*i + 5] = idx[3];
    }


    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    //glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glVertexPointer(3, GL_FLOAT, 0, &m_verts.front() );
    glNormalPointer(   GL_FLOAT, 0, &m_norms.front() );
    glDrawElements(GL_TRIANGLES, 3 * 2 * (int)m_quads.size(), GL_UNSIGNED_INT, indices);

    delete[] norms;
    delete[] verts;
    delete[] indices;

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    //glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  }

  //only for surface where each vertex has unique texture coordinate
  void draw(const float *diff, const float *ambi, const float *spec, const float *shin) const
  {
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, spec);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diff);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambi);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shin);
    draw();
  }

  void DrawEdges() const
  {
    glBegin(GL_LINES);
    for ( const auto &e : m_edges) 
    {
      if ( e.quad_l == -1 || e.quad_r == -1 ) glColor3d(1,0,0); 
      else glColor3d(1,1,0);

      glVertex3fv( m_verts[e.vi[0]].data());
      glVertex3fv( m_verts[e.vi[1]].data());
    }
    glEnd();
  }

  void DrawEdges(int width, double r, double g, double b) const
  {
    glLineWidth((float)width);
    glColor3d(r, g, b);
    DrawEdges();
  }



/*
  void exportObjNoTexCd(const char *fname)
  {
    FILE* fp = fopen(fname, "w");
    if (!fp) return;

    fprintf(fp, "#Obj exported from tmesh\n");

    for (int i = 0; i < m_vSize; ++i)
    {
      fprintf(fp, "v %f %f %f\n", m_vVerts[i][0], m_vVerts[i][1], m_vVerts[i][2]);
    }

    for (int i = 0; i < m_pSize; ++i)
    {
      fprintf(fp, "f %d %d %d\n", m_pPolys[i].idx[0] + 1, m_pPolys[i].idx[1] + 1, m_pPolys[i].idx[2] + 1);
    }
    fclose(fp);
  }



  bool exportStl(const char *fname)
  {
    FILE* fp = fopen(fname, "w");
    if (!fp) return false;

    fprintf(fp, "solid tmesh\n");
    for (int i = 0; i < m_pSize; ++i)
    {
      fprintf(fp, "facet normal %f %f %f\n", m_pNorms[i][0], m_pNorms[i][1], m_pNorms[i][2]);
      fprintf(fp, "  outer loop\n");

      int *p = m_pPolys[i].idx;
      fprintf(fp, "    vertex %f %f %f\n", m_vVerts[p[0]][0], m_vVerts[p[0]][1], m_vVerts[p[0]][2]);
      fprintf(fp, "    vertex %f %f %f\n", m_vVerts[p[1]][0], m_vVerts[p[1]][1], m_vVerts[p[1]][2]);
      fprintf(fp, "    vertex %f %f %f\n", m_vVerts[p[2]][0], m_vVerts[p[2]][1], m_vVerts[p[2]][2]);

      fprintf(fp, "  endloop\n");
      fprintf(fp, "endfacet\n");
    }
    fprintf(fp, "endsolid tmesh\n");
    fclose(fp);

    return true;

  }

*/

  //bool pickByRay(const EVec3f &rayP, const EVec3f &rayD, EVec3f &pos, int &pid) const
  //{
  //  float depth = FLT_MAX;
  //  EVec3f tmpPos;
  //  pid = -1;

  //  for (int pi = 0; pi < m_pSize; ++pi)
  //  {
  //    const int *p = m_pPolys[pi].idx;
  //    if (t_intersectRayToTriangle(rayP, rayD, m_vVerts[p[0]], m_vVerts[p[1]], m_vVerts[p[2]], tmpPos))
  //    {
  //      float d = (tmpPos - rayP).norm();
  //      if (d < depth)
  //      {
  //        depth = d;
  //        pos = tmpPos;
  //        pid = pi;
  //      }
  //    }
  //  }
  //  return depth != FLT_MAX;
  //}



  //bool pickByRay(const EVec3f &rayP, const EVec3f &rayD, EVec3f &pos) const
  //{
  //  int pid;
  //  return pickByRay(rayP, rayD, pos, pid);
  //}


};







#pragma managed

