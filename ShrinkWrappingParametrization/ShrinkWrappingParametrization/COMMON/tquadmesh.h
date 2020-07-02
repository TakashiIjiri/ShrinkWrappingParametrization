#pragma once
#pragma warning(disable : 4996)

//stl
#include <iostream>
#include <list>
#include <vector>
#include <set>
#include <map>

#include "tmath.h"
#include "tmesh.h"
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
  EVec3f norm;

  TQuad(const EVec4i &_vi, const EVec4i &_ei) {
    for ( int i = 0; i < 4; ++i) vi[i] = _vi[i];
    for ( int i = 0; i < 4; ++i) ei[i] = _ei[i];
    norm << 0, 0, 0;
  }

  TQuad(const TQuad &p) { Set(p); }
  TQuad& operator= (const TQuad  &v) { Set(v); return *this; }

  void Set(const TQuad &p) {
    memcpy(vi, p.vi, sizeof(int) * 4);
    memcpy(ei, p.ei, sizeof(int) * 4);
    norm = p.norm;
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
  std::vector<TQuad > m_quads;
  std::vector<EVec3f> m_norms; 
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
    //for ( auto e : m_edges) 
    //  std::cout << e.vi[0] << ", " << e.vi[1] << " -- " << e.quad_l << ", " << e.quad_r << "\n";
  }

  void Translate(const EVec3f t) { 
    for ( auto &v : m_verts) v += t; 
  }

  void Scale(const float  s) { 
    for ( auto &v : m_verts) v *= s; 
  }

  void Rotate(Eigen::AngleAxis<float> &R) { 
    for ( auto &v : m_verts ) v = R * v; 
    for ( auto &q : m_quads ) q.norm = R * q.norm; 
    for ( auto &v : m_norms ) v = R * v; 
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
    
    const int num_v = (int)m_norms.size();    

#pragma omp parallel for
    for (int i = 0; i < num_v; ++i) 
      m_norms[i].setZero();

    for (auto &q : m_quads) 
    { 
      int *idx = q.vi;
      const EVec3f &x0 = m_verts[idx[0]], &x1 = m_verts[idx[1]];
      const EVec3f &x2 = m_verts[idx[2]], &x3 = m_verts[idx[3]];
  
      EVec3f n1 = ( x1 - x0).cross( x2 - x0);
      EVec3f n2 = ( x2 - x0).cross( x3 - x0);
      float l1 = n1.norm();  
      float l2 = n2.norm();

      if ( l1 > 0 ) n1 *= 1.f / l1;  
      if ( l2 > 0 ) n2 *= 1.f / l2;
      q.norm = (l1+l2 > 0) ? (n1 + n2).normalized() : EVec3f(0,0,0);

      m_norms[idx[0]] += n1 + n2;  
      m_norms[idx[1]] += n1;  
      m_norms[idx[2]] += n1 + n2;  
      m_norms[idx[3]] += n2;  
    }

#pragma omp parallel for
    for (int i = 0; i < num_v ; ++i) 
    {
      float len = m_norms[i].norm();
      if ( len > 0 ) m_norms[i] *= 1.0f / len;
    }
  }
  

  void draw() const
  {
    if (m_verts.size() == 0 || m_quads.size() == 0) return;

    unsigned int *indices = new unsigned int[ 3 * 2 * m_quads.size() ]; 
    const int num_q = (int)m_quads.size();

#pragma omp parallel for 
    for ( int i = 0; i < num_q; ++i )
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

    delete[] indices;

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    //glDisableClientState(GL_TEXTURE_COORD_ARRAY);
  }

  //only for surface where each vertex has unique texture coordinate
  void draw(const float *diff, const float *ambi, const float *spec, const float *shin) const
  {
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR , spec);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE  , diff);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT  , ambi);
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



  void DrawDebugInfo() const
  {
    glDisable(GL_LIGHTING);
    glBegin(GL_LINES );
    for ( auto e : m_edges){    
      EVec3f p = 0.5f * (m_verts[e.vi[0]] + m_verts[e.vi[1]]) ;
      EVec3f n = m_quads[e.quad_l].norm + m_quads[e.quad_r].norm;
      n.normalize();
    
      glVertex3fv(  p.data());
      EVec3f c = p+n;
      glVertex3fv( c.data());
    }
    glEnd();
  }


private:

  //q2 +--e1--+ q1
  //   |      |
  //   e2 qc  e0
  //   |      |
  //q3 +--e3--+ q0

  void SubdivisionIter()
  {
    //gen vertices
    const int num_v = (int)m_verts.size();
    const int num_e = (int)m_edges.size();
    const int num_q = (int)m_quads.size();
    
    std::vector<EVec3f> Vs = m_verts;

    for ( int i = 0; i < num_e; ++i)  
    {
      int* ei = m_edges[i].vi;
      EVec3f c = 0.5f * (m_verts[ei[0]] + m_verts[ei[1]]) ;
      Vs.push_back( c );
    } 

    for ( int i = 0; i < num_q; ++i)  
    {
      int* qi = m_quads[i].vi;
      EVec3f c = 0.25f * (m_verts[qi[0]] + m_verts[qi[1]] 
                        + m_verts[qi[2]] + m_verts[qi[3]]) ;
      Vs.push_back( c );
    }

    std::vector<EVec4i> Qs;
    for ( int i = 0; i < num_q; ++i)  
    {
      int* qi = m_quads[i].vi;
      int e0 = m_quads[i].ei[0] + num_v;
      int e1 = m_quads[i].ei[1] + num_v;
      int e2 = m_quads[i].ei[2] + num_v;
      int e3 = m_quads[i].ei[3] + num_v;
      int qc = i + num_e + num_v;
      Qs.push_back( EVec4i(qi[0], e0, qc, e3) );
      Qs.push_back( EVec4i(qi[1], e1, qc, e0) );
      Qs.push_back( EVec4i(qi[2], e2, qc, e1) );
      Qs.push_back( EVec4i(qi[3], e3, qc, e2) );
    }
    initialize(Vs, Qs);
  }


public:

  void Subdivision(int n = 1)
  {
    for ( int iter = 0; iter < n; ++iter)
      SubdivisionIter();
  }

private:

  //エッジの中点とそのエッジの法線を返す．法線が0ならfalseを返す
  inline bool GetEdgeMidPointNorm(const TQuadEdge &e, EVec3f &p, EVec3f &n){ 
    const EVec3f &x0 = m_verts[e.vi[0]];
    const EVec3f &x1 = m_verts[e.vi[1]];
    p[0] = 0.5f * (x0[0] + x1[0]);
    p[1] = 0.5f * (x0[1] + x1[1]);
    p[2] = 0.5f * (x0[2] + x1[2]);
    n[0] = m_quads[e.quad_l].norm[0] + m_quads[e.quad_r].norm[0];
    n[1] = m_quads[e.quad_l].norm[1] + m_quads[e.quad_r].norm[1];
    n[2] = m_quads[e.quad_l].norm[2] + m_quads[e.quad_r].norm[2];

    float len = n.norm();
    if ( len < 0.000001) 
      return false;
    n /= len;
    return true;
  }
  //四角形の中点とその四角形の法線を返す．法線が0ならfalseを返す
  inline bool GetQuadMidPointNorm(const TQuad &q, EVec3f &p, EVec3f &n){ 
    const EVec3f &x0 = m_verts[q.vi[0]];
    const EVec3f &x1 = m_verts[q.vi[1]];
    const EVec3f &x2 = m_verts[q.vi[2]];
    const EVec3f &x3 = m_verts[q.vi[3]];
    p[0] = 0.25f * (x0[0] + x1[0] + x2[0] + x3[0]);
    p[1] = 0.25f * (x0[1] + x1[1] + x2[1] + x3[1]);
    p[2] = 0.25f * (x0[2] + x1[2] + x2[2] + x3[2]);
    n = q.norm;
    if (n.norm() == 0 ) return false;
    return true; 
  }


  void ShringWrapping_ProjectEdgeVertex(
      const TQuadEdge &e, 
      const TMesh &trgtmesh, 
      EVec3f &pos, 
      float &offset)
  {
    EVec3f p, n;
    if ( !GetEdgeMidPointNorm(e, p, n) ) return;

    if ( trgtmesh.pickByRay(p, n, pos) ) {
      offset = (pos - p).norm();
      return;
    }

    if ( trgtmesh.pickByRay(p, -1 * n, pos)) { 
      offset = -(pos - p).norm();
      return;
    }
  }
  
  void ShringWrapping_ProjectQuadVertex(
      const TQuad &q, 
      const TMesh &trgtmesh, 
      EVec3f &pos, 
      float &offset)
  {
    EVec3f p, n;
    GetQuadMidPointNorm( q, p, n);
    if ( trgtmesh.pickByRay(p, n, pos) ) { 
      offset = (pos - p).norm();
      return;
    } 
    if ( trgtmesh.pickByRay(p,-n, pos)) { 
      offset = -(pos - p).norm();
      return;
    }
  }


public:
  std::vector<float> ShrinkWrappingSubdivision_SingleStep(const TMesh &trgtmesh, const std::vector<float> &_offsets)
  {
    std::cout << "ShrinkWrappingIter 11 ";

    const int num_v = (int)m_verts.size();
    const int num_e = (int)m_edges.size();
    const int num_q = (int)m_quads.size();
    
    //gen vertices
    std::vector<EVec3f> Vs = m_verts;
    std::vector<float> verts_offset = _offsets;
    Vs.resize(num_v + num_e + num_q);
    verts_offset.resize(num_v + num_e + num_q);
    
#pragma omp parallel for
    for ( int i = 0; i < num_e; ++i)  
    { 
      EVec3f pos;
      float offset;
      ShringWrapping_ProjectEdgeVertex( m_edges[i], trgtmesh, pos, offset);      
      Vs[num_v + i] = pos;
      verts_offset[num_v + i] = offset;
    } 
    std::cout << "-- 22 ";
    
#pragma omp parallel for
    for ( int i = 0; i < num_q; ++i)  
    {
      EVec3f pos;
      float offset;
      ShringWrapping_ProjectQuadVertex(m_quads[i], trgtmesh, pos, offset);
      Vs[num_v + num_e + i] = pos;
      verts_offset[num_v + num_e + i] = offset;
    }

    std::cout << "-- 33 ";


    std::vector<EVec4i> Qs ( 4 * num_q);
#pragma omp parallel for
    for ( int i = 0; i < num_q; ++i)  
    {
      int* qi = m_quads[i].vi;
      int e0 = m_quads[i].ei[0] + num_v;
      int e1 = m_quads[i].ei[1] + num_v;
      int e2 = m_quads[i].ei[2] + num_v;
      int e3 = m_quads[i].ei[3] + num_v;
      int qc = i + num_e + num_v;
      Qs[4*i+0] << qi[0], e0, qc, e3 ;
      Qs[4*i+1] << qi[1], e1, qc, e0 ;
      Qs[4*i+2] << qi[2], e2, qc, e1 ;
      Qs[4*i+3] << qi[3], e3, qc, e2 ;
    }

    initialize(Vs, Qs);
    std::cout << " -- 44\n";
    return verts_offset;
  }


public:

  // starting from cube shape (8 vertices)
  // perform subdivision and projection iteratively 
  // see Nobuyuki's paper 
  std::vector<float> ShrinkWrappingSubdivision(int n, const TMesh &trgtmesh)
  {
    std::vector<float> verts_offset(m_verts.size(), 0);
    if ( m_verts.size() != 8 || m_quads.size() != 6 ) return verts_offset;

    for ( int i=0; i < n; ++i )
       verts_offset = ShrinkWrappingSubdivision_SingleStep(trgtmesh, verts_offset);
    return verts_offset;
  }



  // starting from cube shape (8 vertices)
  // perform subdivision and projection iteratively 
  // use verts_offset when movind subdivision vertices in their normal direction 
  void Initialize_ShrinkWrappingSubdivision(
    const std::vector<EVec3f> &init_vs, 
    const std::vector<EVec4i> &init_qs,
    const int n,
    const std::vector<float>  &verts_offset )
  {
    initialize(init_vs, init_qs);
    
    for ( int iter = 0; iter < n; ++ iter)
    {
      const int num_v = (int)m_verts.size();
      const int num_e = (int)m_edges.size();
      const int num_q = (int)m_quads.size();
      //gen vertices
      std::vector<EVec3f> Vs = m_verts;
      Vs.resize(num_v + num_e + num_q);

#pragma omp parallel for
      for ( int i = 0; i < num_e; ++i)  
      { 
        EVec3f p, n;
        if ( GetEdgeMidPointNorm( m_edges[i], p, n) ) 
          Vs[num_v + i] = p + verts_offset[num_v + i] * n;
        else 
          Vs[num_v + i] = p;
      } 

#pragma omp parallel for
      for ( int i = 0; i < num_q; ++i)  
      {
        EVec3f p, n;
        if ( GetQuadMidPointNorm( m_quads[i], p, n) ) 
          Vs[num_v + num_e + i] = p + verts_offset[num_v + num_e + i] * n;
        else 
          Vs[num_v + num_e + i] = p + verts_offset[num_v + num_e + i] * n;
      }

      std::vector<EVec4i> Qs ( 4 * num_q);

#pragma omp parallel for
      for ( int i = 0; i < num_q; ++i)  
      {
        int* vi = m_quads[i].vi;
        int* ei = m_quads[i].ei;
        Qs[4*i+0] << vi[0], ei[0] + num_v, i + num_e + num_v, ei[3] + num_v ;
        Qs[4*i+1] << vi[1], ei[1] + num_v, i + num_e + num_v, ei[0] + num_v ;
        Qs[4*i+2] << vi[2], ei[2] + num_v, i + num_e + num_v, ei[1] + num_v ;
        Qs[4*i+3] << vi[3], ei[3] + num_v, i + num_e + num_v, ei[2] + num_v ;
      }
      initialize(Vs, Qs);
    }
  }

};







#pragma managed

