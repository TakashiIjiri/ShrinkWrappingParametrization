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

class TPoly
{
public:
  int idx[3];

  TPoly(int v0 = 0, int v1 = 0, int v2 = 0) {
    idx[0] = v0; idx[1] = v1; idx[2] = v2;
  }
  TPoly(const TPoly &p) { Set(p); }

  void Set(const TPoly &p) {
    memcpy(idx, p.idx, sizeof(int) * 3);
  }
  TPoly& operator= (const TPoly  &v) { Set(v); return *this; }
};






// very simple mesh representation 
// each vertex has 
// - position
// - TexCd
// - Normal 
// - one ring info 

class TMesh
{

public:
  //Vertex Info
  int          m_vSize;
  EVec3f      *m_vVerts;
  EVec3f      *m_vTexCd; //(u,v,w) 
  EVec3f      *m_vNorms;
  std::vector<int> *m_vRingPs;
  std::vector<int> *m_vRingVs;

  //Polygon Info
  int          m_pSize;
  EVec3f      *m_pNorms;
  TPoly       *m_pPolys;

  TMesh()
  {
    m_vSize = 0;
    m_vVerts = 0;
    m_vNorms = 0;
    m_vTexCd = 0;
    m_vRingPs = 0;
    m_vRingVs = 0;
    m_pSize = 0;
    m_pNorms = 0;
    m_pPolys = 0;
  }

  ~TMesh()
  {
    clear();
  }

  void clear()
  {
    if (m_vVerts != 0) delete[] m_vVerts;
    if (m_vNorms != 0) delete[] m_vNorms;
    if (m_vTexCd != 0) delete[] m_vTexCd;
    if (m_pNorms != 0) delete[] m_pNorms;
    if (m_pPolys != 0) delete[] m_pPolys;
    if (m_vRingPs != 0) delete[] m_vRingPs;
    if (m_vRingVs != 0) delete[] m_vRingVs;
    m_vSize = 0;
    m_vVerts = 0;
    m_vNorms = 0;
    m_vTexCd = 0;
    m_pSize = 0;
    m_pNorms = 0;
    m_pPolys = 0;
    m_vRingPs = 0;
    m_vRingVs = 0;
  }


  void Set(const TMesh &v)
  {
    clear();
    m_vSize = v.m_vSize;
    m_pSize = v.m_pSize;

    if (m_vSize != 0)
    {
      m_vVerts = new EVec3f[m_vSize];
      m_vNorms = new EVec3f[m_vSize];
      m_vTexCd = new EVec3f[m_vSize];
      memcpy(m_vVerts, v.m_vVerts, sizeof(EVec3f) * m_vSize);
      memcpy(m_vNorms, v.m_vNorms, sizeof(EVec3f) * m_vSize);
      memcpy(m_vTexCd, v.m_vTexCd, sizeof(EVec3f) * m_vSize);

      m_vRingVs = new std::vector<int>[m_vSize];
      m_vRingPs = new std::vector<int>[m_vSize];
      for (int i = 0; i < m_vSize; ++i)
      {
        m_vRingVs[i] = v.m_vRingVs[i];
        m_vRingPs[i] = v.m_vRingPs[i];
      }
    }


    if (m_pSize != 0)
    {
      m_pNorms = new EVec3f[m_pSize];
      m_pPolys = new TPoly[m_pSize];
      memcpy(m_pNorms, v.m_pNorms, sizeof(EVec3f) * m_pSize);
      memcpy(m_pPolys, v.m_pPolys, sizeof(TPoly) * m_pSize);
    }
  }

  TMesh(const TMesh& src)
  {
    m_vSize = 0;
    m_vVerts = 0;
    m_vNorms = 0;
    m_vTexCd = 0;
    m_vRingPs = 0;
    m_vRingVs = 0;
    m_pSize = 0;
    m_pNorms = 0;
    m_pPolys = 0;
    Set(src);
  }

  TMesh& operator=(const TMesh& src)
  {
    m_vSize = 0;
    m_vVerts = 0;
    m_vNorms = 0;
    m_vTexCd = 0;
    m_vRingPs = 0;
    m_vRingVs = 0;
    m_pSize = 0;
    m_pNorms = 0;
    m_pPolys = 0;
    Set(src);
    return *this;
  }



  bool initialize(const char *fName)
  {

    FILE* fp = fopen(fName, "r");
    if (!fp) return false;

    std::list<EVec3f>  vList;
    std::list<EVec2f>  uvList;
    std::list<TPoly >  pList;
    std::list<TPoly >  pUvList;

    char buf[512];
    while (fgets(buf, 255, fp))
    {
      char* bkup = _strdup(buf);
      char* token = strtok(buf, " \t");

      if (stricmp(token, "vt") == 0)
      {
        double u, v;
        sscanf(bkup, "vt %lf %lf", &u, &v);
        uvList.push_back(EVec2f((float)u, (float)v));
      }
      else if (stricmp(token, "v") == 0)
      {
        double x, y, z;
        sscanf(bkup, "v %lf %lf %lf", &x, &y, &z);
        vList.push_back(EVec3f((float)x, (float)y, (float)z));
      }
      else if (stricmp(token, "f") == 0)
      {
        //sscanfの返り値は正常に読めた数: / が入ったら2文字しか読めない
        int v[4], t[4], s;
        int vtxnum = sscanf(bkup, "f %d %d %d %d", &v[0], &v[1], &v[2], &v[3]);
        if (vtxnum < 3) vtxnum = sscanf(bkup, "f %d/%d %d/%d %d/%d %d/%d", &v[0], &t[0], &v[1], &t[1], &v[2], &t[2], &v[3], &t[3]) / 2;
        if (vtxnum < 3) vtxnum = sscanf(bkup, "f %d/%d/%d %d/%d/%d %d/%d/%d %d/%d/%d", &v[0], &t[0], &s, &v[1], &t[1], &s, &v[2], &t[2], &s, &v[3], &t[3], &s) / 3;
        if (vtxnum < 3) vtxnum = sscanf(bkup, "f %d//%d %d//%d %d//%d %d//%d", &v[0], &s, &v[1], &s, &v[2], &s, &v[3], &s) / 2;

        if ( vtxnum == 4 ){
          pList  .push_back(TPoly(v[0] - 1, v[1] - 1, v[2] - 1));
          pList  .push_back(TPoly(v[0] - 1, v[2] - 1, v[3] - 1));
          pUvList.push_back(TPoly(t[0] - 1, t[1] - 1, t[2] - 1));
          pUvList.push_back(TPoly(t[0] - 1, t[2] - 1, t[3] - 1));
        }
        else{
          pList.push_back(TPoly(v[0] - 1, v[1] - 1, v[2] - 1));
          pUvList.push_back(TPoly(t[0] - 1, t[1] - 1, t[2] - 1));
        }
      }
      free(bkup);
    }
    fclose(fp);


    std::vector<EVec3f> Vs { std::begin(vList  ), std::end(vList  ) };
    std::vector<EVec2f> Ts { std::begin(uvList ), std::end(uvList ) };
    std::vector<TPoly>  Ps { std::begin(pList  ), std::end(pList  ) };
    std::vector<TPoly>  Puv{ std::begin(pUvList), std::end(pUvList) };

    initialize(Vs, Ps);


    //1頂点につき1 texCdのときのみ，入力されたTexCdを利用
    if (Ts.size() == Vs.size() && Ps.size() == Puv.size() && isSame(Ps, Puv))
    {
      for (int i = 0; i < m_vSize; ++i) m_vTexCd[i] << Ts[i][0], Ts[i][1], 0;
    }

    std::cout << "loaded object file info :" << m_vSize << " " << m_pSize << "\n";
    return true;
  }



private:
  bool isSame(const std::vector<TPoly> &Ps, const std::vector<TPoly> &Puv)
  {
    if (Ps.size() != Puv.size()) return false;
    for (int i = 0; i < (int)Ps.size(); ++i)
    {
      if ( Ps[i].idx[0] != Puv[i].idx[0] ||
           Ps[i].idx[1] != Puv[i].idx[1] ||
           Ps[i].idx[2] != Puv[i].idx[2]) return false;
    }
    return true;
  }


public:
  void initialize(const std::vector<EVec3f> &Vs, const std::vector<TPoly> &Ps)
  {
    clear();

    m_vSize = (int)Vs.size();
    if (m_vSize != 0)
    {
      m_vVerts = new EVec3f[m_vSize];
      m_vNorms = new EVec3f[m_vSize];
      m_vTexCd = new EVec3f[m_vSize];
      m_vRingVs = new std::vector<int>[m_vSize];
      m_vRingPs = new std::vector<int>[m_vSize];
      for (int i = 0; i < m_vSize; ++i) m_vVerts[i] = Vs[i];
    }

    m_pSize = (int)Ps.size();
    if (m_pSize != 0)
    {
      m_pPolys = new TPoly[m_pSize];
      m_pNorms = new EVec3f[m_pSize];
      for (int i = 0; i < m_pSize; ++i) m_pPolys[i] = Ps[i];
    }

    updateNormal();
    updateRingInfo();


    //for debug
    std::cout << "for debug : check data\n";
    for (int i = 0; i < m_pSize; ++i)
    {
      int *idx = m_pPolys[i].idx;
      if (idx[0] < 0 || idx[0] >= m_vSize) std::cout << "aaaaaa";
      if (idx[1] < 0 || idx[1] >= m_vSize) std::cout << "bbbbbb";
      if (idx[2] < 0 || idx[2] >= m_vSize) std::cout << "cccccc";
    }
  }

  









  void smoothing(int n)
  {
    EVec3f *vs = new EVec3f[m_vSize];

    for (int k = 0; k < n; ++k)
    {
#pragma omp parallel for
      for (int i = 0; i < m_vSize; ++i)
      {
        vs[i] << 0, 0, 0;
        for (const auto &it : m_vRingVs[i]) vs[i] += m_vVerts[it];
        vs[i] /= (float)m_vRingVs[i].size();
      }
      std::swap(vs, m_vVerts);

    }
    delete[] vs;
    updateNormal();
  }



  void updateNormal()
  {
#pragma omp parallel for
    for (int i = 0; i < m_vSize; ++i) m_vNorms[i].setZero();

    for (int i = 0; i < m_pSize; ++i)
    {
      int *idx = m_pPolys[i].idx;

      // ※ゼロ割に気を付ける
      m_pNorms[i] = (m_vVerts[idx[1]] - m_vVerts[idx[0]]).cross(m_vVerts[idx[2]] - m_vVerts[idx[0]]);
      float l = m_pNorms[i].norm();
      if( l != 0 ) m_pNorms[i] /= l;

      m_vNorms[idx[0]] += m_pNorms[i];
      m_vNorms[idx[1]] += m_pNorms[i];
      m_vNorms[idx[2]] += m_pNorms[i];
    }

#pragma omp parallel for
    for (int i = 0; i < m_vSize; ++i) {
      float n = m_vNorms[i].norm(); 
      if ( n > 0)  m_vNorms[i] /= n;
    }
  }



  void updateRingInfo()
  {
    for (int i = 0; i < m_vSize; ++i) m_vRingPs[i].clear();
    for (int i = 0; i < m_vSize; ++i) m_vRingVs[i].clear();

    for (int i = 0; i < m_pSize; ++i)
    {
      int *idx = m_pPolys[i].idx;
      m_vRingPs[idx[0]].push_back(i);
      m_vRingPs[idx[1]].push_back(i);
      m_vRingPs[idx[2]].push_back(i);
      m_vRingVs[idx[0]].push_back(idx[1]);
      m_vRingVs[idx[0]].push_back(idx[2]);
      m_vRingVs[idx[1]].push_back(idx[0]);
      m_vRingVs[idx[1]].push_back(idx[2]);
      m_vRingVs[idx[2]].push_back(idx[0]);
      m_vRingVs[idx[2]].push_back(idx[1]);
    }

    for (int i = 0; i < m_vSize; ++i)
    {
      sort(m_vRingVs[i].begin(), m_vRingVs[i].end());
      auto it = unique(m_vRingVs[i].begin(), m_vRingVs[i].end());
      m_vRingVs[i].erase(it, m_vRingVs[i].end());
    }
  }

  void Translate(const EVec3f t) { 
    for (int i = 0; i < m_vSize; ++i) m_vVerts[i] += t; 
  }
  void Scale(const float  s) { 
    for (int i = 0; i < m_vSize; ++i) m_vVerts[i] *= s; 
  }
  void Rotate(Eigen::AngleAxis<float> &R) { 
    for (int i = 0; i < m_vSize; ++i) m_vVerts[i] = R * m_vVerts[i]; 
  }
  void Rotate(const EMat3f &R) { 
    for (int i = 0; i < m_vSize; ++i) m_vVerts[i] = R * m_vVerts[i]; 
  }

  void MultMat(const EMat4f M)
  {
    EMat3f R;
    R << M(0, 0), M(0, 1), M(0, 2),
         M(1, 0), M(1, 1), M(1, 2),
         M(2, 0), M(2, 1), M(2, 2);
    EVec3f t(M(0, 3), M(1, 3), M(2, 3));
    for (int i = 0; i < m_vSize; ++i) m_vVerts[i] = R * m_vVerts[i] + t;
    updateNormal();
  }




  EVec3f getGravityCenter() const
  {
    EVec3f p(0,0,0);
    for (int i = 0; i < m_vSize; ++i) p += m_vVerts[i];
    return p / (float)m_vSize;
  }



  void getBoundBox(EVec3f &minV, EVec3f &maxV) const
  {
    minV << FLT_MAX, FLT_MAX, FLT_MAX;
    maxV << -FLT_MAX, -FLT_MAX, -FLT_MAX;
    for (int i = 0; i < m_vSize; ++i)
    {
      minV[0] = std::min(minV[0], m_vVerts[i][0]);
      minV[1] = std::min(minV[1], m_vVerts[i][1]);
      minV[2] = std::min(minV[2], m_vVerts[i][2]);
      maxV[0] = std::max(maxV[0], m_vVerts[i][0]);
      maxV[1] = std::max(maxV[1], m_vVerts[i][1]);
      maxV[2] = std::max(maxV[2], m_vVerts[i][2]);
    }
  }



  void normalizeByUniformScaling()
  {
    EVec3f minV, maxV;
    getBoundBox(minV, maxV);
    EVec3f a = maxV - minV;
    float  s = std::max(a[0], std::max(a[1], a[2]));

    Translate(-minV);
    Scale(1.0f / s);
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

  void draw() const
  {
    if (m_vSize == 0 || m_pSize == 0) return;

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glNormalPointer(GL_FLOAT, 0, m_vNorms);
    glTexCoordPointer(3, GL_FLOAT, 0, m_vTexCd);
    glVertexPointer(3, GL_FLOAT, 0, m_vVerts);
    glDrawElements(GL_TRIANGLES, m_pSize * 3, GL_UNSIGNED_INT, m_pPolys);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
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
    for (int i = 0; i < m_pSize; ++i)
    {
      int* idx = m_pPolys[i].idx;
      glVertex3fv(m_vVerts[idx[0]].data());
      glVertex3fv(m_vVerts[idx[1]].data());
      glVertex3fv(m_vVerts[idx[1]].data());
      glVertex3fv(m_vVerts[idx[2]].data());
      glVertex3fv(m_vVerts[idx[2]].data());
      glVertex3fv(m_vVerts[idx[0]].data());
    }
    glEnd();
  }

  void DrawEdges(int width, double r, double g, double b) const
  {
    glLineWidth((float)width);
    glColor3d(r, g, b);
    DrawEdges();
  }

  //draw edge [i,j] if vert_flg[i] || vert_flg[j] 
  void DrawEdges(
      int width, 
      double r, double g, double b, 
      std::vector<int> &vert_flg 
  ) const
  {
    glLineWidth((float)width);
    glColor3d(r, g, b);

    glBegin(GL_LINES);
    for (int i = 0; i < m_vSize; ++i)
    {
      if ( !vert_flg[i] ) continue;
      for ( const auto &vi : m_vRingVs[i] )
      {
        glVertex3fv(m_vVerts[ i].data());
        glVertex3fv(m_vVerts[vi].data());
      }
    }
    glEnd();
  }







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



  void initializeIcosaHedron(const double r)
  {
    float a = (float)(r * 0.525731);
    float b = (float)(r * 0.850651);

    std::vector<EVec3f> Vs = {
      EVec3f(0, -a,  b), EVec3f(b, 0, a), EVec3f(b, 0,-a), EVec3f(-b, 0, -a), EVec3f(-b, 0, a), EVec3f(-a, b, 0),
      EVec3f(a, b, 0), EVec3f(a,-b, 0), EVec3f(-a,-b, 0), EVec3f(0,-a, -b), EVec3f(0, a, -b), EVec3f(0,  a, b) };
    std::vector<TPoly> Ps = {
      TPoly(1,  2,  6), TPoly(1,  7,  2), TPoly(3,  4,  5), TPoly(4,  3,  8),
      TPoly(6,  5, 11), TPoly(5,  6, 10), TPoly(9, 10,  2), TPoly(10, 9,  3),
      TPoly(7,  8,  9), TPoly(8,  7,  0), TPoly(11,  0,  1), TPoly(0,11,  4),
      TPoly(6,  2, 10), TPoly(1,  6, 11), TPoly(3,  5, 10), TPoly(5, 4, 11),
      TPoly(2,  7,  9), TPoly(7,  1,  0), TPoly(3,  9,  8), TPoly(4, 8,  0) };
    initialize(Vs, Ps);
  }


  // 1 - 2
  // |   |
  // 0 - 3
  void initializeSubdivPlane( EVec3f p[4], int n)
  {
    std::vector<EVec3f> Vs;
    std::vector<TPoly> Ps;

    for ( int x = 0; x < n; ++x) { 
      for ( int y = 0; y < n; ++y) {
        float tx = x / (n-1.0f);
        float ty = y / (n-1.0f);
        EVec3f p1 = tx * p[0] + (1-tx) * p[3];
        EVec3f p2 = tx * p[1] + (1-tx) * p[2];
        Vs.push_back( ty * p1 + (1-ty) * p2 );
     
        if ( x < n - 1 && y < n -1 ) { 
          int i = x + y * n; 
          Ps.push_back( TPoly(i, i+1, i+n+1) );
          Ps.push_back( TPoly(i, i+n+1, i+n) );
        }
      }
    }
    initialize(Vs,Ps);
  }


  void initializeSphere(const float r, const int reso_hori, const int reso_verti)
  {
    std::vector<EVec3f> vs;
    std::vector<TPoly>  ps;

    const float step_phi   = (float)(   M_PI / (reso_verti + 1.0) ); 
    const float step_theta = (float)(2* M_PI / (reso_hori       ) ); 

    vs.push_back( EVec3f(0,0,-1) ); //南極
    for ( int phi_i = 0; phi_i < reso_verti; ++phi_i )
    {
      //std::cout << "\n";
      for ( int theta_i = 0; theta_i < reso_hori; ++theta_i )
      {
        float phi   = step_phi * (phi_i + 1) - (float)(0.5 * M_PI);
        float theta = step_theta * theta_i;
        //std::cout << "(" <<phi << "," << theta << ")";

        vs.push_back( EVec3f( r * cos(theta ) * cos( phi ), 
                              r * sin(theta ) * cos( phi ),
                              r * sin(phi   ) ) );
      }
    }
    vs.push_back( EVec3f(0,0,1) ); //北極
    
    //vertex size == reso_hori * reso_verti + 2

    //蓋
    for ( int theta_i = 0; theta_i < reso_hori; ++theta_i )
    {
      ps.push_back(TPoly(0, 1+(theta_i + 1)%reso_hori, 1+theta_i));
    }
    
    for ( int phi_i = 0; phi_i < reso_verti - 1; ++phi_i )
    {
      for ( int theta_i = 0; theta_i < reso_hori; ++theta_i )
      {
        int i0 = 1+  phi_i    * reso_hori +  theta_i;
        int i1 = 1+  phi_i    * reso_hori + (theta_i + 1)%reso_hori;
        int i2 = 1+ (phi_i+1) * reso_hori + (theta_i + 1)%reso_hori;
        int i3 = 1+ (phi_i+1) * reso_hori +  theta_i;
        ps.push_back(TPoly(i0,i1,i2));
        ps.push_back(TPoly(i0,i2,i3));
      }
    }
    const int n = (int)vs.size()-1;
    for ( int theta_i = 0; theta_i < reso_hori; ++theta_i )
    {
      ps.push_back( TPoly( n, n-reso_hori+theta_i, n-reso_hori+(theta_i+1)%reso_hori) );
    }
    initialize(vs,ps);
  }




  bool pickByRay(const EVec3f &rayP, const EVec3f &rayD, EVec3f &pos, int &pid) const
  {
    float depth = FLT_MAX;
    EVec3f tmpPos;
    pid = -1;

    for (int pi = 0; pi < m_pSize; ++pi)
    {
      const int *p = m_pPolys[pi].idx;
      if (t_intersectRayToTriangle(rayP, rayD, m_vVerts[p[0]], m_vVerts[p[1]], m_vVerts[p[2]], tmpPos))
      {
        float d = (tmpPos - rayP).norm();
        if (d < depth)
        {
          depth = d;
          pos = tmpPos;
          pid = pi;
        }
      }
    }
    return depth != FLT_MAX;
  }



  bool pickByRay(const EVec3f &rayP, const EVec3f &rayD, EVec3f &pos) const
  {
    int pid;
    return pickByRay(rayP, rayD, pos, pid);
  }



  //static void DrawIcosaHedron(
  //  const float r, 
  //  const float *diff,
  //  const float *ambi,
  //  const float *spec,
  //  const float *shin)
  //{
  //  static TMesh m;
  //  if( m.m_vSize == 0 ) m.initializeIcosaHedron(1);

  //  glEnable( GL_NORMALIZE);
  //  glPushMatrix();
  //  glScalef(r,r,r);
  //  m.draw(diff, ambi, spec, shin);
  //  glPopMatrix();
  //  glDisable( GL_NORMALIZE);
  //}



  void TMesh::ExtractLinkedVertsPolys( 
    const int vid,
    std::vector<int> &vert_ids, 
    std::vector<int> &poly_ids) const
  {
    if( vid < 0 || m_vSize - 1 < vid ) return;

    //flg 0:not visited, 1:visited
    byte *flg = new byte[ m_vSize ]; 
    memset( flg, 0, sizeof( byte ) * m_vSize );

    TQueue< int > Q( m_vSize );
    Q.push_back( vid );
    flg[ vid ] = 1;

    while( !Q.empty() )
    {
      const std::vector<int> &ring = m_vRingVs[Q.front()];
      Q.pop_front();
      for ( auto it : ring )
      {
        if ( flg[it] == 0 ) 
        {
          flg[it] = 1;
          Q.push_back(it);
        }
      }
    }

    //verts
    vert_ids.clear();
    vert_ids.reserve(m_vSize);
    for ( int i = 0; i < m_vSize; ++i )
      if ( flg[i] ) 
        vert_ids.push_back( i );
      
    //polys 
    poly_ids.clear();
    poly_ids.reserve(m_pSize);
    for ( int i = 0; i < m_pSize; ++i )
      if ( flg[ m_pPolys[i].idx[0]] ) 
        poly_ids.push_back( i );

    delete[] flg;
  }


  void TMesh::ExtractRingNeighborVerts( 
    const int vid,
    const int ringsize, 
    std::set<int> &output_ids) const
  {
    if( vid < 0 || m_vSize - 1 < vid ) return;

	  output_ids.clear();
	  std::set<int> border_verts; 
	  border_verts.insert( vid );

	  for( int count = -1; count < ringsize; ++count)
	  {
		  std::set<int> new_border;
		  for( auto piv : border_verts)
		  {
			  output_ids.insert( piv );
			  for( auto ringidx : m_vRingVs[piv] )
			  {
				  if( output_ids  .find( ringidx ) != output_ids  .end() ) continue;
          if( border_verts.find( ringidx ) != border_verts.end() ) continue;
          new_border.insert( ringidx );
			  }
		  }
		  border_verts = new_border;
	  }
  }




  int GetNearestVertexIdx ( const EVec3f &p ) const
  {
    int idx;
    float distsq;
    t_verts_GetNearestPoint(m_vSize, m_vVerts, p, idx, distsq);
    return idx;
  }


  //最近傍点検索
  void GetDistToPoint(
      const EVec3f &p, 
      const int nearest_vid, 

      EVec3f &pos_onsurf, 
      double &dist) const
  {
	  dist = DBL_MAX;
    std::set<int> trgt_ps;
    for ( const auto& vid : m_vRingVs[nearest_vid] )
      for ( const auto& pid : m_vRingPs[vid] )
        trgt_ps.insert(pid);

    //edge
	  if ( trgt_ps.size() == 0)
	  { 
      std::cout << "strange input !!!!!!! " << nearest_vid << "\n";
		  Trace( m_vVerts[ nearest_vid] );
	  }

    //check on edges
	  for (const auto &it : m_vRingVs[nearest_vid])
	  {
		  const EVec3f &x0 = m_vVerts[ nearest_vid ];
		  const EVec3f &x1 = m_vVerts[ it  ];
      EVec3f tmppos;
      float  tmpdist = t_DistPointAndLineSegm_sq( p, x0, x1, tmppos);
      if ( tmpdist < dist ) {
        dist = tmpdist;
        pos_onsurf = tmppos;
      }
	  }
  
    //check on polyton 
	  for (auto it : trgt_ps)
	  {
		  // d1 = x1-x0, d2 = x2-x0, h = x0 + s d1 + t d2
		  // d1*(h-p) = 0
		  // d2*(h-p) = 0 よりstを計算
		  const EVec3f &x0 = m_vVerts[ m_pPolys[it].idx[0] ];
		  EVec3f d1 = m_vVerts[ m_pPolys[it].idx[1] ] - x0;
		  EVec3f d2 = m_vVerts[ m_pPolys[it].idx[2] ] - x0;
		  double s,t;
		  t_solve2by2LinearEquation( d1.dot(d1), d1.dot(d2), 
			                           d2.dot(d1), d2.dot(d2), d1.dot(p-x0),    
			                                                   d2.dot(p-x0), s,t);
		  if (0 <= s && 0 <= t && s + t <= 1)
		  {
			  EVec3f h = x0 + (float)s * d1 + (float)t * d2;
			  double d = t_dist_sq( p, h);
			  if( d < dist ) {
				  dist = d; 
				  pos_onsurf = h;
			  }
		  }
	  }

	  if( dist == DBL_MAX )
	  {
      std::cout << "never comes here error !!\n" ;
		  for( auto i : trgt_ps ) 
			  std::cout << i << "--" << i ;
	  }
  }


#define BUFFER_OFFSET(bytes) ((GLubyte *)NULL + (bytes))



  static void DrawSphere
  (
    const EVec3f &position, 
    const float radius
  )
  {
    static int polygon_size;
    static GLuint m_buffers[3] = {0,0,0};
    if ( m_buffers[0] == 0 )
    {
      const int M = 12;
      const int N = 12;
      TMesh s;
      s.initializeSphere( 1.0f, M, N);
      polygon_size = s.m_pSize;
    
      //generat vertex buffer object 
      float  *vs = new float[9*s.m_pSize]; // 3*pize * 3
      float  *ns = new float[9*s.m_pSize]; // 3*pize * 3
      GLuint *id = new GLuint[3*s.m_pSize];
      for( int p = 0; p < s.m_pSize; ++p)
      {
        const int *vidx = s.m_pPolys[p].idx;
        for ( int i=0; i < 3; ++i ) vs[9*p+i+0] = s.m_vVerts[vidx[0]][i];
        for ( int i=0; i < 3; ++i ) vs[9*p+i+3] = s.m_vVerts[vidx[1]][i];
        for ( int i=0; i < 3; ++i ) vs[9*p+i+6] = s.m_vVerts[vidx[2]][i];
        for ( int i=0; i < 3; ++i ) ns[9*p+i+0] = s.m_vNorms[vidx[0]][i];
        for ( int i=0; i < 3; ++i ) ns[9*p+i+3] = s.m_vNorms[vidx[1]][i];
        for ( int i=0; i < 3; ++i ) ns[9*p+i+6] = s.m_vNorms[vidx[2]][i];
        id[p*3+0] = p*3+0;
        id[p*3+1] = p*3+1;
        id[p*3+2] = p*3+2;
      }
      //init vbo
      glGenBuffers(3, m_buffers);
      //１つ目のバッファオブジェクトに頂点データ配列を転送する 
      glBindBuffer(GL_ARRAY_BUFFER, m_buffers[0]);
      glBufferData(GL_ARRAY_BUFFER, sizeof(float)*s.m_pSize*9, vs, GL_STATIC_DRAW);
      //２つ目のバッファオブジェクトに法線データ配列を転送する 
      glBindBuffer(GL_ARRAY_BUFFER, m_buffers[1]);
      glBufferData(GL_ARRAY_BUFFER, sizeof(float)*s.m_pSize*9, ns, GL_STATIC_DRAW);
      //3つ目のバッファオブジェクトに頂点のインデックスを転送する 
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffers[2]);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*s.m_pSize*3, id, GL_STATIC_DRAW);
      std::cout << radius << " " << polygon_size << " " << m_buffers[0] << "\n";
      delete[] vs;
      delete[] ns;
      delete[] id;
    }

    //rendering 
    glEnable(GL_NORMALIZE);
    glPushMatrix();
    glTranslated(position[0], position[1], position[2]);
    glScalef(radius, radius, radius);

    //頂点データ，法線データ，テクスチャ座標の配列を有効にする 
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
  
    //頂点/法線/indexデータの場所を指定する 
    glBindBuffer(GL_ARRAY_BUFFER, m_buffers[0]);
    glVertexPointer(3, GL_FLOAT, 0, BUFFER_OFFSET(0));
    glBindBuffer(GL_ARRAY_BUFFER, m_buffers[1]);
    glNormalPointer(GL_FLOAT, 0, BUFFER_OFFSET(0));
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_buffers[2]);
    glDrawElements(GL_TRIANGLES, polygon_size * 3, GL_UNSIGNED_INT, BUFFER_OFFSET(0));

    //頂点データ，法線データ
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);
 
    glPopMatrix();
    glDisable(GL_NORMALIZE);
  }

  static void DrawSphere
  (
    const EVec3f &position, 
    const float radius,
    const float *diff, 
    const float *ambi, 
    const float *spec, 
    const float *shin) 
  {
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR , spec);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE  , diff);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT  , ambi);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shin);
    DrawSphere(position, radius);
  }


};






//triangle soup class
// added 2018/3/21
class TTriangle 
{
public:
  EVec3f m_verts[3];

  TTriangle(){
    m_verts[0] = m_verts[1] = m_verts[2] = EVec3f(0, 0, 0);
  }

  TTriangle& operator= (const TTriangle  &v) { 
    Set(v); 
    return *this; 
  }

  TTriangle(const TTriangle &src){
    Set(src);
  }

  void Set(const TTriangle &src){
    m_verts[0] = src.m_verts[0]; 
    m_verts[1] = src.m_verts[1]; 
    m_verts[2] = src.m_verts[2];   
  }

};


class TTriangleSoup 
{
public:
  int        m_num_triangles;
  TTriangle *m_triangles   ;
  EVec3f    *m_normals  ;

  void Clear(){
    if( m_triangles != 0 ) delete[] m_triangles; 
    if( m_normals   != 0 ) delete[] m_normals  ; 
    m_triangles = 0;
    m_normals   = 0;
    m_num_triangles = 0;
  }

  void Set(const TTriangleSoup &src)
  {
    Clear();
    
    m_num_triangles = src.m_num_triangles;
    if( m_num_triangles == 0 ) return;

    m_triangles = new TTriangle[m_num_triangles];
    m_normals   = new EVec3f   [m_num_triangles];

    for( int i = 0 ; i < m_num_triangles ; ++i)
    {
      m_triangles[i] = src.m_triangles[i];
      m_normals[i] = src.m_normals[i];
    }
  }



  ~TTriangleSoup(){
    Clear();
  }

  TTriangleSoup(){
    m_num_triangles = 0;
    m_triangles = 0;
    m_normals = 0;
  }

  TTriangleSoup( const TTriangleSoup &src){
    m_num_triangles = 0;
    m_triangles = 0;
    m_normals = 0;
    Set( src );
  }

  TTriangleSoup& operator= (const TTriangleSoup  &v) { 
    Set(v); 
    return *this; 
  }

  void Allocate(int num_triangles, EVec3f *triangle_vertex_array = 0)
  {
    Clear();
    m_num_triangles = num_triangles;
    m_triangles = new TTriangle[m_num_triangles];
    m_normals   = new EVec3f   [m_num_triangles];

    if( triangle_vertex_array == 0) return;
    
    for(int i = 0; i <  m_num_triangles; ++i )
    {
      m_triangles[i].m_verts[0] = triangle_vertex_array[3*i+0];
      m_triangles[i].m_verts[1] = triangle_vertex_array[3*i+1];
      m_triangles[i].m_verts[2] = triangle_vertex_array[3*i+2];
    }

    UpdateNormal();
  }

  void UpdateNormal()
  {
#pragma omp parallel for
    for (int i = 0; i < m_num_triangles ; ++i) m_normals[i].setZero();

    for (int i = 0; i < m_num_triangles ; ++i)
    {
      TTriangle &t = m_triangles[i];

      m_normals[i] = (t.m_verts[1] - t.m_verts[0]).cross(t.m_verts[2] - t.m_verts[0]);
      float l = m_normals[i].norm();
      if( l != 0 ) m_normals[i] /= l;
    }
  }
  
  
  bool ExportAsStl(const char *fname)
  {
    FILE* fp = fopen(fname, "w");
    if (!fp) return false;

    fprintf(fp, "solid ttrianglesoup\n");
    for (int i = 0; i < m_num_triangles; ++i)
    {
      const TTriangle &t = m_triangles[i];
      fprintf(fp, "facet normal %f %f %f\n", m_normals[i][0], m_normals[i][1], m_normals[i][2]);
      fprintf(fp, "  outer loop\n");
      fprintf(fp, "    vertex %f %f %f\n", t.m_verts[0][0], t.m_verts[0][1], t.m_verts[0][2]);
      fprintf(fp, "    vertex %f %f %f\n", t.m_verts[1][0], t.m_verts[1][1], t.m_verts[1][2]);
      fprintf(fp, "    vertex %f %f %f\n", t.m_verts[2][0], t.m_verts[2][1], t.m_verts[2][2]);
      fprintf(fp, "  endloop\n");
      fprintf(fp, "endfacet\n");
    }
    fprintf(fp, "endsolid tmesh\n");
    fclose(fp);

    return true;
  }



  void Draw() const
  {
    if (m_num_triangles == 0 ) return;
    glBegin(GL_TRIANGLES);
    for( int i=0; i < m_num_triangles; ++i) 
    {
      const TTriangle& t = m_triangles[i];
      glNormal3fv( m_normals[i].data() );
      glVertex3fv( t.m_verts[0].data() );
      glVertex3fv( t.m_verts[1].data() );
      glVertex3fv( t.m_verts[2].data() );
    }
    glEnd();
  }



  //only for surface where each vertex has unique texture coordinate
  void Draw(const float *diff, const float *ambi, const float *spec, const float *shin) const
  {
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, spec);
    glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diff);
    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambi);
    glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shin);
    Draw();
  }

  void DrawEdges()
  {
    glBegin(GL_LINES);
    for (int i = 0; i < m_num_triangles; ++i)
    {
      const TTriangle& t = m_triangles[i];
      glVertex3fv( t.m_verts[0].data() ); glVertex3fv( t.m_verts[1].data() );
      glVertex3fv( t.m_verts[1].data() ); glVertex3fv( t.m_verts[2].data() );
      glVertex3fv( t.m_verts[2].data() ); glVertex3fv( t.m_verts[0].data() );
    }
    glEnd();
  }

  void DrawEdges(int width, double r, double g, double b)
  {
    glLineWidth((float)width);
    glColor3d(r, g, b);
    DrawEdges();
  }




  void ImportObj( const char* fname )
  {
    TMesh mesh;
    mesh.initialize(fname);
    
    Allocate(mesh.m_pSize);

    if( m_num_triangles == 0 ) return;

    for( int t = 0; t < mesh.m_pSize; ++t)
    {
      int* idx = mesh.m_pPolys[t].idx;
      m_triangles[t].m_verts[0] = mesh.m_vVerts[idx[0]];
      m_triangles[t].m_verts[1] = mesh.m_vVerts[idx[1]];
      m_triangles[t].m_verts[2] = mesh.m_vVerts[idx[2]];
    }
    UpdateNormal();
  }

  EVec3f GetGravityCenter()
  {
    EVec3f gc(0,0,0);

    for( int t = 0; t < m_num_triangles; ++t)
    {
      gc += m_triangles[t].m_verts[0];
      gc += m_triangles[t].m_verts[1];
      gc += m_triangles[t].m_verts[2];
    }
    return gc / (float)(3 * m_num_triangles);
  }
  
  void Translate(const EVec3f &trans )
  {
    for( int t = 0; t < m_num_triangles; ++t)
    {
      m_triangles[t].m_verts[0] += trans;
      m_triangles[t].m_verts[1] += trans;
      m_triangles[t].m_verts[2] += trans;
    }
  }

  void MultMat(const EMat4f M)
  {
    EMat3f R;
    R <<  M(0, 0), M(0, 1), M(0, 2),
          M(1, 0), M(1, 1), M(1, 2),
          M(2, 0), M(2, 1), M(2, 2);
    EVec3f t(M(0, 3), M(1, 3), M(2, 3));

    Trace(R);
    Trace(t);

    for (int i = 0; i < m_num_triangles; ++i) 
    {
      m_triangles[i].m_verts[0] = R * m_triangles[i].m_verts[0] + t;
      m_triangles[i].m_verts[1] = R * m_triangles[i].m_verts[1] + t;
      m_triangles[i].m_verts[2] = R * m_triangles[i].m_verts[2] + t;
    }
    UpdateNormal();

  }





};






///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
//Mesh filling ////////////////////////////////////////////////////////////////////////////


inline void calcBoundBox(
  const EVec3f &v0,
  const EVec3f &v1,
  const EVec3f &v2,
  EVec3f &bbMin,
  EVec3f &bbMax)
{
  bbMin << min3(v0[0], v1[0], v2[0]), min3(v0[1], v1[1], v2[1]), min3(v0[2], v1[2], v2[2]);
  bbMax << max3(v0[0], v1[0], v2[0]), max3(v0[1], v1[1], v2[1]), max3(v0[2], v1[2], v2[2]);
}


// 
// intersection h = v0 + s(v1-v0) + t(v2-v0) = (x,y,z)    // two of x,y,z are known , v0,v1,v2 are known 

inline bool intersectTriangleToRayX(
  const EVec3f &v0,
  const EVec3f &v1,
  const EVec3f &v2,
  const double y,
  const double z,

  double &x //output 
)
{
  //pre-check
  if ((y < v0[1] && y < v1[1] && y < v2[1]) || (y > v0[1] && y > v1[1] && y > v2[1])) return false;
  if ((z < v0[2] && z < v1[2] && z < v2[2]) || (z > v0[2] && z > v1[2] && z > v2[2])) return false;

  double s, t;
  if (!t_solve2by2LinearEquation(v1[1] - v0[1], v2[1] - v0[1],
    v1[2] - v0[2], v2[2] - v0[2], y - v0[1], z - v0[2], s, t)) return false;
  if (s < 0 || t < 0 || s + t > 1) return false;

  x = (1 - s - t)*v0[0] + s*v1[0] + t*v2[0];
  return true;
}
inline bool intersectTriangleToRayY(
  const EVec3f &v0,
  const EVec3f &v1,
  const EVec3f &v2,
  const double x,
  const double z,

  double &y //output 
)
{
  //pre-check
  if ((x < v0[0] && x < v1[0] && x < v2[0]) || (x > v0[0] && x > v1[0] && x > v2[0])) return false;
  if ((z < v0[2] && z < v1[2] && z < v2[2]) || (z > v0[2] && z > v1[2] && z > v2[2])) return false;

  double s, t;
  if (!t_solve2by2LinearEquation(v1[0] - v0[0], v2[0] - v0[0],
    v1[2] - v0[2], v2[2] - v0[2], x - v0[0], z - v0[2], s, t)) return false;
  if (s < 0 || t < 0 || s + t > 1) return false;

  y = (1 - s - t)*v0[1] + s*v1[1] + t*v2[1];
  return true;
}



// cast ray in X axis 
static void genBinaryVolumeInTriangleMeshX
(
  const int W,
  const int H,
  const int D,
  const double px,
  const double py,
  const double pz,

  const int vSize,
  const int pSize,
  const EVec3f *verts,
  const EVec3f *vNorm,
  const TPoly  *polys,
  const EVec3f *pNorm,

  byte *binVol //allocated[WxHxD], 0:out, 1:in
)
{
  //clock_t t0 = clock();
  const int WH = W*H, WHD = W*H*D;
  const EVec3f cuboid((float)(W*px), (float)(H*py), (float)(D*pz));

  EVec3f BBmin, BBmax;
  t_calcBoundBox3D(vSize, verts, BBmin, BBmax);

  memset(binVol, 0, sizeof(byte) * WHD);


  // insert triangles in BINs -- divide yz space into (BIN_SIZE x BIN_SIZE)	
  const int BIN_SIZE = 100;
  std::vector< std::vector<int> > polyID_Bins(BIN_SIZE * BIN_SIZE, std::vector<int>());

  for (int p = 0; p < pSize; ++p)
  {
    EVec3f bbMin, bbMax;
    calcBoundBox(verts[polys[p].idx[0]], verts[polys[p].idx[1]], verts[polys[p].idx[2]], bbMin, bbMax);
    int yS = std::min((int)(bbMin[1] / cuboid[1] * BIN_SIZE), BIN_SIZE - 1);
    int zS = std::min((int)(bbMin[2] / cuboid[2] * BIN_SIZE), BIN_SIZE - 1);
    int yE = std::min((int)(bbMax[1] / cuboid[1] * BIN_SIZE), BIN_SIZE - 1);
    int zE = std::min((int)(bbMax[2] / cuboid[2] * BIN_SIZE), BIN_SIZE - 1);
    for (int z = zS; z <= zE; ++z) for (int y = yS; y <= yE; ++y) polyID_Bins[z*BIN_SIZE + y].push_back(p);
  }

  //clock_t t1 = clock();

  // ray casting along x axis to fill inside the mesh 
#pragma omp parallel for
  for (int zI = 0; zI < D; ++zI) if (BBmin[2] <= (0.5 + zI) * pz && (0.5 + zI) * pz <= BBmax[2])
    for (int yI = 0; yI < W; ++yI) if (BBmin[1] <= (0.5 + yI) * py && (0.5 + yI) * px <= BBmax[1])
    {
      double y = (0.5 + yI) * px;
      double z = (0.5 + zI) * pz;
      int bin_yi = std::min((int)(y / cuboid[1] * BIN_SIZE), BIN_SIZE - 1);
      int bin_zi = std::min((int)(z / cuboid[2] * BIN_SIZE), BIN_SIZE - 1);
      std::vector<int> &trgtBin = polyID_Bins[bin_zi * BIN_SIZE + bin_yi];

      std::multimap<double, double> blist;// (xPos, normInXdir);

      for (const auto pi : trgtBin) if (pNorm[pi][1] != 0)
      {
        const EVec3f &V0 = verts[polys[pi].idx[0]];
        const EVec3f &V1 = verts[polys[pi].idx[1]];
        const EVec3f &V2 = verts[polys[pi].idx[2]];
        double x;
        if (intersectTriangleToRayX(V0, V1, V2, y, z, x)) blist.insert(std::make_pair(x, pNorm[pi][0])); //(x 座標, normal[0])
      }

      //clean blist (edge上で起こった交差重複を削除)
      while (blist.size() != 0)
      {
        if (blist.size() == 1) { blist.clear(); break; }

        bool found = false;
        auto it0 = blist.begin();
        auto it1 = blist.begin(); it1++;

        for (; it1 != blist.end(); ++it0, ++it1) if (it0->second * it1->second > 0)
        {
          blist.erase(it1);
          found = true;
          break;
        }
        if (!found) break;
      }

      bool flag = false;
      int xI = 0;

      //int pivIdx = xI ;
      for (auto it = blist.begin(); it != blist.end(); ++it)
      {
        int pivXi = (int)(it->first / py);
        for (; xI <= pivXi && xI < W; ++xI) binVol[xI + yI * W + zI*WH] = flag;
        flag = !flag;
      }
      if (flag == true) std::cout << "error double check here!\n";
    }

  //clock_t t2 = clock();
  //std::cout << "compute time : " << (t1-t0)/ (double) CLOCKS_PER_SEC << " " << (t2-t1)/ (double) CLOCKS_PER_SEC) << "\n";
}
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////




// cast ray in Y axis ( divide ZX plane )  
inline void genBinaryVolumeInTriangleMeshY
(
  const EVec3i &reso,
  const EVec3f &pitch,
  const TMesh  &mesh,

  byte *binVol //allocated[WxHxD], 0:out, 1:in
)
{
  const int    W = reso[0];
  const int    H = reso[1];
  const int    D = reso[2];
  const double px = pitch[0];
  const double py = pitch[1];
  const double pz = pitch[2];
  const int     vSize = mesh.m_vSize;
  const int     pSize = mesh.m_pSize;
  const EVec3f *verts = mesh.m_vVerts;
  const EVec3f *vNorm = mesh.m_vNorms;
  const TPoly  *polys = mesh.m_pPolys;
  const EVec3f *pNorm = mesh.m_pNorms;


  //clock_t t0 = clock();
  const int WH = W*H, WHD = W*H*D;
  const EVec3f cuboid((float)(W*px), (float)(H*py), (float)(D*pz));

  EVec3f BBmin, BBmax;
  t_calcBoundBox3D(vSize, verts, BBmin, BBmax);

  memset(binVol, 0, sizeof(byte) * WHD);


  // insert triangles in BINs -- divide yz space into (BIN_SIZE x BIN_SIZE)	
  const int BIN_SIZE = 20;
  std::vector< std::vector<int> > polyID_Bins(BIN_SIZE * BIN_SIZE, std::vector<int>());

  for (int p = 0; p < pSize; ++p)
  {
    EVec3f bbMin, bbMax;
    calcBoundBox(verts[polys[p].idx[0]], verts[polys[p].idx[1]], verts[polys[p].idx[2]], bbMin, bbMax);
    int xS = std::min((int)(bbMin[0] / cuboid[0] * BIN_SIZE), BIN_SIZE - 1);
    int zS = std::min((int)(bbMin[2] / cuboid[2] * BIN_SIZE), BIN_SIZE - 1);
    int xE = std::min((int)(bbMax[0] / cuboid[0] * BIN_SIZE), BIN_SIZE - 1);
    int zE = std::min((int)(bbMax[2] / cuboid[2] * BIN_SIZE), BIN_SIZE - 1);
    for (int z = zS; z <= zE; ++z) for (int x = xS; x <= xE; ++x) polyID_Bins[z*BIN_SIZE + x].push_back(p);
  }

  //clock_t t1 = clock();

  // ray casting along x axis to fill inside the mesh 
#pragma omp parallel for
  for (int zI = 0; zI < D; ++zI) if (BBmin[2] <= (0.5 + zI) * pz && (0.5 + zI) * pz <= BBmax[2])
    for (int xI = 0; xI < W; ++xI) if (BBmin[0] <= (0.5 + xI) * px && (0.5 + xI) * px <= BBmax[0])
    {
      double x = (0.5 + xI) * px;
      double z = (0.5 + zI) * pz;
      int bin_xi = std::min((int)(x / cuboid[0] * BIN_SIZE), BIN_SIZE - 1);
      int bin_zi = std::min((int)(z / cuboid[2] * BIN_SIZE), BIN_SIZE - 1);
      std::vector<int> &trgtBin = polyID_Bins[bin_zi * BIN_SIZE + bin_xi];

      std::multimap<double, double> blist;// (xPos, normInXdir);

      for (const auto pi : trgtBin) if (pNorm[pi][1] != 0)
      {
        const EVec3f &V0 = verts[polys[pi].idx[0]];
        const EVec3f &V1 = verts[polys[pi].idx[1]];
        const EVec3f &V2 = verts[polys[pi].idx[2]];
        double y;
        if (intersectTriangleToRayY(V0, V1, V2, x, z, y))
          blist.insert( std::make_pair(y, pNorm[pi][1])); //(y 座標, normal)
      }

      if (blist.size() == 0) continue;

      //clean blist (edge上で起こった交差重複を削除)
      while (blist.size() != 0)
      {
        if (blist.size() == 1) { blist.clear(); break; }

        bool found = false;
        auto it0 = blist.begin();
        auto it1 = blist.begin(); it1++;

        for (; it1 != blist.end(); ++it0, ++it1) if (it0->second * it1->second > 0)
        {
          blist.erase(it1);
          found = true;
          break;
        }
        if (!found) break;
      }

      bool flag = false;
      int yI = 0;

      //int pivIdx = xI ;
      for (auto it = blist.begin(); it != blist.end(); ++it)
      {
        int pivYi = (int)(it->first / py);
        for (; yI <= pivYi && yI < H; ++yI) binVol[xI + yI * W + zI*WH] = flag;
        flag = !flag;
      }
      if (flag == true) std::cout << "error double check here!";
    }

  //clock_t t2 = clock();
  //std::cout << "compute time : " << (t1-t0)/ (double) CLOCKS_PER_SEC << " " << (t2-t1)/ (double) CLOCKS_PER_SEC) << "\n";
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////


static void t_calcBoundBox3D(const int n, const TTriangle* tris, EVec3f &BBmin, EVec3f &BBmax)
{
  BBmin <<  FLT_MAX,  FLT_MAX,  FLT_MAX;
  BBmax << -FLT_MAX, -FLT_MAX, -FLT_MAX;

  for (int i = 0; i < n; ++i)
  {
    const TTriangle& t = tris[i];
    for( int k=0; k < 3; ++k)
    {
      BBmin[0] = std::min(BBmin[0], t.m_verts[k][0]);
      BBmin[1] = std::min(BBmin[1], t.m_verts[k][1]);
      BBmin[2] = std::min(BBmin[2], t.m_verts[k][2]);
      BBmax[0] = std::max(BBmax[0], t.m_verts[k][0]);
      BBmax[1] = std::max(BBmax[1], t.m_verts[k][1]);
      BBmax[2] = std::max(BBmax[2], t.m_verts[k][2]);
    }
  }
}


// cast ray in Y axis ( divide ZX plane )  
inline void genBinaryVolumeInTriangleMeshY
(
  const EVec3i &reso,
  const EVec3f &pitch,
  const TTriangleSoup  &mesh,

  byte *binVol //allocated[WxHxD], 0:out, 1:in
)
{
  const int    W  = reso[0];
  const int    H  = reso[1];
  const int    D  = reso[2];
  const double px = pitch[0];
  const double py = pitch[1];
  const double pz = pitch[2];
  const int num_tris = mesh.m_num_triangles;
  const TTriangle *tris  = mesh.m_triangles;
  const EVec3f    *norms = mesh.m_normals;


  //clock_t t0 = clock();
  const int WH = W*H, WHD = W*H*D;
  const EVec3f cuboid((float)(W*px), (float)(H*py), (float)(D*pz));

  EVec3f BBmin, BBmax;
  t_calcBoundBox3D(num_tris, tris, BBmin, BBmax);

  memset(binVol, 0, sizeof(byte) * WHD);


  // insert triangles in BINs -- divide yz space into (BIN_SIZE x BIN_SIZE)	
  const int BIN_SIZE = 100;
  std::vector< std::vector<int> > polyID_Bins(BIN_SIZE * BIN_SIZE, std::vector<int>());

  for (int p = 0; p < num_tris; ++p)
  {
    const TTriangle &t = tris[p];

    EVec3f bbMin, bbMax;
    calcBoundBox( t.m_verts[0], t.m_verts[1], t.m_verts[2], bbMin, bbMax);
    int xS = t_crop<int>(0, BIN_SIZE - 1, (int)(bbMin[0] / cuboid[0] * BIN_SIZE));
    int zS = t_crop<int>(0, BIN_SIZE - 1, (int)(bbMin[2] / cuboid[2] * BIN_SIZE));
    int xE = t_crop<int>(0, BIN_SIZE - 1, (int)(bbMax[0] / cuboid[0] * BIN_SIZE));
    int zE = t_crop<int>(0, BIN_SIZE - 1, (int)(bbMax[2] / cuboid[2] * BIN_SIZE));

    for (int z = zS; z <= zE; ++z) for (int x = xS; x <= xE; ++x) polyID_Bins[z*BIN_SIZE + x].push_back(p);
  }

  //clock_t t1 = clock();

  // ray casting along x axis to fill inside the mesh 
#pragma omp parallel for
  for (int zI = 0; zI < D; ++zI) if (BBmin[2] <= (0.5 + zI) * pz && (0.5 + zI) * pz <= BBmax[2])
  {
    for (int xI = 0; xI < W; ++xI) if (BBmin[0] <= (0.5 + xI) * px && (0.5 + xI) * px <= BBmax[0])
    {
      double x = (0.5 + xI) * px;
      double z = (0.5 + zI) * pz;
      int bin_xi = t_crop<int>( 0, BIN_SIZE-1, (int)(x / cuboid[0] * BIN_SIZE) );
      int bin_zi = t_crop<int>( 0, BIN_SIZE-1, (int)(z / cuboid[2] * BIN_SIZE) );
      std::vector<int> &trgtBin = polyID_Bins[bin_zi * BIN_SIZE + bin_xi];

      std::multimap<double, double> blist;// (xPos, normInXdir);

      for (const auto pi : trgtBin) if ( norms[pi][1] != 0)
      {
        const TTriangle &t = tris[pi];
        double y;
        if (intersectTriangleToRayY(t.m_verts[0], t.m_verts[1], t.m_verts[2], x, z, y))
          blist.insert( std::make_pair(y, norms[pi][1] )); //(y 座標, normal)
      }

      if (blist.size() == 0) continue;

      //clean blist (edge上で起こった交差重複を削除)
      while (blist.size() != 0)
      {
        if (blist.size() == 1) { blist.clear(); break; }

        bool found = false;
        auto it0 = blist.begin();
        auto it1 = blist.begin(); it1++;

        for (; it1 != blist.end(); ++it0, ++it1) if (it0->second * it1->second > 0)
        {
          blist.erase(it1);
          found = true;
          break;
        }
        if (!found) break;
      }

      bool flag = false;
      int yI = 0;

      //int pivIdx = xI ;
      for (auto it = blist.begin(); it != blist.end(); ++it)
      {
        int pivYi = (int)(it->first / py);
        for (; yI <= pivYi && yI < H; ++yI) binVol[xI + yI * W + zI*WH] = flag;
        flag = !flag;
      }
      if (flag == true) std::cout << "error double check here!";
    }
  }
  //clock_t t2 = clock();
  //std::cout << "compute time : " << (t1-t0)/ (double) CLOCKS_PER_SEC << " " << (t2-t1)/ (double) CLOCKS_PER_SEC) << "\n";
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////



// draw translate / rotate handle 
inline void t_DrawCylinder(double len, double r)
{
  //上面
  glBegin(GL_TRIANGLE_FAN);
  glNormal3d(0, 1, 0);
  for (int i = 0; i < 6; i++)
  {
    double a = i * M_PI / 3;
    glVertex3d(r * sin(a), len, r * cos(a));
  }
  glEnd();
  //側面
  glBegin(GL_TRIANGLE_STRIP);
  for (int i = 0; i <= 6; i++) {
    double a = -i * M_PI / 3;
    glNormal3d(sin(a), 0, cos(a));
    glVertex3d(r * sin(a), 0, r * cos(a));
    glVertex3d(r * sin(a), len, r * cos(a));
  }
  glEnd();
  //底面
  glBegin(GL_TRIANGLE_FAN);
  glNormal3d(0, -1, 0);
  for (int i = 0; i < 6; i++) {
    double a = -i * M_PI / 3;
    glVertex3d(r * sin(a), 0, r * cos(a));
  }
  glEnd();
}

inline void t_DrawCone(double len, double r) {
  //座標変換

  glBegin(GL_TRIANGLE_FAN);
  glNormal3d(0, 1, 0);
  glVertex3d(0, len, 0);
  for (int i = 0; i <= 6; i++) {
    double a = i * M_PI / 3;
    glNormal3d(sin(a), 0, cos(a));
    glVertex3d(r * sin(a), 0, r * cos(a));
  }
  glEnd();

  glBegin(GL_TRIANGLE_FAN);
  glNormal3d(0, -1, 0);
  for (int i = 0; i < 6; i++) {
    double a = -i * M_PI / 3;
    glVertex3d(r * sin(a), 0, r * cos(a));
  }
  glEnd();
}


inline void t_DrawCube( double size ) 
{
  //座標変換
  glBegin( GL_TRIANGLES );
  
  //
  glNormal3d(0, 0, -1);
  glVertex3d(-size, -size, -size); glVertex3d( size,  size, -size); glVertex3d( size, -size, -size);
  glVertex3d(-size, -size, -size); glVertex3d(-size,  size, -size); glVertex3d( size,  size, -size);

  glNormal3d(0, 0, 1);
  glVertex3d(-size, -size,  size); glVertex3d( size, -size,  size); glVertex3d( size,  size,  size); 
  glVertex3d(-size, -size,  size); glVertex3d( size,  size,  size); glVertex3d(-size,  size,  size); 

  glNormal3d(0, -1, 0);
  glVertex3d(-size, -size, -size); glVertex3d( size, -size, -size); glVertex3d( size, -size,  size);
  glVertex3d(-size, -size, -size); glVertex3d( size, -size,  size); glVertex3d(-size, -size,  size);

  glNormal3d(0,  1, 0);
  glVertex3d(-size,  size, -size); glVertex3d( size,  size,  size); glVertex3d( size,  size, -size); 
  glVertex3d(-size,  size, -size); glVertex3d(-size,  size,  size); glVertex3d( size,  size,  size); 

  glNormal3d(-1, 0, 0);
  glVertex3d(-size, -size, -size); glVertex3d(-size,  size,  size); glVertex3d(-size,  size, -size);
  glVertex3d(-size, -size, -size); glVertex3d(-size, -size,  size); glVertex3d(-size,  size,  size);

  glNormal3d( 1, 0, 0);
  glVertex3d( size, -size, -size); glVertex3d( size,  size, -size); glVertex3d( size,  size,  size); 
  glVertex3d( size, -size, -size); glVertex3d( size,  size,  size); glVertex3d( size, -size,  size); 

  glEnd();
}



enum ORTHO_HANDLE_ID{
  OHDL_NON, 
  OHDL_X, 
  OHDL_Y, 
  OHDL_Z,
  OHDL_XY, 
  OHDL_YZ, 
  OHDL_ZX
};



//handle for translation
inline void t_DrawHandleOrthoArrows(
  const EVec3f &gc, 
  const double length,
  const double radius, 
  const float* color_x,
  const float* color_y,
  const float* color_z )
{
  const double cylinder_length = length * 0.8;
  const double cylinder_radius = radius;
  const double cone_length     = length * 0.2;
  const double cone_radius     = radius * 1.5;
  const float white[4] = {1,1,1,1};
  const float shin[1] = {64};
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shin);

  glPushMatrix();
  glTranslated((double)gc[0], (double)gc[1], (double)gc[2]);
  glEnable(GL_LIGHTING);

  //y
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color_y);
  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color_y);

  glTranslated(0, -cylinder_length, 0);
  t_DrawCylinder( 2*cylinder_length, cylinder_radius);
  glTranslated(0, 2*cylinder_length, 0);
  t_DrawCone( cone_length, cone_radius);
  glTranslated(0, -cylinder_length, 0);

  //x
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color_x);
  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color_x);
  glRotated(-90, 0, 0, 1);
  glTranslated(0, -cylinder_length, 0);
  t_DrawCylinder( 2*cylinder_length, cylinder_radius);
  glTranslated(0, 2*cylinder_length, 0);
  t_DrawCone( cone_length, cone_radius);
  glTranslated(0, -cylinder_length, 0);

  //z
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color_z);
  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color_z);
  glRotated(90, 1, 0, 0);
  glTranslated(0, -cylinder_length, 0);
  t_DrawCylinder( 2*cylinder_length, cylinder_radius);
  glTranslated(0, 2*cylinder_length, 0);
  t_DrawCone( cone_length, cone_radius);
  glTranslated(0, -cylinder_length, 0);

  glPopMatrix();
}




//handle for rotation
inline void t_DrawHandleOrthoCircles
(
  const EVec3f &gc, 
  const double radius
)
{
  glDisable(GL_LIGHTING);
  glLineWidth(5);

  glPushMatrix();
  glTranslated((double)gc[0], (double)gc[1], (double)gc[2]);
  
  const int RES = 50;
  const double step = 2 * M_PI / (RES-1);
  const double r = radius;

  glBegin(GL_LINE_STRIP);
  glColor3d(1, 0, 0);
  for (int i = 0; i < RES; ++i) 
    glVertex3d( r*cos(step * i), r* sin(step * i), 0);
  glEnd();

  glBegin(GL_LINE_STRIP);
  glColor3d(0, 1, 0);
  for (int i = 0; i < RES; ++i) 
    glVertex3d( 0, r*cos(step * i), r* sin(step * i));
  glEnd();
  
  glBegin(GL_LINE_STRIP);
  glColor3d(0, 0, 1);
  for (int i = 0; i < RES; ++i) 
    glVertex3d( r* sin(step * i), 0, r*cos(step * i));
  glEnd();

  glPopMatrix();
}




//for scaling handle
inline void t_DrawHandleOrthoCubes(
  const EVec3f &gc, 
  const double length,
  const double radius, 
  const float* color_x,
  const float* color_y,
  const float* color_z
)
{
  const float white[4] = {1,1,1,1};
  const float shin[1] = {64};
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, white);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, shin);

  glPushMatrix();
  glTranslated((double)gc[0], (double)gc[1], (double)gc[2]);

  glEnable(GL_LIGHTING);

  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color_y);
  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color_y);
  glTranslated(0, -length, 0);
  t_DrawCylinder(2*length, radius);
  glTranslated(0, length, 0);
  t_DrawCube( 1.5 * radius );
  glTranslated(0, -length, 0);

  //x
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color_x);
  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color_x);
  glRotated(-90, 0, 0, 1);
  glTranslated(0, -length, 0);
  t_DrawCylinder(2*length, radius);
  glTranslated(0, length, 0);
  t_DrawCube( 1.5 * radius );
  glTranslated(0, -length, 0);


  //z
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, color_z);
  glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, color_z);
  glRotated(90, 1, 0, 0);
  glTranslated(0, -length, 0);
  t_DrawCylinder(2*length, radius);
  glTranslated(0, length, 0);
  t_DrawCube( 1.5 * radius );
  glTranslated(0, -length, 0);

  glPopMatrix();
}





inline ORTHO_HANDLE_ID t_PickHandleOrthoArrows
(
    const EVec3f &ray_pos,
    const EVec3f &ray_dir,
    const EVec3f &center,
    const float   length,
    const float   radius
)
{
  EVec3f line_x0(center[0] - length, center[1], center[2]);
  EVec3f line_x1(center[0] + length, center[1], center[2]);
  EVec3f line_y0(center[0], center[1] - length, center[2]);
  EVec3f line_y1(center[0], center[1] + length, center[2]);
  EVec3f line_z0(center[0], center[1], center[2] - length);
  EVec3f line_z1(center[0], center[1], center[2] + length);

  float dx,dy,dz;
  EVec3f px,py,pz;
  std::tie(dx,px) = t_DistRayAndLineSegm(ray_pos, ray_dir, line_x0, line_x1 );
  std::tie(dy,py) = t_DistRayAndLineSegm(ray_pos, ray_dir, line_y0, line_y1 );
  std::tie(dz,pz) = t_DistRayAndLineSegm(ray_pos, ray_dir, line_z0, line_z1 );

  int min_idx = -1;
  float min_dist = FLT_MAX;

  if ( dx < radius ) {
    min_idx = 0;
    min_dist = t_dist(ray_pos, px);
  }

  if ( dy < radius && t_dist(ray_pos, py) < min_dist ) {
    min_idx = 1;
    min_dist = t_dist(ray_pos, py);
  }

  if ( dz < radius && t_dist(ray_pos, pz) < min_dist) {
    min_idx = 2;
    min_dist = t_dist(ray_pos, pz);
  }

  return min_idx == 0 ? OHDL_X : 
         min_idx == 1 ? OHDL_Y : 
         min_idx == 2 ? OHDL_Z : OHDL_NON;
}




inline ORTHO_HANDLE_ID t_PickHandleOrthoCircles
(
    const EVec3f &ray_pos,
    const EVec3f &ray_dir,
    const EVec3f &center,
    const float   length,
    const float   radius
)
{
  //x (yz平面)

  const int   RES  = 50;
  const float step = (float)(2 * M_PI / RES);

  float min_dist = FLT_MAX;
  int   min_idx  = -1;

  float d;
  EVec3f p;
  EVec3f p0, p1;
 
  //xy平面
  for (int i = 0; i < RES; ++i) 
  {
    p0 = EVec3f( length*cos(step * (i-1)), length* sin(step * (i-1)), 0) + center;
    p1 = EVec3f( length*cos(step * i    ), length* sin(step * i    ), 0) + center;

    std::tie(d,p) = t_DistRayAndLineSegm(ray_pos, ray_dir, p0, p1);
    if ( d < radius && t_dist(p, ray_pos) < min_dist ) {
      min_dist = t_dist(p, ray_pos);
      min_idx = 0;
    }
  }
   
  //yz平面
  for (int i = 0; i < RES; ++i) 
  {
    p0 = EVec3f( 0, length*cos(step * (i-1)), length* sin(step * (i-1)) ) + center;
    p1 = EVec3f( 0, length*cos(step * i    ), length* sin(step * i    ) ) + center;

    std::tie(d,p) = t_DistRayAndLineSegm(ray_pos, ray_dir, p0, p1);
    if ( d < radius && t_dist(p, ray_pos) < min_dist ) {
      min_dist = t_dist(p, ray_pos);
      min_idx = 1;
    }
  }

  //zx平面
  for (int i = 0; i < RES; ++i) 
  {
    p0 = EVec3f( length* sin(step * (i-1)), 0, length*cos(step * (i-1)) ) + center;
    p1 = EVec3f( length* sin(step * i    ), 0, length*cos(step * i    ) ) + center;

    std::tie(d,p) = t_DistRayAndLineSegm(ray_pos, ray_dir, p0, p1);
    if ( d < radius && t_dist(p, ray_pos) < min_dist ) {
      min_dist = t_dist(p, ray_pos);
      min_idx = 2;
    }
  }

  return min_idx == 0 ? OHDL_XY : 
         min_idx == 1 ? OHDL_YZ : 
         min_idx == 2 ? OHDL_ZX : OHDL_NON;
}















inline void t_DrawPolyLine(
  const EVec3f color,
  const float  width,
  const std::vector<EVec3f> &points, 
  const bool b_closed = false
)
{
  int N = (int)points.size();
  if( b_closed ) N += 1;

  int *idx = new int[N];
  for (int i = 0; i < N; ++i) idx[i] = i;
  if( b_closed ) idx[N-1] = 0;

  glColor3d(color[0],color[1],color[2]);
  glLineWidth(width);

  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, points.data());
  glDrawElements(GL_LINE_STRIP, N, GL_UNSIGNED_INT, idx);
  glDisableClientState(GL_VERTEX_ARRAY);

  delete[] idx;
}



inline void t_DrawPolyLine(
  const EVec3f  color,
  const float   width,
  const int     num_points, 
  const EVec3f *points, 
  const bool b_closed = false
)
{
  int N = num_points;
  if( b_closed ) N += 1;

  int *idx = new int[N];
  for (int i = 0; i < N; ++i) idx[i] = i;
  if( b_closed ) idx[N-1] = 0;

  glColor3d(color[0],color[1],color[2]);
  glLineWidth(width);

  glEnableClientState(GL_VERTEX_ARRAY);
  glVertexPointer(3, GL_FLOAT, 0, points );
  glDrawElements(GL_LINE_STRIP, N, GL_UNSIGNED_INT, idx);
  glDisableClientState(GL_VERTEX_ARRAY);

  delete[] idx;
}





inline void t_drawCylinder
(
  const EVec3f &p1,
  const EVec3f &p2,
  const float  R
)
{
  const int N = 20;

  EVec3f dir = (p2 - p1).normalized();
  EVec3f zAxis(0, 0, 1);
  EVec3f axis = zAxis.cross(dir).normalized();

  float theta = acos(zAxis.dot(dir));
  Eigen::AngleAxisf Rot(theta, axis);


  std::vector<EVec3f> v1(N + 1), v2(N + 1), norm(N + 1);
  for (int i = 0; i <= N; i++)
  {
    float t = i * ((float)M_PI * 2.0f / N);
    float x = cos(t);
    float y = sin(t);
    norm[i] = Rot * EVec3f(x, y, 0);
    v1[i] = Rot * EVec3f(R *x, R *y, 0) + p1;
    v2[i] = Rot * EVec3f(R *x, R *y, 0) + p2;
  }

  glBegin(GL_TRIANGLE_STRIP);
  for (int i = 0; i <= N; ++i)
  {
    glNormal3fv(norm[i].data());
    glVertex3fv(v1[i].data());
    glVertex3fv(v2[i].data());
  }
  glEnd();

}





#pragma managed

