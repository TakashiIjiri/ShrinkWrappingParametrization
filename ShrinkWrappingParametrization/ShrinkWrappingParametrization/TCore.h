#pragma once


#include "COMMON/OglForCLI.h"
#include "COMMON/tmath.h"
#include "COMMON/tmesh.h"
#include "COMMON/tquadmesh.h"

#pragma unmanaged


class TCore
{
  bool m_bL, m_bR, m_bM;
  std::vector<float> m_quadmesh_offset;

  int m_num_subdiv;
  TQuadMesh m_quadmesh;

  //autoencoder動作確認用，元データ と 再構築データ
  std::vector<TQuadMesh> m_origmesh; 
  std::vector<TQuadMesh> m_gengmesh;

  TCore();
public:
  
  static TCore* GetInst(){
    static TCore p;
    return &p;
  }
  
  void MouseMove( EVec2i p, OglForCLI* ogl);
  void LBtnDown ( EVec2i p, OglForCLI* ogl);
  void RBtnDown ( EVec2i p, OglForCLI* ogl);
  void MBtnDown ( EVec2i p, OglForCLI* ogl);
  void LBtnUp   ( EVec2i p, OglForCLI* ogl);
  void RBtnUp   ( EVec2i p, OglForCLI* ogl);
  void MBtnUp   ( EVec2i p, OglForCLI* ogl);
  void DrawScene( OglForCLI* ogl );
  void KeyDown(int keycode);
};

#pragma managed
