#pragma once


#include "COMMON/OglForCLI.h"
#include "COMMON/tmath.h"
#include "COMMON/tmesh.h"
#include "COMMON/tquadmesh.h"

#pragma unmanaged



class TCore
{
  bool m_bL, m_bR, m_bM;

  TMesh m_vis_mesh, m_panel;

  std::vector<float> m_quadmesh_offset;
  TQuadMesh m_quadmesh;
  TCore();
public:
  
  static TCore* GetInst(){
    static TCore p;
    return &p;
  }
  
  void MouseMove(EVec2i p, OglForCLI* ogl);
  void LBtnDown(EVec2i p, OglForCLI* ogl);
  void RBtnDown(EVec2i p, OglForCLI* ogl);
  void MBtnDown(EVec2i p, OglForCLI* ogl);
  void LBtnUp(EVec2i p, OglForCLI* ogl);
  void RBtnUp(EVec2i p, OglForCLI* ogl);
  void MBtnUp(EVec2i p, OglForCLI* ogl);
  void DrawScene( OglForCLI* ogl );
  void KeyDown(int keycode);
};

#pragma managed
