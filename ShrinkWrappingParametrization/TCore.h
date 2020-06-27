#pragma once


#include "COMMON/OglForCLI.h"
#include "COMMON/tmath.h"

#pragma unmanaged



class TCore
{
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
};

#pragma managed
