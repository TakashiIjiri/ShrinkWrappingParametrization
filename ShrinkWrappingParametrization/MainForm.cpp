#include "stdafx.h"
#include "MainForm.h"

#include "TCore.h"



using namespace ShrinkWrappingParametrization;



System::Void MainForm::m_main_panel_Paint(System::Object^  sender, System::Windows::Forms::PaintEventArgs^  e){}


System::Void MainForm::m_main_panel_MouseDoubleClick(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e){}


System::Void MainForm::m_main_panel_MouseDown(
    System::Object^  sender, 
    System::Windows::Forms::MouseEventArgs^  e)
{
  if (e->Button == System::Windows::Forms::MouseButtons::Left) 
    TCore::GetInst()->LBtnDown( EVec2i(e->X, e->Y), m_ogl);
  if (e->Button == System::Windows::Forms::MouseButtons::Middle) 
    TCore::GetInst()->MBtnDown( EVec2i(e->X, e->Y), m_ogl);
  if (e->Button == System::Windows::Forms::MouseButtons::Right) 
    TCore::GetInst()->RBtnDown( EVec2i(e->X, e->Y), m_ogl);
}


System::Void MainForm::m_main_panel_MouseMove(System::Object^  sender, System::Windows::Forms::MouseEventArgs^  e)
{
  TCore::GetInst()->MouseMove( EVec2i(e->X, e->Y), m_ogl );
}


System::Void MainForm::m_main_panel_MouseUp(
    System::Object^  sender, 
    System::Windows::Forms::MouseEventArgs^  e)
{
  if (e->Button == System::Windows::Forms::MouseButtons::Left) 
    TCore::GetInst()->LBtnUp( EVec2i(e->X, e->Y), m_ogl);
  if (e->Button == System::Windows::Forms::MouseButtons::Middle) 
    TCore::GetInst()->MBtnUp( EVec2i(e->X, e->Y), m_ogl);
  if (e->Button == System::Windows::Forms::MouseButtons::Right) 
    TCore::GetInst()->RBtnUp( EVec2i(e->X, e->Y), m_ogl);
}


System::Void MainForm::m_main_panel_Resize(
    System::Object^  sender, 
    System::EventArgs^  e)
{
}







void MainForm::RedrawMainPanel()
{
  static bool isFirst = true;
/*
  if (isFirst)
  {
    isFirst = false;

    //このタイミングで 他のformを生成し, Show()も読んでOK
    initializeSingletons();
    ModeCore::getInst()->ModeSwitch(MODE_VIS_NORMAL);
  }

  EVec3f cuboid = ImageCore::GetInst()->GetCuboidF();
  float  nearDist = (cuboid[0] + cuboid[1] + cuboid[2]) / 3.0f * 0.01f;
  float  farDist = (cuboid[0] + cuboid[1] + cuboid[2]) / 3.0f * 8;


  m_ogl->OnDrawBegin(FormMainPanel->Width, FormMainPanel->Height, 45.0, nearDist, farDist);
  initializeLights();
  if (FormVisParam::getInst()->bRendFrame()) t_drawFrame(cuboid);
  ModeCore::getInst()->drawScene(cuboid, m_ogl->GetCamPos(), m_ogl->GetCamCnt());
  if (FormVisParam::getInst()->bRendIndi())
  {
    ViewIndiCore::getInst()->drawIndicator(FormMainPanel->Width, FormMainPanel->Height, m_ogl->GetCamPos(), m_ogl->GetCamCnt(), m_ogl->GetCamUp());
  }
  m_ogl->OnDrawEnd();
*/
}

