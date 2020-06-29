#include "stdafx.h"
#include "MainForm.h"

#include "TCore.h"



using namespace ShrinkWrappingParametrization;

#pragma comment( lib, "opengl32.lib" )
#pragma comment( lib, "glu32.lib" )
#pragma comment( lib, "gdi32.lib" )
#pragma comment( lib, "User32.lib" )



MainForm::MainForm()
{
  InitializeComponent();

  m_ogl = new OglForCLI(GetDC((HWND)m_main_panel->Handle.ToPointer()));
  m_ogl->SetCam(EVec3f(0,0,5), EVec3f(0,0,0), EVec3f(0,1,0));
  m_ogl->SetBgColor(0.3f, 0.3f, 0.3f, 0.5f);

}


System::Void MainForm::m_main_panel_Paint(
    System::Object^  sender, 
    System::Windows::Forms::PaintEventArgs^  e)
{
  this->RedrawMainPanel();
}


System::Void MainForm::m_main_panel_MouseDoubleClick(
    System::Object^  sender, 
    System::Windows::Forms::MouseEventArgs^  e)
{ 
}


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
  this->RedrawMainPanel();
}





static void initializeLights()
{
  GLfloat pos0 [4] = {0,0, 3000,1};
  GLfloat pos1 [4] = {0,0,-3000,1};
  GLfloat pos2 [4] = {3000,-3000, 3000,1};
  GLfloat dir0 [3] = {0,0,-1};
  GLfloat dir1 [3] = {0,0, 1};
  GLfloat dir2 [3] = {-1,1,-1};


  GLfloat ambi0[3] = {0.5f,0.5f,0.5f};
  GLfloat ambi1[3] = {0,0,0};
  GLfloat ambi2[3] = {0,0,0};
  GLfloat diff0[3] = {0.5f,0.5f,0.5f};
  GLfloat diff1[3] = {0.5f,0.5f,0.5f};
  GLfloat diff2[3] = {0.5f,0.5f,0.5f};
  GLfloat spec0[3] = {0.3f,0.3f,0.3f};
  GLfloat spec1[3] = {0.3f,0.3f,0.3f};
  GLfloat spec2[3] = {0.3f,0.3f,0.3f};

  glEnable(GL_LIGHT0);
  glLightfv(GL_LIGHT0, GL_POSITION, pos0);
  glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, dir0 );
  glLightf( GL_LIGHT0, GL_SPOT_CUTOFF,  180.0 );
  glLightf( GL_LIGHT0, GL_SPOT_EXPONENT, 0 );
  glLightfv(GL_LIGHT0, GL_AMBIENT , ambi0);
  glLightfv(GL_LIGHT0, GL_DIFFUSE , diff0);
  glLightfv(GL_LIGHT0, GL_SPECULAR, spec0);

  glEnable(GL_LIGHT1);
  glLightfv(GL_LIGHT1, GL_POSITION, pos1);
  glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, dir1 );
  glLightf( GL_LIGHT1, GL_SPOT_CUTOFF,  180.0 );
  glLightf( GL_LIGHT1, GL_SPOT_EXPONENT, 0 );
  glLightfv(GL_LIGHT1, GL_AMBIENT , ambi1);
  glLightfv(GL_LIGHT1, GL_DIFFUSE , diff1);
  glLightfv(GL_LIGHT1, GL_SPECULAR, spec1);
  
  glEnable(GL_LIGHT2);
  glLightfv(GL_LIGHT2, GL_POSITION, pos2);
  glLightfv(GL_LIGHT2, GL_SPOT_DIRECTION, dir2 );
  glLightf( GL_LIGHT2, GL_SPOT_CUTOFF,  180.0 );
  glLightf( GL_LIGHT2, GL_SPOT_EXPONENT, 0 );

  glLightfv(GL_LIGHT2, GL_AMBIENT , ambi2);
  glLightfv(GL_LIGHT2, GL_DIFFUSE , diff2);
  glLightfv(GL_LIGHT2, GL_SPECULAR, spec2);
}


//(x,y,z) = 2m 2m 6m
void MainForm::RedrawMainPanel()
{
  // unit : [m]
  float  nearDist = 0.1f;
  float  farDist  = 25.0f;

  m_ogl->OnDrawBegin ( m_main_panel->Width, m_main_panel->Height, 45.0, nearDist, farDist);

  initializeLights();
  TCore::GetInst()->DrawScene(m_ogl);

  m_ogl->OnDrawEnd();
}


System::Void MainForm::MainForm_KeyDown(System::Object^  sender, System::Windows::Forms::KeyEventArgs^  e) 
{
  TCore::GetInst()->KeyDown((int)e->KeyCode);
  this->RedrawMainPanel();
}
