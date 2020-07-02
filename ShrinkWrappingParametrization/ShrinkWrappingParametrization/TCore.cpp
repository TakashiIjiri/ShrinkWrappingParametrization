#include "stdafx.h"
#include "Windows.h"
#include "MainForm.h"
#include "TCore.h"
#include "COMMON/tmesh.h"

#include <string>
#include <vector>
#include <fstream>


#pragma unmanaged
using namespace ShrinkWrappingParametrization;
using namespace std;


std::vector<std::string> GetAllFiles( std::string dir, string ext)
{
  vector<string> fnames;
  string path = dir + "/*." + ext;

  WIN32_FIND_DATA fd; 
  HANDLE hFind = ::FindFirstFile( path.c_str(), &fd); 

  if ( hFind == INVALID_HANDLE_VALUE) return fnames;
 
  do {  
    if ( !(fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) ) 
    {
      fnames.push_back( fd.cFileName );
    }
  }while(::FindNextFile(hFind, &fd)); 
  ::FindClose(hFind); 

  return fnames;
}


// vtx 6146, 12288
//  
//
//






TCore::TCore()
{
  m_bL = m_bR = m_bM = false;
  vector<string> fnames = GetAllFiles("./models", "obj");
  

  for ( int i = 0; i < fnames.size(); ++i ) 
  {
    //load obj mesh
    TMesh m;
    m.initialize(("./models/" + fnames[i]).c_str());
    std::cout << i << " " << fnames[i] << " " << m.m_vSize << " " << m.m_pSize << "\n";

    //init quad mesh
    vector<EVec3f> Vs = { m.m_vVerts[0], m.m_vVerts[1], 
                          m.m_vVerts[2], m.m_vVerts[3], 
                          m.m_vVerts[4], m.m_vVerts[5], 
                          m.m_vVerts[6], m.m_vVerts[7] };   
    vector<EVec4i> Qs = { EVec4i(0,2,3,1), EVec4i(0,1,5,4), EVec4i(0,4,6,2), 
                          EVec4i(2,6,7,3), EVec4i(1,3,7,5), EVec4i(5,7,6,4) };
    TQuadMesh q;
    q.initialize(Vs, Qs);

    //shring wrap subdivision
    vector<float> offsets = q.ShrinkWrappingSubdivision(5, m);
    
    ofstream ofs( ("./models/" + fnames[i] + ".txt").c_str() );
    
    for ( int i = 0; i < 8; ++i ) 
      ofs << q.m_verts[i][0] << " " << q.m_verts[i][1] << " " << q.m_verts[i][2] << "\n";
    
    for ( auto it : offsets )
      ofs << it << " " ;
    ofs << "\n";
    ofs.close();
  }
  

  m_vis_mesh.initialize(("./models/" + fnames[35]).c_str());
  vector<EVec3f> Vs = { m_vis_mesh.m_vVerts[0], m_vis_mesh.m_vVerts[1], 
                        m_vis_mesh.m_vVerts[2], m_vis_mesh.m_vVerts[3], 
                        m_vis_mesh.m_vVerts[4], m_vis_mesh.m_vVerts[5], 
                        m_vis_mesh.m_vVerts[6], m_vis_mesh.m_vVerts[7] };   
  vector<EVec4i> Qs = { EVec4i(0,2,3,1), EVec4i(0,1,5,4), EVec4i(0,4,6,2), 
                        EVec4i(2,6,7,3), EVec4i(1,3,7,5), EVec4i(5,7,6,4) };
  
  m_quadmesh_offset = {0,0,0,0,0,0,0,0};
  m_quadmesh.initialize(Vs, Qs);
  m_num_subdiv = 0;  

}

void TCore::KeyDown(int keycode)
{
  if ( keycode == 'A') { 
    m_quadmesh_offset  = m_quadmesh.ShrinkWrappingSubdivision_SingleStep(m_vis_mesh, m_quadmesh_offset);
    ++m_num_subdiv;
    
    vector<EVec3f> vs = { m_vis_mesh.m_vVerts[0], m_vis_mesh.m_vVerts[1], 
                          m_vis_mesh.m_vVerts[2], m_vis_mesh.m_vVerts[3], 
                          m_vis_mesh.m_vVerts[4], m_vis_mesh.m_vVerts[5], 
                          m_vis_mesh.m_vVerts[6], m_vis_mesh.m_vVerts[7] };   
    vector<EVec4i> qs = { EVec4i(0,2,3,1), EVec4i(0,1,5,4), EVec4i(0,4,6,2), 
                          EVec4i(2,6,7,3), EVec4i(1,3,7,5), EVec4i(5,7,6,4) };

    m_quadmesh_regen.Initialize_ShrinkWrappingSubdivision ( vs, qs, m_num_subdiv, m_quadmesh_offset);  
  }
}





void TCore::LBtnDown ( EVec2i p, OglForCLI* ogl)
{
  m_bL = true;
  ogl->BtnDown_Trans(p);  
}
void TCore::RBtnDown ( EVec2i p, OglForCLI* ogl)
{
  m_bR = true;
  ogl->BtnDown_Rot(p);
}
void TCore::MBtnDown ( EVec2i p, OglForCLI* ogl)
{
  m_bM = true;
  ogl->BtnDown_Zoom(p);
}

void TCore::LBtnUp   ( EVec2i p, OglForCLI* ogl)
{
  m_bL = false;
  ogl->BtnUp();
}
void TCore::RBtnUp   ( EVec2i p, OglForCLI* ogl)
{
  m_bR = false;
  ogl->BtnUp();
}
void TCore::MBtnUp   ( EVec2i p, OglForCLI* ogl)
{
  m_bM = false;
  ogl->BtnUp();
}

void TCore::MouseMove( EVec2i p, OglForCLI* ogl)
{
  if ( !m_bL && !m_bR && !m_bM ) return;
  ogl->MouseMove(p);
  RedrawMainForm();
}


static float diff[4]  = {0.3f, 0.3f,0.3f,0.3f};
static float ambi[4]  = {0.3f, 0.3f,0.3f,0.3f};
static float diffB[4] = {0.5f, 0.0f,0.1f,0.3f};
static float ambiB[4] = {0.2f, 0.0f,0.1f,0.3f};
static float spec[4]  = {1.0f, 1.0f,1.0f,0.3f};
static float shin[1]  = {128.0f};


void TCore::DrawScene( OglForCLI* ogl)
{
  glDisable(GL_LIGHTING);
  glDisable(GL_BLEND);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glEnable(GL_DEPTH_TEST);

  glBegin(GL_LINES);
  {
    glColor3d ( 1, 0, 0);
    glVertex3d( 0, 0, 0);
    glVertex3d( 5, 0, 0);
    glColor3d ( 0, 1, 0);
    glVertex3d( 0, 0, 0);
    glVertex3d( 0, 5, 0);
    glColor3d ( 0, 0, 1);
    glVertex3d( 0, 0, 0);
    glVertex3d( 0, 0, 5);
  }
  glEnd();

  //draw frame
  glLineWidth(2);
  glColor3d(1,1,0);
  
  glBegin(GL_LINES);
  glVertex3d( -1, 0, -3); glVertex3d( 1, 0, -3);
  glVertex3d(  1, 0, -3); glVertex3d( 1, 2, -3);
  glVertex3d(  1, 2, -3); glVertex3d(-1, 2, -3);
  glVertex3d( -1, 2, -3); glVertex3d(-1, 0, -3);
  glVertex3d( -1, 0, +3); glVertex3d( 1, 0, +3);
  glVertex3d(  1, 0, +3); glVertex3d( 1, 2, +3);
  glVertex3d(  1, 2, +3); glVertex3d(-1, 2, +3);
  glVertex3d( -1, 2, +3); glVertex3d(-1, 0, +3);
  glEnd();
  
  //glEnable(GL_LIGHTING);
  //m_vis_mesh.draw(diff, ambi, spec, shin);
  glDisable(GL_LIGHTING);
  m_vis_mesh.DrawEdges(1,1,1,1);
  
/*
  glDisable(GL_LIGHTING);
  int piv = 24;
  glColor3d(0.5,0,0); TMesh::DrawSphere(m_vis_mesh.m_vVerts[piv+0], 0.03f);
  glColor3d(1.0,0,0); TMesh::DrawSphere(m_vis_mesh.m_vVerts[piv+1], 0.02f);
  glColor3d(0,0.5,0); TMesh::DrawSphere(m_vis_mesh.m_vVerts[piv+2], 0.02f);
  glColor3d(0,1.0,0); TMesh::DrawSphere(m_vis_mesh.m_vVerts[piv+3], 0.02f);
  glColor3d(0,0,0.5); TMesh::DrawSphere(m_vis_mesh.m_vVerts[piv+4], 0.02f);
  glColor3d(0,0,1.0); TMesh::DrawSphere(m_vis_mesh.m_vVerts[piv+5], 0.02f);
  glColor3d(0.5,0,1); TMesh::DrawSphere(m_vis_mesh.m_vVerts[piv+6], 0.02f);
  glColor3d(1.0,0,1); TMesh::DrawSphere(m_vis_mesh.m_vVerts[piv+7], 0.02f);
*/

  glEnable(GL_LIGHTING);
  m_quadmesh.draw(diffB, ambiB, spec, shin);
  glDisable(GL_LIGHTING);
  m_quadmesh.DrawEdges();

  glTranslated(-4,0,0);
  glEnable(GL_LIGHTING);
  m_quadmesh_regen.draw(diffB, ambiB, spec, shin);
  glDisable(GL_LIGHTING);
  m_quadmesh_regen.DrawEdges();
  glTranslated(-4,0,0);
  

  //m_quadmesh.DrawDebugInfo();

  //glColor3d(0,1,0);
  //glBegin(GL_LINES);
  //for( int i = 0; i < m_panel.m_vSize; ++i )
  //{ 
  //  EVec3f p = m_panel.m_vVerts[i] + 1 * m_panel.m_vNorms[i];
  //  glVertex3fv(m_panel.m_vVerts[i].data());
  //  glVertex3fv(p.data());
  //}
  //glEnd();

  //for ( int i = 8; i < 12; ++i)
  //  TMesh::DrawSphere(m_vis_mesh.m_vVerts[i], 0.02);

  /*
  glColor3d(0,1,0);
  for ( int i = 12; i < 16; ++i)
    TMesh::DrawSphere(m_vis_mesh.m_vVerts[i], 0.02);

  glColor3d(0,0,1);
  for ( int i = 16; i < 20; ++i)
    TMesh::DrawSphere(m_vis_mesh.m_vVerts[i], 0.02);


  glColor3d(1,0,1);
  for ( int i = 20; i < 26; ++i)
    TMesh::DrawSphere(m_vis_mesh.m_vVerts[i], 0.02);
  */

  
}
  
 

#pragma managed
