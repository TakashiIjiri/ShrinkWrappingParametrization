#include "stdafx.h"
#include "Windows.h"
#include "MainForm.h"
#include "TCore.h"
#include "COMMON/tmesh.h"

#include <string>
#include <vector>
#include <fstream>


//dll/lib file 読み込み（libtorchを使ったオートエンコーダ）
#include "./mydll/AutoencoderForCarDll.h"
#pragma comment( lib, "mydll/AutoencoderCarDll.lib" )

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




//梅谷さんのobjからShrinkWrapParameterizationを取得するコード
//obj --> 3次元*8頂点 + 6146分のoffset vector
static void ConvertObjToSWParameters(
    const TMesh m,
    vector<EVec3f> &points_8,
    vector<float>  &offset_6146)
{
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
  points_8 = points_8;
  offset_6146 = offsets;
}




// parameter 6162 = 3次元 x 8頂点 + (6146 - 8) 点分のoffset値
static void ConvertSWParametersToObj
(
  const float *param_6162, 
  TQuadMesh &q
)
{
  vector<EVec3f> vs(8);
  for ( int i=0; i < 8; ++i) 
    vs[i] << param_6162[i*3 + 0], param_6162[i*3+1], param_6162[i*3+2];

  vector<EVec4i> qs = { EVec4i(0,2,3,1), EVec4i(0,1,5,4), EVec4i(0,4,6,2), 
                        EVec4i(2,6,7,3), EVec4i(1,3,7,5), EVec4i(5,7,6,4) };

  vector<float> offsets(6146, 0);
  for ( int i = 8; i < 6146; ++i) offsets[i] = param_6162[ 3*8 + i - 8];
  q.Initialize_ShrinkWrappingSubdivision ( vs, qs, 5, offsets);  

}









void TCore::KeyDown(int keycode)
{
  if ( keycode == 'A'){
  
    float a[AE_MID_SIZE  ];
    float b[AE_INPUT_SIZE];
    for ( int i = 0; i < AE_MID_SIZE; ++i ) a[i] = (rand() / (float)RAND_MAX ) ;
    AutoencoderCar_Decode(a,b);
    ConvertSWParametersToObj(b, m_quadmesh);

    for ( int i = 0; i < 10; ++i ) std::cout << a[i] << " " ;
    std::cout << "\n";
    for ( int i = 0; i < 10; ++i ) std::cout << b[i] << " " ;
    std::cout << "\n";
    
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
static float diffR[4] = {0.8f, 0.0f,0.1f,0.3f};
static float ambiR[4] = {0.8f, 0.0f,0.1f,0.3f};
static float diffB[4] = {0.5f, 0.0f,0.1f,0.3f};
static float ambiB[4] = {0.2f, 0.0f,0.1f,0.3f};
static float diffW[4] = {0.8f, 0.8f,0.8f,0.3f};
static float ambiW[4] = {0.7f, 0.7f,0.3f,0.3f};

static float spec[4]  = {1.0f, 1.0f,1.0f,0.3f};
static float shin[1]  = {128.0f};


static void DrawFrame()
{

  glBegin(GL_LINES);
  glColor3d ( 1, 0, 0); glVertex3d( 0, 0, 0); glVertex3d( 5, 0, 0);
  glColor3d ( 0, 1, 0); glVertex3d( 0, 0, 0); glVertex3d( 0, 5, 0);
  glColor3d ( 0, 0, 1); glVertex3d( 0, 0, 0); glVertex3d( 0, 0, 5);
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
  
}




void TCore::DrawScene( OglForCLI* ogl)
{
  glDisable(GL_LIGHTING);
  glDisable(GL_BLEND);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  glEnable(GL_DEPTH_TEST);

  glEnable(GL_LIGHTING);
  m_quadmesh.draw(diffW, ambiW, spec, shin);
  glDisable(GL_LIGHTING);
  m_quadmesh.DrawEdges();


  glEnable(GL_LIGHTING);

  glPushMatrix();
  for ( int i = 0; i < (int)m_origmesh.size(); ++i)
  {
    glTranslated(-2.5,0,0);  
    m_origmesh[i].draw(diffR, ambiR, spec, shin);
    glTranslated( 0,0,-5);  
    m_gengmesh[i].draw(diffW, ambiW, spec, shin);
    glTranslated( 0,0,+5);  
  }
  glPopMatrix();
  
}
  



TCore::TCore()
{
  m_bL = m_bR = m_bM = false;

  //梅谷さん提供objをshrinkwrappingparameterizationへ変換
  //これを実施して作成したファイルをオートエンコーダに書ける
  if (false) {
    vector<string> fnames = GetAllFiles("./models", "obj");

    for (int i = 0; i < fnames.size(); ++i)
    {
      //load obj mesh
      TMesh m;
      m.initialize(("./models/" + fnames[i]).c_str());

      vector<EVec3f> points;
      vector<float > offsets;
      ConvertObjToSWParameters(m, points, offsets);

      //export parametrization
      ofstream ofs(("./models/" + fnames[i] + ".txt").c_str());

      for (auto p : points) ofs << p[0] << " " << p[1] << " " << p[2] << "\n";
      for (auto o : offsets) ofs << o << " ";
      ofs << "\n";
      ofs.close();
    }
  }


  //学習済みオートエンコーダの初期化とテスト実行
  InitializeAutoencoderCar("full.pt", "encoder.pt", "decoder.pt");
  //InitializeAutoencoderCar( string("decoder.pt").c_str() );
  float a[AE_MID_SIZE];
  float b[AE_INPUT_SIZE];
  for (int i = 0; i < AE_MID_SIZE; ++i)
    a[i] = 0.5f;
  AutoencoderCar_Decode(a, b);
  ConvertSWParametersToObj(b, m_quadmesh);


  vector<string> fnames = GetAllFiles("./models", "obj");

  const int NUM_TEST_GEN = 20;
  m_origmesh.resize(NUM_TEST_GEN);
  m_gengmesh.resize(NUM_TEST_GEN);

  for (int i = 0; i < NUM_TEST_GEN; ++i)
  {
    // obj読み込み --> quad mesh生成
    TMesh mesh;
    mesh.initialize(("./models/" + fnames[35 + i]).c_str());

    // obj --> quadmesh --> parameterization
    vector<EVec3f> Vs =
    { mesh.m_vVerts[0], mesh.m_vVerts[1], mesh.m_vVerts[2], mesh.m_vVerts[3],
      mesh.m_vVerts[4], mesh.m_vVerts[5], mesh.m_vVerts[6], mesh.m_vVerts[7] };
    vector<EVec4i> Qs = { EVec4i(0,2,3,1), EVec4i(0,1,5,4), EVec4i(0,4,6,2),
                          EVec4i(2,6,7,3), EVec4i(1,3,7,5), EVec4i(5,7,6,4) };

    m_origmesh[i].initialize(Vs, Qs);
    vector<float> offsets = m_origmesh[i].ShrinkWrappingSubdivision(5, mesh);

    // parameter --> encoder --> decoder --> quadmesh
    float *input = new float[AE_INPUT_SIZE]; //AE_INPUT_SIZE 6162
    float *output_full = new float[AE_INPUT_SIZE];
    float *output_enco = new float[AE_MID_SIZE]; //AE_MID_SIZE   20 
    float *output_deco = new float[AE_INPUT_SIZE];
    for (int i = 0; i < 8; ++i) {
      input[3 * i + 0] = Vs[i][0];
      input[3 * i + 1] = Vs[i][1];
      input[3 * i + 2] = Vs[i][2];
    }

    //6146頂点 最初の8点は読み飛ばす
    for (int i = 8; i < 6146; ++i) input[24 + i - 8] = offsets[i];
    AutoencoderCar_EncoderDecoder(input, output_full);
    AutoencoderCar_Encoder(input, output_enco);
    AutoencoderCar_Decode(output_enco, output_deco);

    for (int i = 0; i < AE_MID_SIZE; ++i) {
      std::cout << output_full[i] << " " << output_deco[i] << "     " << output_enco[i] << " done\n";
    }

    ConvertSWParametersToObj(output_full, m_gengmesh[i]);
    delete[] input;
    delete[] output_full;
    delete[] output_enco;
    delete[] output_deco;
  }
}




 

#pragma managed
