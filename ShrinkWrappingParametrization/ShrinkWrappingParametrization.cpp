#include "stdafx.h"
#include <iostream>
#include "MainForm.h"

using namespace System;
using namespace ShrinkWrappingParametrization;



//1 first commit
//2 generate form 
//todo 
//3 vis opengl
//4 load objs
//5 get depthmap
//6 get depth surface 
//7 shrinkWrapping


[STAThreadAttribute]
int main()
{
  std::cout << "FormMain::getInst()->ShowDialog() \n";
  MainForm::GetInst()->ShowDialog();

  return 0;
}
