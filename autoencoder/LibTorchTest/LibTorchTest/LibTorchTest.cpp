// LibTorchTest.cpp : このファイルには 'main' 関数が含まれています。プログラム実行の開始と終了がそこで行われます。
//

#include "pch.h"
#include <iostream>
#include "torch/script.h"

using namespace std;


//プロパティを以下のサイトに従って変更
//https://medium.com/@boonboontongbuasirilai/building-pytorch-c-integration-libtorch-with-ms-visual-studio-2017-44281f9921ea
//
//言語より準拠モードをnonに
// 追加のインクルードディレクトリに以下を追加
// ./3rdparty/libtorch/include;
// ./3rdparty/libtorch/include/torch/csrc;%(AdditionalIncludeDirectories)
//追加のライブラリディレクトリに以下を追加
// ./3rdparty/libtorch/lib
//追加の依存ファイルに以下を追加
// torch_cpu.lib; c10.lib
//必要なdllをexeと同じ場所へ移動


int main()
{
  std::cout << "Hello World!\n"; 
  torch::jit::script::Module full = torch::jit::load("../../full.pt");
  torch::jit::script::Module enco = torch::jit::load("../../encoder.pt");
  torch::jit::script::Module deco = torch::jit::load("../../decoder.pt");

  //for( auto m : module.get_methods() ) { 
  //  cout << m.name();
  //  cout << "want to get multi functions";
  //} 


  at::Tensor input  = torch::ones({1, 6162});
  at::Tensor output = full.forward({input}).toTensor();

  at::Tensor output1 = enco.forward({input}).toTensor();
  at::Tensor output2 = deco.forward({output1}).toTensor();

  std::cout << "dim = [" << output.size(0) << " x "     << output.size(1) << "]\n";
  std::cout << "dim = [" << output1.size(0) << " x "     << output1.size(1) << "]\n";
  std::cout << "dim = [" << output2.size(0) << " x "     << output2.size(1) << "]\n";

  const float* out = output.data_ptr<float>();
  const float* out1 = output2.data_ptr<float>();

  for ( int i = 0; i < 6162; ++i)
  {
    cout << out[i] << " " << out1[i] << "\n";
  }

}

